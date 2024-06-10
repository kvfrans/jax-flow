from localutils.debugger import enable_debug
enable_debug()

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib.pyplot as plt

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from diffusion_transformer import DiT

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 200000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')

flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
model_config = ml_collections.ConfigDict({
    # Make sure to run with Large configs when we actually want to run!
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.99,
    'hidden_size': 64, # set by preset
    'patch_size': 8, # set by preset
    'depth': 2, # set by preset
    'num_heads': 2, # set by preset
    'mlp_ratio': 1, # set by preset
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 32,
    'cfg_scale': 4.0,
    'target_update_rate': 0.9999,
    't_sampler': 'log-normal',
    't_conditioning': 1,
    'preset': 'debug',
    'use_stable_vae': 1,
})

preset_configs = {
    'debug': {
        'hidden_size': 64,
        'patch_size': 8,
        'depth': 2,
        'num_heads': 2,
        'mlp_ratio': 1,
    },
    'big': { 
        'hidden_size': 768,
        'patch_size': 2,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'semilarge': { # local-batch of 32 achieved, (16 with eps)
        'hidden_size': 1024,
        'patch_size': 2,
        'depth': 22, # Should be 24, but this fits in memory better.
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'large': { # local-batch of 2 achieved
        'hidden_size': 1024,
        'patch_size': 2,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'xlarge': {
        'hidden_size': 1152,
        'patch_size': 2,
        'depth': 28,
        'num_heads': 16,
        'mlp_ratio': 4,
    }
}

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'flow',
    'name': 'flow_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
## Model Definitions.
##############################################

# x_0 = Noise
# x_1 = Data
def get_x_t(images, eps, t):
    x_0 = eps
    x_1 = images
    t = jnp.clip(t, 0, 1-0.01) # Always include a little bit of noise.
    return (1-t) * x_0 + t * x_1

def get_v(images, eps):
    x_0 = eps
    x_1 = images
    return x_1 - x_0

class FlowTrainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    # Train
    @partial(jax.pmap, axis_name='data')
    def update(self, images, labels, pmap_axis='data'):
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)

        def loss_fn(params):
            # Sample a t for training.
            if self.config['t_sampler'] == 'log-normal':
                t = jax.random.normal(time_key, (images.shape[0],))
                t = ((1 / (1 + jnp.exp(-t))))
            elif self.config['t_sampler'] == 'uniform':
                t = jax.random.uniform(time_key, (images.shape[0],), minval=0, maxval=1)

            t_full = t[:, None, None, None] # [batch, 1, 1, 1]
            eps = jax.random.normal(noise_key, images.shape)
            x_t = get_x_t(images, eps, t_full)
            v_t = get_v(images, eps)

            if self.config['t_conditioning'] == 0:
                t = jnp.zeros_like(t)
            
            v_prime = self.model(x_t, t, labels, train=True, rngs={'label_dropout': label_key}, params=params)
            loss = jnp.mean((v_prime - v_t) ** 2)
            
            return loss, {
                'l2_loss': loss,
                'v_abs_mean': jnp.abs(v_t).mean(),
                'v_pred_abs_mean': jnp.abs(v_prime).mean(),
            }
        
        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
        info = jax.lax.pmean(info, axis_name=pmap_axis)

        updates, new_opt_state = self.model.tx.update(grads, self.model.opt_state, self.model.params)
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(step=self.model.step + 1, params=new_params, opt_state=new_opt_state)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)

        # Update the model_eps
        new_model_eps = target_update(self.model, self.model_eps, 1-self.config['target_update_rate'])
        if self.config['target_update_rate'] == 1:
            new_model_eps = new_model
        new_trainer = self.replace(rng=new_rng, model=new_model, model_eps=new_model_eps)
        return new_trainer, info

    
    @partial(jax.jit, static_argnames=('cfg'))
    def call_model(self, images, t, labels, cfg=True, cfg_val=1.0):
        if self.config['t_conditioning'] == 0:
            t = jnp.zeros_like(t)
        if not cfg:
            return self.model_eps(images, t, labels, train=False, force_drop_ids=False)
        else:
            labels_uncond = jnp.ones(labels.shape, dtype=jnp.int32) * self.config['num_classes'] # Null token
            images_expanded = jnp.tile(images, (2, 1, 1, 1)) # (batch*2, h, w, c)
            t_expanded = jnp.tile(t, (2,)) # (batch*2,)
            labels_full = jnp.concatenate([labels, labels_uncond], axis=0)
            v_pred = self.model_eps(images_expanded, t_expanded, labels_full, train=False, force_drop_ids=False)
            v_label = v_pred[:images.shape[0]]
            v_uncond = v_pred[images.shape[0]:]
            v = v_uncond + cfg_val * (v_label - v_uncond)
            return v
    
    @partial(jax.pmap, axis_name='data', in_axes=(0, 0, 0, 0), static_broadcasted_argnums=(4,5))
    def call_model_pmap(self, images, t, labels, cfg=True, cfg_val=1.0):
        return self.call_model(images, t, labels, cfg=cfg, cfg_val=cfg_val)

##############################################
## Training Code.
##############################################
def main(_):

    preset_dict = preset_configs[FLAGS.model.preset]
    for k, v in preset_dict.items():
        FLAGS.model[k] = v

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

    def get_dataset(is_train):
        print("Loading dataset")
        if 'imagenet' in FLAGS.dataset_name:
            def deserialization_fn(data):
                image = data['image']
                min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
                image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
                if 'imagenet256' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (256, 256), antialias=True)
                elif 'imagenet128' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (256, 256), antialias=True)
                else:
                    raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
                if is_train:
                    image = tf.image.random_flip_left_right(image)
                image = tf.cast(image, tf.float32) / 255.0
                image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
                return image, data['label']

            split = tfds.split_for_jax_process('train' if (is_train or FLAGS.debug_overfit) else 'validation', drop_remainder=True)
            dataset = tfds.load('imagenet2012', split=split)
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if FLAGS.debug_overfit:
                dataset = dataset.take(8)
                dataset = dataset.repeat()
                dataset = dataset.batch(local_batch_size)
            else:
                dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
                dataset = dataset.repeat()
                dataset = dataset.batch(local_batch_size)
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)
            dataset = iter(dataset)
            return dataset
        elif FLAGS.dataset_name == 'celebahq256':
            def deserialization_fn(data):
                image = data['image']
                image = tf.cast(image, tf.float32)
                image = image / 255.0
                image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
                return image,  data['label']

            dataset = tfds.load('celebahq256', split='train')
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(local_batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)
            dataset = iter(dataset)
            return dataset
        else:
            raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
        
    dataset = get_dataset(is_train=True)
    dataset_valid = get_dataset(is_train=False)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]


    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        vae_rng = flax.jax_utils.replicate(jax.random.PRNGKey(42))
        vae_encode_pmap = jax.pmap(vae.encode)
        vae_decode = jax.jit(vae.decode)
        vae_decode_pmap = jax.pmap(vae.decode)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    print("Total Memory on device:", float(jax.local_devices()[0].memory_stats()['bytes_limit']) / 1024**3, "GB")

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs.shape[-1]
    FLAGS.model.image_size = example_obs.shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
    }
    model_def = DiT(**dit_args)
    
    example_t = jnp.zeros((1,))
    example_label = jnp.zeros((1,), dtype=jnp.int32)
    model_rngs = {'params': param_key, 'label_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_label)['params']
    print("Total num of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    tx = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    model = FlowTrainer(rng, model_ts, model_ts_eps, FLAGS.model)

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)

    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network() 
        truth_fid_stats = np.load(FLAGS.fid_stats)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    jax.debug.visualize_array_sharding(model.model.params['FinalLayer_0']['Dense_0']['bias'])

    valid_images_small, valid_labels_small = next(dataset_valid)
    valid_images_small = valid_images_small[:device_count, None]
    valid_labels_small = valid_labels_small[:device_count, None]
    visualize_labels = example_labels.reshape((device_count, -1, *example_labels.shape[1:]))
    visualize_labels = visualize_labels[:, 0:1]
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()
    if FLAGS.model.use_stable_vae:
        valid_images_small = vae_encode_pmap(vae_rng, valid_images_small)

    ###################################
    # Train Loop
    ###################################

    def eval_model():
        # Needs to be in a separate function so garbage collection works correctly.

        # Validation Losses
        valid_images, valid_labels = next(dataset_valid)
        valid_images = valid_images.reshape((len(jax.local_devices()), -1, *valid_images.shape[1:])) # [devices, batch//devices, etc..]
        valid_labels = valid_labels.reshape((len(jax.local_devices()), -1, *valid_labels.shape[1:]))
        if FLAGS.model.use_stable_vae:
            valid_images = vae_encode_pmap(vae_rng, valid_images)
        _, valid_update_info = model.update(valid_images, valid_labels)
        valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
        valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
        if jax.process_index() == 0:
            wandb.log(valid_metrics, step=i)

        def process_img(img):
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img

        # Training loss on various t.
        mse_total = []
        for t in np.arange(0, 11):
            key = jax.random.PRNGKey(42)
            t = t / 10
            t_full = jnp.full((batch_images.shape), t)
            t_vector = jnp.full((batch_images.shape[0], batch_images.shape[1]), t)
            eps = jax.random.normal(key, batch_images.shape)
            x_t = get_x_t(batch_images, eps, t_full)
            v = get_v(batch_images, eps)
            pred_v = model.call_model_pmap(x_t, t_vector, batch_labels, False, 0.0)
            assert pred_v.shape == v.shape
            mse_loss = jnp.mean((v - pred_v) ** 2)
            mse_total.append(mse_loss)
            if jax.process_index() == 0:
                wandb.log({f'training_loss_t/{t}': mse_loss}, step=i)
        mse_total = jnp.array(mse_total[1:-1])
        if jax.process_index() == 0:
            wandb.log({'training_loss_t/mean': mse_total.mean()}, step=i)

        # Validation loss on various t.
        mse_total = []
        fig, axs = plt.subplots(3, 10, figsize=(30, 20))
        for t in np.arange(0, 11):
            key = jax.random.PRNGKey(42)
            t = t / 10
            t_full = jnp.full((valid_images.shape), t)
            t_vector = jnp.full((valid_images.shape[0], valid_images.shape[1]), t)
            eps = jax.random.normal(key, valid_images.shape)
            x_t = get_x_t(valid_images, eps, t_full)
            v = get_v(valid_images, eps)
            pred_v = model.call_model_pmap(x_t, t_vector, valid_labels, False, 0.0)
            assert pred_v.shape == v.shape
            mse_loss = jnp.mean((v - pred_v) ** 2)
            mse_total.append(mse_loss)
            if jax.process_index() == 0:
                wandb.log({f'validation_loss_t/{t}': mse_loss}, step=i)
        mse_total = jnp.array(mse_total[1:-1])
        if jax.process_index() == 0:
            wandb.log({'validation_loss_t/mean': mse_total.mean()}, step=i)
            plt.close(fig)

        # One-step denoising at various noise levels.
        # This only works on a TPU node with 8 devices for now...
        if len(jax.local_devices()) == 8:
            assert valid_images.shape[0] == len(jax.local_devices()) # [devices, batch//devices, etc..]
            t = jnp.arange(8) / 8 # between 0 and 0.875
            t = jnp.repeat(t[:, None], valid_images.shape[1], axis=1) # [8, batch//devices, etc..] DEVICES=8
            eps = jax.random.normal(key, valid_images.shape)
            x_t = get_x_t(valid_images, eps, t[..., None, None, None])
            v_pred = model.call_model_pmap(x_t, t, valid_labels, False, 0.0)
            x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(8, 8*3, figsize=(90, 30))
                for j in range(8):
                    for k in range(8):
                        axs[j,3*k].imshow(process_img(valid_images[j,k]), vmin=0, vmax=1)
                        axs[j,3*k+1].imshow(process_img(x_t[j,k]), vmin=0, vmax=1)
                        axs[j,3*k+2].imshow(process_img(x_1_pred[j,k]), vmin=0, vmax=1)
                wandb.log({f'reconstruction_n': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # Full Denoising with different CFG;
        key = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps = jax.random.normal(key, valid_images_small.shape) # [devices, batch//devices, etc..]
        delta_t = 1.0 / FLAGS.model.denoise_timesteps
        for cfg_scale in [0, 0.1, 1, 4, 10]:
            x = eps
            all_x = []
            for ti in range(FLAGS.model.denoise_timesteps):
                t = ti / FLAGS.model.denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                v = model.call_model_pmap(x, t_vector, visualize_labels, True, cfg_scale)
                x = x + v * delta_t
                if ti % (FLAGS.model.denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    all_x.append(np.array(x))
            all_x = np.stack(all_x, axis=2) # [devices, batch//devices, timesteps, etc..]
            all_x = all_x[:, :, -8:]

            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for j in range(8):
                    for t in range(8):
                        axs[t, j].imshow(process_img(all_x[j, 0, t]), vmin=0, vmax=1)
                    axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
                wandb.log({f'sample_cfg_{cfg_scale}': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # Denoising at different numbers of steps.
        key = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps = jax.random.normal(key, valid_images_small.shape) # [devices, batch//devices, etc..]
        delta_t = 1.0 / FLAGS.model.denoise_timesteps
        for denoise_timesteps in [1, 4, 32]:
            x = eps
            all_x = []
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                v = model.call_model_pmap(x, t_vector, visualize_labels, True, FLAGS.model.cfg_scale)
                x = x + v * delta_t
            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for j in range(8):
                    for t in range(8):
                        axs[t, j].imshow(process_img(x[j, t]), vmin=0, vmax=1)
                    axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
                wandb.log({f'sample_N/{denoise_timesteps}': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # FID calculation.
        if FLAGS.fid_stats is not None:
            activations = []
            valid_images_shape = valid_images.shape
            for fid_it in range(4096 // FLAGS.batch_size):
                _, valid_labels = next(dataset_valid)
                valid_labels = valid_labels.reshape((len(jax.local_devices()), -1, *valid_labels.shape[1:]))

                key = jax.random.PRNGKey(42 + fid_it)
                x = jax.random.normal(key, valid_images_shape)
                delta_t = 1.0 / FLAGS.model.denoise_timesteps
                for ti in range(FLAGS.model.denoise_timesteps):
                    t = ti / FLAGS.model.denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                    if FLAGS.model.cfg_scale == -1:
                        v = model.call_model_pmap(x, t_vector, valid_labels, False, 0.0)
                    else:
                        v = model.call_model_pmap(x, t_vector, valid_labels, True, FLAGS.model.cfg_scale)
                    x = x + v * delta_t
                if FLAGS.model.use_stable_vae:
                    x = vae_decode_pmap(x)
                x = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), method='bilinear', antialias=False)
                x = 2 * x - 1
                acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
                acts = jax.pmap(lambda x: jax.lax.all_gather(x, 'i', axis=0), axis_name='i')(acts)[0] # [global_devices, batch//global_devices, 2048]
                acts = np.array(acts)
                activations.append(acts)
            if jax.process_index() == 0:
                activations = np.concatenate(activations, axis=0)
                activations = activations.reshape((-1, activations.shape[-1]))
                mu1 = np.mean(activations, axis=0)
                sigma1 = np.cov(activations, rowvar=False)
                fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                wandb.log({'fid': fid}, step=i)
        
        del valid_images, valid_labels
        del all_x, x, x_t, eps
        print("Finished all the eval stuff")

    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = next(dataset)
            batch_images = batch_images.reshape((len(jax.local_devices()), -1, *batch_images.shape[1:])) # [devices, batch//devices, etc..]
            batch_labels = batch_labels.reshape((len(jax.local_devices()), -1, *batch_labels.shape[1:]))
            if FLAGS.model.use_stable_vae:
                batch_images = vae_encode_pmap(vae_rng, batch_images)

        model, update_info = model.update(batch_images, batch_labels)

        if i % FLAGS.log_interval == 0:
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0 or i == 1000:
            eval_model()

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                cp = Checkpoint(FLAGS.save_dir, parallel=False)
                cp.set_model(model_single)
                cp.save()
                del cp, model_single

if __name__ == '__main__':
    app.run(main)