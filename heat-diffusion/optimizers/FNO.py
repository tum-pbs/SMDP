from Optimizer import Optimizer
from tqdm import tqdm

import jax.random as jr
import wandb

import optax

import sys

from utils.utils import create_default_update_fn, save_model_weights, count_weights

sys.path.append("..")

from models.FNO2d import *
from physics import forward_full, forward_step
from utils.plots import *


class FNO(Optimizer):

    def gradient_forward_fn(self):

        physics_forward = forward_step(self.params['data_shape'], self.params['t1'])

        def model_loss_supervised(params, state, target, rng):

            state = self.forward_fn(params, rng, jnp.expand_dims(state, axis=1)) + state
            state = state[:, 0]

            batch_loss = jnp.mean(jnp.square(state - target))

            return batch_loss

        def model_loss_physics(params, state, target, rng):

            state_ = self.forward_fn(params, rng, jnp.expand_dims(state, axis=1)) + state
            state_ = state_[:, 0]

            forward_state = state_ + physics_forward(state_)

            batch_loss = jnp.mean(jnp.square(state - forward_state))

            return batch_loss

        if self.params['fno']['loss'] == 'supervised':
            return jax.jit(model_loss_supervised)
        elif self.params['fno']['loss'] == 'physics':
            return jax.jit(model_loss_physics)
        else:
            raise ValueError(f'Loss type {self.params["loss"]} not implemented')

    def predict(self, data_keys, rng, realizations=8, **kwargs):

        data_elems = [(self.test_generator.load((list(self.test_generator.h5files.keys())[0], key), transform=False)[
                           ..., 0] - self.params['data_mean']) / self.params['data_std'] for key in data_keys]

        forward_ = [forward_full(data, self.params['t1'], self.params['t1'])[-1] for data in data_elems]

        forward_noise = []

        for elem in forward_:
            elem = elem + self.params['fno']['noise'] * jr.normal(rng, shape=elem.shape)
            forward_noise.append(elem[None])
            rng, _ = jax.random.split(rng)

        forward_noise = jnp.stack(forward_noise)
        data_elems = jnp.stack(data_elems)

        return_ = {}

        pred_fn = lambda weights, data, rng: self.forward_fn(weights, rng, data)
        pred_fn = jax.jit(pred_fn)

        for i in range(realizations):
            prediction = pred_fn(self.model_weights, forward_noise, rng) + forward_noise
            rng, _ = jax.random.split(rng)

            return_[f'prediction_{i}'] = list(prediction[:, 0])

        return_['ground_truth'] = list(data_elems)
        return_['input'] = list(forward_noise[:, 0])
        return_['forward'] = forward_

        return return_

    def train(self):

        simulations_per_epoch = self.params['simulations_per_epoch']
        batches_per_epoch = min(len(self.train_generator), simulations_per_epoch)

        opt = optax.adamw(learning_rate=self.params['lr'], weight_decay=1e-4)

        opt_state = opt.init(self.model_weights)

        _global_step = 0

        t1 = self.params['t1']

        model_loss_forward_fn = self.gradient_forward_fn()

        grad_update_forward = create_default_update_fn(opt, model_loss_forward_fn)

        rng = jax.random.PRNGKey(self.params['seed'])

        log_dict = {}

        print_every_batch = 20
        save_model_every = 10

        count_updates = 0

        for epoch in range(self.params['epochs']):

            log_dict['lr'] = self.params['lr']  # scheduler(count_updates)

            forward_loss_avg = 0

            for batch_n in tqdm(range(batches_per_epoch)):

                data = self.train_generator.__getitem__(batch_n, forward=False)[:, :, :, 0]

                full_forward_simulation = forward_full(data, t1, t1)
                target = full_forward_simulation[0]

                state = full_forward_simulation[1] + self.params['fno']['noise'] * jr.normal(rng, shape=target.shape)
                rng, _ = jax.random.split(rng)

                self.model_weights, opt_state, batch_loss = grad_update_forward(self.model_weights, opt_state,
                                                                                [state, target, rng], rng=rng)
                rng, _ = jax.random.split(rng)

                count_updates += 1

                forward_loss_avg += batch_loss

                if batch_n % print_every_batch == 0:
                    print(
                        f'{epoch:3d} batch {batch_n:4d} forward loss: {jnp.abs(forward_loss_avg / (batch_n + 1)):.10f}')

            if epoch % 1 == 0:
                print(f'{epoch:3d} forward loss: {jnp.abs(forward_loss_avg / batches_per_epoch):.10f}')

            self.train_generator.on_epoch_end()

            if self.val_generator:

                val_loss_avg = 0

                batches_validation = min(len(self.val_generator), 50)

                for batch_n in tqdm(range(batches_validation)):
                    data = self.train_generator.__getitem__(batch_n, forward=False)[:, :, :, 0]

                    full_forward_simulation = forward_full(data, t1, t1)
                    target = full_forward_simulation[0]

                    state = full_forward_simulation[1] + self.params['fno']['noise'] * jr.normal(rng,
                                                                                                 shape=target.shape)
                    rng, _ = jax.random.split(rng)

                    val_loss_avg += model_loss_forward_fn(self.model_weights, state, target, rng)
                    rng, _ = jax.random.split(rng)

                val_loss_avg = val_loss_avg / batches_validation

                log_dict['val_loss'] = val_loss_avg
            log_dict['epoch'] = epoch
            log_dict['forward_loss'] = jnp.abs(forward_loss_avg / batches_per_epoch)

            if (epoch % save_model_every) == 0 or epoch == self.params['epochs'] - 1:
                keys = ['00000', '00001']

                prediction_data = self.predict(keys, self.params['sample_key'])

                log_dict['prediction'] = [wandb.Image(im) for im in plot_pictures(prediction_data)]

                save_model_weights(self.model_weights, f'weights/{wandb.run.id}_{epoch:04d}.p')

            wandb.log(log_dict)

        # Test model

        test_mse = self.eval_reconstruction()

        print(f'Test MSE: {test_mse}')

        fft_normalized, fft_power = self.eval_power_spectrum()

        log_dict['fft_normalized'] = [wandb.Image(im) for im in fft_normalized]

        log_dict['fft_power'] = [wandb.Image(im) for im in fft_power]

        wandb.log(log_dict)

    def build_model(self):

        forward_fn, init_params = get_model(self.params['fno']['modes'], self.params['fno']['width'])
        self.model_weights = init_params
        self.forward_fn = forward_fn

        print(f'model weights: {count_weights(init_params)}')
