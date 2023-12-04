import pickle
from tqdm import tqdm

from jax import jacfwd

from copy import deepcopy

import optax

from Optimizer import *

import sys

from optimizers.EmbeddedPhysics import predict_static, get_correction_term_probability_flow, \
    get_correction_term_reverse_sde, eval_backward_rand, eval_backward, save_simulation_video
from utils.utils import get_times_log, get_times_linear, log_time_to_network, get_rand_times_log, get_rand_times_linear, \
    create_default_update_fn_aux, load_model_weights, count_weights, save_model_weights

sys.path.append("..")

from models.ModelLoader import *
from physics import forward_step_custom, get_forward_full_custom
from utils.plots import *

class SlicedScoreMatching(Optimizer):

    def save_data(self):

        data_time_dict = {}

        params = deepcopy(self.params)

        keys = [key[1] for key in self.test_generator.keys]

        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:

            params['diffphylog']['time_format'] = 'linear'
            params['diffphylog']['num_steps'] = n

            if not self.params['diffphylog']['forward_sim']:
                forward_fn = self.forward_fn
            else:
                forward_fn = lambda *args: 0.5 * self.forward_fn(*args)

            data_n_0 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      jax.random.PRNGKey(0), version=0, realizations=1)
            data_n_1 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      jax.random.PRNGKey(0), version=1, realizations=1)
            data_n_3 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      jax.random.PRNGKey(0), version=3, realizations=1)
            data_n_6 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      jax.random.PRNGKey(0), version=6, realizations=1)
            data_n_7 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      jax.random.PRNGKey(0), version=7, realizations=1)

            data_time_dict[n] = {0: data_n_0, 1: data_n_1, 3: data_n_3, 6: data_n_6, 7: data_n_7}

        with open(f'saves/{wandb.run.id}.p', 'wb') as file:

            pickle.dump(data_time_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, data_keys, rng, realizations=8, probability_flow=True, **kwargs):

        t0 = self.params['t0']
        t1 = self.params['t1']
        num_steps = self.params['diffphylog']['num_steps']

        if self.params['diffphylog']['time_format'] == 'log':
            sim_times = get_times_log(t0, t1, num_steps)
            log_time = True
        elif self.params['diffphylog']['time_format'] == 'linear':
            sim_times = get_times_linear(t0, t1, num_steps)
            log_time = False
        else:
            raise ValueError('Unknown time format ', self.params['diffphylog']['time_format'])

        if not self.params['diffphylog']['forward_sim']:
            inference_correction_probability_flow_fn = get_correction_term_probability_flow(self.forward_fn,
                                                                                            self.params['diffphylog'][
                                                                                                'inference'][
                                                                                                'noise'] / jnp.sqrt(2))
            inference_correction_reverse_sde_fn = get_correction_term_reverse_sde(self.forward_fn,
                                                                                  self.params['diffphylog'][
                                                                                      'inference']['noise'] / jnp.sqrt(
                                                                                      2))
        else:
            inference_correction_probability_flow_fn = get_correction_term_probability_flow(self.forward_fn,
                                                                                            self.params['diffphylog'][
                                                                                                'inference']['noise'])
            inference_correction_reverse_sde_fn = get_correction_term_reverse_sde(self.forward_fn,
                                                                                  self.params['diffphylog'][
                                                                                      'inference']['noise'])

        noise = self.params['diffphylog']['inference']['noise']

        if probability_flow:
            return self.predict_(data_keys, rng, inference_correction_probability_flow_fn, (sim_times, log_time), noise,
                                 realizations=realizations, sde=False)
        else:
            return self.predict_(data_keys, rng, inference_correction_reverse_sde_fn, (sim_times, log_time), noise,
                                 realizations=realizations, sde=True)

    def predict_(self, data_keys, rng, correction_reverse_sde_fn_, times, noise, sde=True, realizations=8):

        sim_times, log_time = times
        correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(self.model_weights, state, t)
        update_step = self.params['diffphylog']['update_step']

        data_elems = [(self.test_generator.load((list(self.test_generator.h5files.keys())[0], key), transform=False)[
                           ..., 0] - self.params['data_mean']) / self.params['data_std'] for key in data_keys]

        forward_noise = []
        forward_clean = []
        single_times = jnp.array([self.params['t0'], self.params['t1']])
        for data in data_elems:
            forward_noise.append(self.forward_full_custom_fn(data, sim_times, rng=rng,
                                                             noise=self.params['diffphylog']['training']['noise'])[-1])
            forward_clean.append(self.forward_full_custom_fn(data, single_times, rng=rng, noise=0.0)[-1])
            rng, _ = jax.random.split(rng)

        forward_noise = jnp.stack(forward_noise)
        data_elems = jnp.stack(data_elems)

        return_ = {}

        for i in range(realizations):
            if sde:
                prediction = eval_backward_rand(forward_noise, correction_reverse_sde_fn, sim_times[::-1], rng, noise,
                                                log_time=log_time, update_step=update_step)[-1]
            else:
                prediction = eval_backward(forward_noise, correction_reverse_sde_fn, sim_times[::-1], log_time=log_time,
                                           update_step=update_step)[-1]

            rng, _ = jax.random.split(rng)

            return_[f'prediction_{i}'] = list(prediction)

        return_['ground_truth'] = list(data_elems)
        return_['input'] = list(forward_noise)
        return_['forward'] = forward_clean

        return return_

    def log_eval_statistics(self, log_dict):

        fft_normalized, fft_power = self.eval_power_spectrum(probability_flow=True)

        log_dict['fft_normalized_probability_flow'] = [wandb.Image(im) for im in fft_normalized]

        log_dict['fft_power_probability_flow'] = [wandb.Image(im) for im in fft_power]

        fft_normalized, fft_power = self.eval_power_spectrum(probability_flow=False)

        log_dict['fft_normalized_reverse_sde'] = [wandb.Image(im) for im in fft_normalized]

        log_dict['fft_power_reverse_sde'] = [wandb.Image(im) for im in fft_power]

        return log_dict

    def gradient_forward_fn_manual_zero(self, probability_flow_correction_fn):

        physics_value_fn = forward_step_custom(self.params['data_shape'])

        if self.params['diffphylog']['time_format'] == 'log':
            sim_time_to_network_fn = lambda t: log_time_to_network(t, self.params['t0'], self.params['t1'])
        elif self.params['diffphylog']['time_format'] == 'linear':
            sim_time_to_network_fn = lambda t: t
        else:
            raise ValueError('Unknown time format ', self.params['diffphylog']['time_format'])

        def model_loss_grad_1(params, t_sim_train, state, target, rng):

            batch_loss = 0.0
            backprop = []

            batch_size = state.shape[0]

            err_ = probability_flow_correction_fn(params, jnp.expand_dims(target, axis=1),
                                                  jnp.array(sim_time_to_network_fn(t_sim_train[0])).tile(batch_size))

            batch_loss += jnp.mean(jnp.square(err_))

            return batch_loss

        def model_loss_grad_2(params, t_sim_train, state, target, rng):

            batch_loss = 0.0
            backprop = []

            batch_size = state.shape[0]

            for t0, t1 in zip(t_sim_train, t_sim_train[1:]):
                delta_t = t1 - t0

                P = physics_value_fn(state, delta_t)

                in_ = jnp.stack([state, P / delta_t], axis=1)

                err_ = probability_flow_correction_fn(params, in_,
                                                      jnp.array(sim_time_to_network_fn(t0)).tile(batch_size))

            batch_loss += jnp.mean(jnp.square(err_))

            return batch_loss

        if self.params['diffphylog']['update_step'] == 1:
            return jax.jit(model_loss_grad_1)
        elif self.params['diffphylog']['update_step'] == 2:
            return jax.jit(model_loss_grad_2)
        else:
            raise ValueError('Unknown update step ', self.params['diffphylog']['update_step'])

    def gradient_forward_fn_manual(self, probability_flow_correction_fn, training_noise):

        if self.params['ssm']['time_format'] == 'log':
            sim_time_to_network_fn = lambda t: log_time_to_network(t, self.params['t0'], self.params['t1'])
        elif self.params['ssm']['time_format'] == 'linear':
            sim_time_to_network_fn = lambda t: t
        else:
            raise ValueError('Unknown time format ', self.params['ssm']['time_format'])

        def mul_v(params, rng, v, target, t):

            n = self.forward_fn(params, rng, jnp.transpose(target, (1, 0, 2, 3)), jnp.tile(t, (target.shape[1],)), rng)

            s = jnp.einsum('bwh, bwh ->', n, v)

            return s

        def model_loss(params, t_sim_train, target, rng):

            v = jr.rademacher(rng, target[0].shape)

            # print('shapes: ', target.shape, t_sim_train.shape)

            grad_x = jacfwd(mul_v, argnums=(3,))(params, rng, v, target, t_sim_train)

            # print('grad_x ', grad_x[0].shape)

            reduced_x = jnp.einsum('bwh, bwh ->', grad_x[0][0], v)  # jnp.transpose(v, (0,2,1)))

            # print('rediced_x ', reduced_x.shape)

            norm = jnp.sum(self.forward_fn(params, rng, jnp.transpose(target, (1, 0, 2, 3)),
                                           jnp.tile(t_sim_train, (target.shape[1],)), rng) ** 2)

            print('norm ', norm.shape)

            rng, _ = jax.random.split(rng)

            return 0.5 * norm + reduced_x, 0.0

        return jax.jit(model_loss)

    def gradient_backward_fn_manual(self, probability_flow_correction_fn, training_noise):

        physics_value_fn = forward_step_custom(self.params['data_shape'])

        if self.params['diffphylog']['time_format'] == 'log':
            sim_time_to_network_fn = lambda t: log_time_to_network(t, self.params['t0'], self.params['t1'])
        elif self.params['diffphylog']['time_format'] == 'linear':
            sim_time_to_network_fn = lambda t: t
        else:
            raise ValueError('Unknown time format ', self.params['diffphylog']['time_format'])

        def model_loss_grad_1(params, t_sim_train, state, target, rng):

            t_sim_batch = t_sim_train[0]

            batch_loss = 0.0
            backprop = []

            batch_size = state.shape[0]

            for target_state, t in zip(target[1:], t_sim_train[1:]):

                correction = probability_flow_correction_fn(params, jnp.expand_dims(state, axis=1),
                                                            jnp.array(sim_time_to_network_fn(t_sim_batch)).tile(
                                                                batch_size))

                state_forward = state + physics_value_fn(state, (t_sim_batch - t))

                state = 2 * state - state_forward

                state = state - (t_sim_batch - t) * correction

                t_sim_batch = t

                batch_loss += jnp.mean(jnp.square(state - target_state))

                if training_noise > 0:
                    state += jnp.sqrt(t_sim_batch - t) * jax.random.normal(rng, shape=state.shape) * training_noise
                    rng, _ = jax.random.split(rng)

            return batch_loss, state

        def model_loss_grad_2(params, t_sim_train, state, target, rng):

            t_sim_batch = t_sim_train[0]

            batch_loss = 0.0
            backprop = []

            batch_size = state.shape[0]

            for target_state, t in zip(target[1:], t_sim_train[1:]):

                P = physics_value_fn(state, (t_sim_batch - t))

                in_ = jnp.stack([state, P / (t_sim_batch - t)], axis=1)

                correction = probability_flow_correction_fn(params, in_,
                                                            jnp.array(sim_time_to_network_fn(t_sim_batch)).tile(
                                                                batch_size))

                state_forward = state + P

                state = 2 * state - state_forward

                state = state - (t_sim_batch - t) * correction

                t_sim_batch = t

                batch_loss += jnp.mean(jnp.square(state - target_state))

                if training_noise > 0:
                    state += jnp.sqrt(t_sim_batch - t) * jax.random.normal(rng, shape=state.shape) * training_noise
                    rng, _ = jax.random.split(rng)

            return batch_loss, state

        if self.params['diffphylog']['update_step'] == 1:
            return jax.jit(model_loss_grad_1)
        elif self.params['diffphylog']['update_step'] == 2:
            return jax.jit(model_loss_grad_2)
        else:
            raise ValueError('Unknown update step ', self.params['diffphylog']['update_step'])

    def train_inner(self, log_dict, epoch_, opt, opt_state, rng, log_video_every=1):

        print_every_batch = 1
        save_model_every = log_video_every

        video_key = self.params['train_key']
        sample_key = self.params['train_key']
        train_key = self.params['train_key']
        loader_key = self.params['loader_key']

        DT = self.params['step_size']
        t1 = self.params['t1']
        t0 = self.params['t0']

        simulations_per_epoch = int(self.params['simulations_per_epoch'])

        num_steps = self.params['diffphylog']['num_steps']

        time_steps_per_batch = self.params['diffphylog']['time_steps_per_batch']

        num_batches = min(len(self.train_generator), simulations_per_epoch)

        for epoch in range(epoch_):

            print(f'Epoch: {epoch}/{epoch_}')

            for batch_n in tqdm(range(num_batches)):

                data = self.train_generator.__getitem__(batch_n, forward=False)[:, :, :, 0]

                if self.params['ssm']['time_format'] == 'log':
                    sim_times = get_rand_times_log(rng, t0, t1, num_steps)
                    rng, _ = jax.random.split(rng)
                elif self.params['ssm']['time_format'] == 'linear':
                    sim_times = get_rand_times_linear(rng, t0, t1, num_steps)
                else:
                    raise ValueError('Unknown time format ', self.params['ssm']['time_format'])

                full_forward_simulation = self.forward_full_custom_fn(data, sim_times, rng=rng,
                                                                      noise=self.params['ssm']['training']['noise'])
                rng, _ = jax.random.split(rng)

                # Forward
                forward_loss_avg = 0.0
                length_simulation = len(full_forward_simulation)

                batch_loss = 0

                for i in range(length_simulation):

                    target = jnp.array(full_forward_simulation[i:i + 1])
                    sim_times_batch = jnp.array(sim_times[i:i + 1])

                    self.model_weights, opt_state, batch_loss, state = self.grad_update_forward(self.model_weights,
                                                                                                opt_state,
                                                                                                [sim_times_batch,
                                                                                                 target, rng], rng=rng)
                    rng, _ = jax.random.split(rng)

                    forward_loss_avg += batch_loss / len(target)

                    if batch_n % print_every_batch == 0:
                        print(f'{epoch:3d} batch {batch_n:4d} forward loss: {jnp.abs(batch_loss):.10f}')

            rng, _ = jax.random.split(rng)

            self.train_generator.on_epoch_end()

            log_dict['forward_loss'] = forward_loss_avg / num_batches

            forward_loss_avg = 0

            if (epoch % log_video_every) == 0:

                if self.params['ssm']['time_format'] == 'log':
                    vid_times = get_times_log(t0, t1, num_steps)
                    log_time = True
                elif self.params['ssm']['time_format'] == 'linear':
                    vid_times = get_times_linear(t0, t1, num_steps)
                    log_time = False
                else:
                    raise ValueError('Unknown time format ', self.params['ssm']['time_format'])

                log_dict = self.log_eval_statistics(log_dict)

                video_elem = (self.test_generator.load((list(self.test_generator.h5files.keys())[0], '00000'),
                                                       transform=False)[..., 0] - self.params['data_mean']) / \
                             self.params['data_std']

                log_dict = save_simulation_video(log_dict, self.model_weights,
                                                 self.inference_correction_probability_flow_fn,
                                                 self.inference_correction_reverse_sde_fn, video_key, video_elem[None],
                                                 vid_times, self.params['diffphylog']['inference']['noise'],
                                                 'diffusion_1', log_time=log_time,
                                                 update_step=self.params['diffphylog']['update_step'])

                log_dict = save_simulation_video(log_dict, self.model_weights,
                                                 self.inference_correction_probability_flow_fn,
                                                 self.inference_correction_probability_flow_fn, video_key,
                                                 video_elem[None], vid_times,
                                                 self.params['diffphylog']['inference']['noise'],
                                                 'diffusion_1_probability_flow', log_time=log_time,
                                                 update_step=self.params['diffphylog']['update_step'])

                video_elem = (self.test_generator.load((list(self.test_generator.h5files.keys())[0], '00001'),
                                                       transform=False)[..., 0] - self.params['data_mean']) / \
                             self.params['data_std']

                log_dict = save_simulation_video(log_dict, self.model_weights,
                                                 self.inference_correction_probability_flow_fn,
                                                 self.inference_correction_reverse_sde_fn, video_key, video_elem[None],
                                                 vid_times, self.params['diffphylog']['inference']['noise'],
                                                 'diffusion_2', log_time=log_time,
                                                 update_step=self.params['diffphylog']['update_step'])

                log_dict = save_simulation_video(log_dict, self.model_weights,
                                                 self.inference_correction_probability_flow_fn,
                                                 self.inference_correction_probability_flow_fn, video_key,
                                                 video_elem[None], vid_times,
                                                 self.params['diffphylog']['inference']['noise'],
                                                 'diffusion_2_probability_flow', log_time=log_time,
                                                 update_step=self.params['diffphylog']['update_step'])

                keys = ['00000', '00001']

                prediction_data = self.predict(keys, self.params['sample_key'], probability_flow=False)

                log_dict['prediction'] = [wandb.Image(im) for im in plot_pictures(prediction_data)]

            if (epoch % save_model_every) == 0:
                save_model_weights(self.model_weights, f'weights/{wandb.run.id}_{epoch:04d}.p')

            wandb.log(log_dict)

        return opt_state, rng

    def train(self):

        print('Noise for training set at ', self.params['ssm']['training']['noise'])

        print('**************************')

        if self.params['ssm']['inference_is_training']:
            print('Inference settings are the same as training settings!')
            self.params['ssm']['inference'] = self.params['ssm']['training']

        print('Noise for inference set at ', self.params['ssm']['inference']['noise'])

        opt_lr = lambda lr: optax.chain(
            optax.clip(10),
            optax.zero_nans(),
            optax.scale_by_adam(b1=0.9, b2=0.999),
            optax.scale(step_size=-lr))

        opt = optax.inject_hyperparams(opt_lr)(lr=self.params['lr'])

        opt_state = opt.init(self.model_weights)

        print('Optimizer hyperparams ', opt_state.hyperparams)

        print('Learning rate: ', opt_state.hyperparams['lr'])

        self.training_correction_probability_flow_fn = jax.jit(
            get_correction_term_probability_flow(self.forward_fn, self.params['ssm']['training']['noise']))
        self.training_correction_reverse_sde_fn = jax.jit(
            get_correction_term_reverse_sde(self.forward_fn, self.params['ssm']['training']['noise']))

        self.inference_correction_probability_flow_fn = jax.jit(
            get_correction_term_probability_flow(self.forward_fn, self.params['ssm']['inference']['noise']))
        self.inference_correction_reverse_sde_fn = jax.jit(
            get_correction_term_reverse_sde(self.forward_fn, self.params['ssm']['inference']['noise']))

        training_noise = self.params['ssm']['training']['noise']

        self.model_loss_forward_fn = self.gradient_forward_fn_manual(self.training_correction_probability_flow_fn,
                                                                     training_noise)

        self.grad_update_forward = jax.jit(create_default_update_fn_aux(opt, self.model_loss_forward_fn))

        _global_step = 0

        shape = (self.params['resolution'], self.params['resolution'])
        self.forward_full_custom_fn = jax.jit(get_forward_full_custom(shape))

        rng = jax.random.PRNGKey(self.params['seed'])

        log_video_every = 1

        if not self.params['test_only']:
            current_step = self.params['start_epoch']

            log_dict = {}
            log_dict['epoch'] = current_step
            log_dict['lr'] = self.params['lr']

            opt_state, rng = self.train_inner(log_dict, self.params['epochs'], opt, opt_state, rng,
                                              log_video_every=log_video_every)

        log_dict = {}

        log_dict = self.log_eval_statistics(log_dict)

        test_mse_probability_flow = self.eval_reconstruction(probability_flow=True)

        print(f'Test MSE probability flow: {test_mse_probability_flow}')

        wandb.summary['test_mse_probability_flow'] = test_mse_probability_flow

        test_mse_reverse_sde = self.eval_reconstruction(probability_flow=False)

        print(f'Test MSE reverse sde: {test_mse_reverse_sde}')

        wandb.summary['test_mse_reverse_sde'] = test_mse_reverse_sde

        print(f'Saving Data....')

        self.save_data()

        wandb.log(log_dict)

    def build_model(self):

        if self.params['diffphylog']['update_step'] == 2:
            two_step = True
        else:
            two_step = False

        forward_fn, init_params = get_model(self.params['data_shape'], time_dim=True, two_step=two_step)

        if self.params['network_weights']:
            init_params = load_model_weights(self.params['network_weights'])
            print('Loading parameters from ', self.params['network_weights'])
        else:
            print('Starting from random initialization')

        print(f'model weights: {count_weights(init_params)}')
        self.model_weights = init_params
        self.forward_fn = jax.jit(forward_fn)

        self.cached_forward_clean = None
        self.cached_forward_noise = None
