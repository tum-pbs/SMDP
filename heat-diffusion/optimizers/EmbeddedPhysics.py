import pickle

import optax

from optimizers.Optimizer import Optimizer

from tqdm import tqdm
from copy import deepcopy
import wandb

import jax
import jax.random as jr

import sys

from utils.utils import get_times_log, get_times_linear, log_time_to_network, count_weights, load_model_weights, \
    create_default_update_fn_aux, create_default_update_fn, save_model_weights, get_rand_times_log, \
    get_rand_times_linear

sys.path.append("..")

from models.ModelLoader import get_model
from physics import forward_full, forward_step_custom, get_forward_full_custom
from utils.plots import *


def predict_static(generator, params, forward_fn, model_weights, data_keys, rng, realizations=8, version=0,
                   noise_start=0.01, update_step=1):
    inference_correction_probability_flow_fn = get_correction_term_probability_flow(forward_fn,
                                                                                    params['embedded_physics'][
                                                                                        'inference'][
                                                                                        'noise'])
    inference_correction_reverse_sde_fn = get_correction_term_reverse_sde(forward_fn,
                                                                          params['embedded_physics']['inference'][
                                                                              'noise'])

    noise = params['embedded_physics']['inference']['noise']

    correction_reverse_sde_fn = lambda state, t: inference_correction_reverse_sde_fn(model_weights, state, t)

    correction_probability_flow_fn = lambda state, t: inference_correction_probability_flow_fn(model_weights, state, t)

    data_elems = [
        (generator.load((list(generator.h5files.keys())[0], key), transform=False)[..., 0] - params['data_mean']) /
        params['data_std'] for key in data_keys]

    t0 = params['t0']
    t1 = params['t1']
    num_steps = params['embedded_physics']['num_steps']
    if params['embedded_physics']['time_format'] == 'log':
        times = get_times_log(t0, t1, num_steps)[::-1]
        log_time = True
    elif params['embedded_physics']['time_format'] == 'linear':
        times = get_times_linear(t0, t1, num_steps)[::-1]
        log_time = False
    else:
        raise ValueError('Unknown time format ', params['embedded_physics']['time_format'])

    forward_ = [forward_full(data, params['t1'], params['t1'])[-1] for data in data_elems]
    forward_noise = []

    # for data in data_elems:
    #     forward_noise.append(eval_forward(data, times[::-1], rng = rng, noise = noise)[-1])
    #     rng, _ = jax.random.split(rng)

    for elem in forward_:
        elem = elem + noise_start * jr.normal(rng, shape=elem.shape)
        forward_noise.append(elem)
        rng, _ = jax.random.split(rng)

    forward_noise = jnp.stack(forward_noise)
    data_elems = jnp.stack(data_elems)

    return_ = {}

    for i in range(realizations):

        if version == 0:  # Reverse-sde solution

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, noise, log_time=log_time,
                                      update_step=update_step)[-1]
        elif version == 1:  # Probability flow solution (different weighting of score) + noise

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_probability_flow_fn, times, rng, noise,
                                      log_time=log_time,
                                      update_step=update_step)[-1]

        elif version == 2:  # Reverse-sde solution + Predictor-Corrector

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, noise, corrector_steps=100,
                                      log_time=log_time, update_step=update_step)[-1]

        elif version == 3:  # Probability flow solution

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_probability_flow_fn, times, rng, 0.0, log_time=log_time,
                                      update_step=update_step)[-1]

        elif version == 4:  # Reverse-sde solution with increased noise (2x)

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, 2 * noise,
                                      log_time=log_time,
                                      update_step=update_step)[-1]

        elif version == 5:  # Reverse SDE solution with decreased noise (0.5x)

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, 0.5 * noise,
                                      log_time=log_time,
                                      update_step=update_step)[-1]

        elif version == 6:  # Reverse SDE solution, split update (first physics update, then correction)

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, noise, log_time=log_time,
                                      update_step=3)[-1]

        elif version == 7:  # Reverse SDE solution, no noise

            prediction = \
                eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, 0.0, log_time=log_time,
                                      update_step=3)[-1]

        else:

            raise ValueError('Not implemented')

        rng, _ = jax.random.split(rng)

        return_[f'prediction_{i}'] = list(prediction)

    return_['ground_truth'] = list(data_elems)
    return_['input'] = list(forward_noise)
    return_['forward'] = list(forward_)

    return return_


def eval_backward_rand_pc(initial_value, correction_fn, times, rng, noise, corrector_steps=0, epsilon=2e-7,
                          log_time=False, update_step=1):
    if log_time:
        time_to_network_fn = lambda t: log_time_to_network(t, times[-1], times[0])
    else:
        time_to_network_fn = lambda t: t

    initial_shape = initial_value.shape

    y = initial_value

    physics_value_fn = forward_step_custom(initial_shape)

    y_list = [y]

    # Predictor steps
    for t1, t0 in tqdm(zip(times, times[1:])):

        delta_t = t1 - t0

        if update_step == 1:

            P = physics_value_fn(y, delta_t)
            C = correction_fn(jnp.expand_dims(y, axis=1), jnp.tile(jnp.array(time_to_network_fn(t1)), y.shape[0]))

            y = y - P - delta_t * C

        elif update_step == 2:

            P = physics_value_fn(y, delta_t)

            in_ = jnp.stack([y, P / delta_t], axis=1)

            C = correction_fn(in_, jnp.tile(jnp.array(time_to_network_fn(t1)), y.shape[0]))

            y = y - P - delta_t * C

        elif update_step == 3:

            P = physics_value_fn(y, delta_t)
            C = correction_fn(jnp.expand_dims(y, axis=1), jnp.tile(jnp.array(time_to_network_fn(t1)), y.shape[0]))

            y = y - P - (delta_t / 2) * C

        else:
            raise ValueError('Unknown update step ', update_step)

        # Corrector step

        if t0 > times[-1]:

            for _ in range(corrector_steps):

                bm = jnp.sqrt(2 * epsilon) * jax.random.normal(rng, shape=y.shape)
                rng, _ = jax.random.split(rng)

                if update_step == 1:
                    C = correction_fn(jnp.expand_dims(y, axis=1),
                                      jnp.tile(jnp.array(time_to_network_fn(t0)), y.shape[0]))
                elif update_step == 2:
                    in_ = jnp.stack([y, P / delta_t], axis=1)
                    C = correction_fn(in_, jnp.tile(jnp.array(time_to_network_fn(t0)), y.shape[0]))

                y = y - epsilon * C
                y = y + bm

            bm = noise * jax.random.normal(rng, shape=y.shape) * jnp.sqrt(delta_t)
            rng, _ = jax.random.split(rng)
            y = y + bm

            if update_step == 3:
                C = correction_fn(jnp.expand_dims(y, axis=1), jnp.tile(jnp.array(time_to_network_fn(t1)), y.shape[0]))

                y = y - (delta_t / 2) * C

        y_list.append(y)

    return y_list


def eval_forward(initial_state, times, noise=0.0, rng=jax.random.PRNGKey(0)):
    _forward = [initial_state]

    shape = initial_state.shape

    physics_value_fn = forward_step_custom(shape)

    for t1, t0 in zip(times[1:], times):
        delta_t = t1 - t0

        state = _forward[-1]

        state = state + physics_value_fn(state, delta_t)

        state = state + jnp.sqrt(delta_t) * noise * jax.random.normal(rng, shape=state.shape)
        rng, _ = jax.random.split(rng)

        _forward.append(state)

    return _forward


def eval_backward(initial_state, correction_fn, times, correction_coefficient=1.0, log_time=False, update_step=1):
    # times must be decreasing

    shape = initial_state.shape

    start_t = times[-1]
    end_t = times[0]

    if log_time:
        time_to_network_fn = lambda t: log_time_to_network(t, start_t, end_t)
    else:
        time_to_network_fn = lambda t: t

    physics_value_fn = forward_step_custom(shape)

    state = initial_state

    _forward = [initial_state]

    for t1, t0 in zip(times[1:], times):

        delta_t = t0 - t1
        state = _forward[-1]

        if update_step == 1:

            correction = correction_fn(jnp.expand_dims(state, axis=1),
                                       jnp.tile(jnp.array(time_to_network_fn(t0)), state.shape[0]))

            state_forward = state + physics_value_fn(state, delta_t)

            state = 2 * state - state_forward

            state = state - delta_t * correction_coefficient * correction

        elif update_step == 2:

            P = physics_value_fn(state, delta_t)

            in_ = jnp.stack([state, P / delta_t], axis=1)

            correction = correction_fn(in_, jnp.tile(jnp.array(time_to_network_fn(t0)), state.shape[0]))

            state_forward = state + P

            state = 2 * state - state_forward

            state = state - delta_t * correction_coefficient * correction

        else:
            raise ValueError('Unknown update step ', update_step)

        _forward.append(state)

    return _forward


def eval_backward_rand(initial_value, correction_fn, times, rng, noise, log_time=False, update_step=1):
    # times must be decreasing

    start_t = times[-1]
    end_t = times[0]

    if log_time:
        time_to_network_fn = lambda t: log_time_to_network(t, start_t, end_t)
    else:
        time_to_network_fn = lambda t: t

    initial_shape = initial_value.shape

    y0 = initial_value

    physics_value_fn = forward_step_custom(initial_shape)

    y = y0

    y_list = [y]

    for t1, t0 in zip(times[1:], times):

        delta_t = t0 - t1

        if update_step == 1:

            correction = correction_fn(jnp.expand_dims(y, axis=1),
                                       jnp.tile(jnp.array(time_to_network_fn(t0)), y.shape[0]))

            state_forward = y + physics_value_fn(y, delta_t)

            y = 2 * y - state_forward

            y = y - delta_t * correction

        elif update_step == 2:

            P = physics_value_fn(y, delta_t)

            in_ = jnp.stack([y, P / delta_t], axis=1)

            state_forward = y + P

            correction = correction_fn(in_, jnp.tile(jnp.array(time_to_network_fn(t0)), y.shape[0]))

            y = 2 * y - state_forward

            y = y - delta_t * correction

        else:
            raise ValueError('Unknown update step ', update_step)

        bm = noise * jax.random.normal(rng, shape=y.shape) * jnp.sqrt(delta_t)
        rng, _ = jax.random.split(rng)

        if t1 > times[-1]:
            y = y + bm

        y_list.append(y)

    return y_list


def get_correction_term_reverse_sde(forward_fn, noise):
    rng = jax.random.PRNGKey(0)

    def correction_term_reverse_sde(params, state, t_sim_batch):
        correction = forward_fn(params, rng, state, t_sim_batch, rng)

        correction_ = - (noise ** 2) * correction

        return correction_

    return correction_term_reverse_sde


def get_correction_term_probability_flow(forward_fn, noise):
    correction_term = get_correction_term_reverse_sde(forward_fn, noise)
    f = lambda params, state, t: 0.5 * correction_term(params, state, t)

    return f


def save_simulation_video(log_dict, model_weights, correction_probability_flow_fn_, correction_reverse_sde_fn_, rng,
                          init_data, times, noise, savename, log_time=False, update_step=1):
    correction_probability_flow_fn = lambda state, t: correction_probability_flow_fn_(model_weights, state, t)
    correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(model_weights, state, t)

    states_forward_0_coeff = eval_forward(init_data, times, rng=rng, noise=noise)
    rng, _ = jax.random.split(rng)

    ground_truth = states_forward_0_coeff

    vmin = np.min(states_forward_0_coeff)
    vmax = np.max(states_forward_0_coeff)

    data_dict = {}

    for i in range(9):
        backward_init = states_forward_0_coeff[-1]
        rng, _ = jax.random.split(rng)

        states_backward_rand = eval_backward_rand(backward_init, correction_reverse_sde_fn, times[::-1], rng, noise,
                                                  log_time=log_time, update_step=update_step)

        data_dict[i] = states_backward_rand

    backward_init = states_forward_0_coeff[-1]

    states_backward_0_coeff = eval_backward(backward_init, correction_probability_flow_fn, times[::-1],
                                            correction_coefficient=0.0, log_time=log_time, update_step=update_step)
    states_backward_1_coeff = eval_backward(backward_init, correction_probability_flow_fn, times[::-1],
                                            correction_coefficient=1.0, log_time=log_time, update_step=update_step)

    def get_image(idx, title, data_dict, no_correction, probability_flow, ground_truth):

        height = 6
        width = 12
        dpi = 100

        fig = plt.figure(figsize=(width, height))
        fig.set_dpi(dpi)

        gs = GridSpec(6, 12, figure=fig)

        rand_ax_dict = {}

        rand_ax_dict[0] = fig.add_subplot(gs[0:2, 0:2])
        rand_ax_dict[1] = fig.add_subplot(gs[0:2, 2:4])
        rand_ax_dict[2] = fig.add_subplot(gs[0:2, 4:6])

        rand_ax_dict[3] = fig.add_subplot(gs[2:4, 0:2])
        rand_ax_dict[4] = fig.add_subplot(gs[2:4, 2:4])
        rand_ax_dict[5] = fig.add_subplot(gs[2:4, 4:6])

        rand_ax_dict[6] = fig.add_subplot(gs[4:6, 0:2])
        rand_ax_dict[7] = fig.add_subplot(gs[4:6, 2:4])
        rand_ax_dict[8] = fig.add_subplot(gs[4:6, 4:6])

        for i in range(9):
            rand_ax_dict[i].imshow(data_dict[i][idx][0], cmap='jet', vmin=vmin, vmax=vmax)
            make_axes_invisible(rand_ax_dict[i])

        ax_probability_flow = fig.add_subplot(gs[0:3, 6:9])
        make_axes_invisible(ax_probability_flow)
        ax_no_correction = fig.add_subplot(gs[3:6, 6:9])
        make_axes_invisible(ax_no_correction)

        ax_probability_flow.imshow(probability_flow[idx][0], cmap='jet', vmin=vmin, vmax=vmax)
        ax_probability_flow.text(0.1, 0.9, 'probability flow', color='white', ha='left', va='center',
                                 transform=ax_probability_flow.transAxes, fontsize='large')

        ax_no_correction.imshow(no_correction[idx][0], cmap='jet', vmin=vmin, vmax=vmax)
        ax_no_correction.text(0.1, 0.9, 'no correction', color='white', ha='left', va='center',
                              transform=ax_no_correction.transAxes, fontsize='large')

        ax_gt = fig.add_subplot(gs[0:3, 9:12])
        make_axes_invisible(ax_gt)
        ax_init_state = fig.add_subplot(gs[3:6, 9:12])
        make_axes_invisible(ax_init_state)

        ax_gt.imshow(ground_truth[-1][0], cmap='jet', vmin=vmin, vmax=vmax)
        ax_init_state.imshow(ground_truth[0][0], cmap='jet', vmin=vmin, vmax=vmax)

        plt.suptitle(title)

        plt.tight_layout()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

        plt.close(fig)

        return image.reshape(dpi * height, dpi * width, 3)

    images = []

    for i, time in zip(range(len(times)), times[::-1]):
        images.append(
            get_image(i, f'Backward - {time: .3f}', data_dict, states_backward_0_coeff, states_backward_1_coeff,
                      ground_truth))

    for i in range(10):
        images.append(images[-1])

    # Forward Simulation 

    data_dict_forward = {}

    for i in range(9):
        data_dict_forward[i] = eval_forward(data_dict[i][-1], times)

    states_backward_0_coeff_forward = eval_forward(states_backward_0_coeff[-1], times)
    states_backward_1_coeff_forward = eval_forward(states_backward_1_coeff[-1], times)

    for i, time in zip(range(len(times)), times):
        images.append(get_image(i, f'Forward - {time: .3f}', data_dict_forward,
                                states_backward_0_coeff_forward, states_backward_1_coeff_forward, ground_truth))

    for i in range(10):
        images.append(images[-1])

    clip = mp.ImageSequenceClip(images, fps=8)

    clip.write_videofile(f'videos/{wandb.run.id}_{savename}.mp4', fps=8)

    log_dict[savename] = wandb.Video(f'videos/{wandb.run.id}_{savename}.mp4')

    return log_dict


def get_sim_time_to_network_fn(time_format, t0, t1):
    if time_format == 'log':
        return lambda t: log_time_to_network(t, t0, t1)
    elif time_format == 'linear':
        return lambda t: t
    else:
        raise ValueError('Unknown time format ', time_format)


class EmbeddedPhysics(Optimizer):

    def __init__(self, generator, params):
        super().__init__(generator, params)

        self.forward_full_custom_fn = None
        self.cached_forward_noise = None
        self.cached_forward_clean = None
        self.training_correction_reverse_sde_fn = None
        self.training_correction_probability_flow_fn = None

    def save_data(self):

        data_time_dict = {}

        params = deepcopy(self.params)

        keys = [key[1] for key in self.test_generator.keys]

        rng = jr.PRNGKey(0)

        for num_steps in [8, 16, 32, 64, 128]:

            params['embedded_physics']['time_format'] = 'linear'
            params['embedded_physics']['num_steps'] = num_steps

            if not self.params['embedded_physics']['forward_sim']:
                forward_fn = self.forward_fn
            else:
                def forward_fn(*args):
                    return 0.5 * self.forward_fn(*args)

            data_n_0 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      rng, version=0, realizations=1)
            data_n_1 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      rng, version=1, realizations=1)
            data_n_3 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      rng, version=3, realizations=1)
            data_n_6 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      rng, version=6, realizations=1)
            data_n_7 = predict_static(self.test_generator, params, forward_fn, self.model_weights, keys,
                                      rng, version=7, realizations=1)

            data_time_dict[num_steps] = {0: data_n_0, 1: data_n_1, 3: data_n_3, 6: data_n_6, 7: data_n_7}

        with open(f'saves/{wandb.run.id}.p', 'wb') as file:

            pickle.dump(data_time_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, data_keys, rng, realizations=8, probability_flow=True, **kwargs):

        t0 = self.params['t0']
        t1 = self.params['t1']
        num_steps = self.params['embedded_physics']['num_steps']

        if self.params['embedded_physics']['time_format'] == 'log':
            sim_times = get_times_log(t0, t1, num_steps)
            log_time = True
        elif self.params['embedded_physics']['time_format'] == 'linear':
            sim_times = get_times_linear(t0, t1, num_steps)
            log_time = False
        else:
            raise ValueError('Unknown time format ', self.params['embedded_physics']['time_format'])

        if not self.params['embedded_physics']['forward_sim']:
            inference_correction_probability_flow_fn = get_correction_term_probability_flow(self.forward_fn,
                                                                                            self.params[
                                                                                                'embedded_physics'][
                                                                                                'inference'][
                                                                                                'noise'] / jnp.sqrt(2))
            inference_correction_reverse_sde_fn = get_correction_term_reverse_sde(self.forward_fn,
                                                                                  self.params['embedded_physics'][
                                                                                      'inference']['noise'] / jnp.sqrt(
                                                                                      2))
        else:
            inference_correction_probability_flow_fn = get_correction_term_probability_flow(self.forward_fn,
                                                                                            self.params[
                                                                                                'embedded_physics'][
                                                                                                'inference']['noise'])
            inference_correction_reverse_sde_fn = get_correction_term_reverse_sde(self.forward_fn,
                                                                                  self.params['embedded_physics'][
                                                                                      'inference']['noise'])

        noise = self.params['embedded_physics']['inference']['noise']

        if probability_flow:
            return self.predict_(data_keys, rng, inference_correction_probability_flow_fn, (sim_times, log_time), noise,
                                 realizations=realizations, sde=False)
        else:
            return self.predict_(data_keys, rng, inference_correction_reverse_sde_fn, (sim_times, log_time), noise,
                                 realizations=realizations, sde=True)

    def predict_(self, data_keys, rng, correction_reverse_sde_fn_, times, noise, sde=True, realizations=8):

        sim_times, log_time = times
        correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(self.model_weights, state, t)
        update_step = self.params['embedded_physics']['update_step']

        data_elems = [(self.test_generator.load((list(self.test_generator.h5files.keys())[0], key), transform=False)[
                           ..., 0] - self.params['data_mean']) / self.params['data_std'] for key in data_keys]

        forward_noise = []
        forward_clean = []
        single_times = jnp.array([self.params['t0'], self.params['t1']])
        for data in data_elems:
            forward_noise.append(self.forward_full_custom_fn(data, sim_times, rng=rng,
                                                             noise=self.params['embedded_physics']['training'][
                                                                 'noise'])[-1])
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

        if self.params['embedded_physics']['time_format'] == 'log':
            sim_time_to_network_fn = lambda t: log_time_to_network(t, self.params['t0'], self.params['t1'])
        elif self.params['embedded_physics']['time_format'] == 'linear':
            sim_time_to_network_fn = lambda t: t
        else:
            raise ValueError('Unknown time format ', self.params['embedded_physics']['time_format'])

        def model_loss_grad_1(params, t_sim_train, state, *args):
            """
            Applies an L2 regularization to the network outputs
            Parameters
            ----------
            params : model weights
            t_sim_train : simulation times
            state : current state
            args

            Returns
            -------

            """

            batch_size = state.shape[0]

            err_ = probability_flow_correction_fn(params, jnp.expand_dims(state, axis=1),
                                                  jnp.tile(jnp.array(sim_time_to_network_fn(t_sim_train[0])),
                                                           batch_size))

            batch_loss = jnp.mean(jnp.square(err_))

            return batch_loss

        def model_loss_grad_2(params, t_sim_train, state, *args):
            """
            Applies an L2 regularization to the network outputs (network inputs include physics update)
            Parameters
            ----------
            params : model weights
            t_sim_train : simulation times
            state : current state
            args

            Returns
            -------

            """

            batch_loss = 0.0
            backprop = []

            batch_size = state.shape[0]

            t0 = t_sim_train[0]
            t1 = t_sim_train[1]

            delta_t = t1 - t0

            P = physics_value_fn(state, delta_t)

            in_ = jnp.stack([state, P / delta_t], axis=1)

            err_ = probability_flow_correction_fn(params, in_,
                                                  jnp.tile(jnp.array(sim_time_to_network_fn(t0)), batch_size))

            batch_loss = jnp.mean(jnp.square(err_))

            return batch_loss

        if self.params['embedded_physics']['update_step'] == 1:
            return jax.jit(model_loss_grad_1)
        elif self.params['embedded_physics']['update_step'] == 2:
            return jax.jit(model_loss_grad_2)
        else:
            raise ValueError('Unknown update step ', self.params['embedded_physics']['update_step'])

    def gradient_forward_fn_manual(self, probability_flow_correction_fn, training_noise):

        physics_value_fn = forward_step_custom(self.params['data_shape'])

        sim_time_to_network_fn = get_sim_time_to_network_fn(self.params['embedded_physics']['time_format'],
                                                            self.params['t0'], self.params['t1'])

        @jax.jit
        def model_loss_grad(params, t_sim_train, state, target, rng):
            """
            Computes L2 distance between predicted simulation trajectory and target states
            Parameters
            ----------
            params : model weights
            t_sim_train : simulation times
            state : initial state
            target : target states
            rng : jr.PRNGKey

            Returns
            -------

            """

            t_sim_batch = t_sim_train[0]

            batch_loss = 0.0

            batch_size = state.shape[0]

            for target_state, t in zip(target[1:], t_sim_train[1:]):

                P = physics_value_fn(state, (t - t_sim_batch))

                if self.params['embedded_physics']['update_step'] == 1:
                    correction = probability_flow_correction_fn(params, jnp.expand_dims(state, axis=1),
                                                                jnp.tile(jnp.array(sim_time_to_network_fn(t_sim_batch)),
                                                                         batch_size))
                elif self.params['embedded_physics']['update_step'] == 2:
                    in_ = jnp.stack([state, P / (t - t_sim_batch)], axis=1)
                    correction = probability_flow_correction_fn(params, in_,
                                                                jnp.tile(jnp.array(sim_time_to_network_fn(t_sim_batch)),
                                                                         batch_size))
                else:
                    raise ValueError('Unknown update step ', self.params['embedded_physics']['update_step'])

                state = state + P + (t - t_sim_batch) * correction

                batch_loss += jnp.mean(jnp.square(state - target_state))

                if training_noise > 0:
                    state += jnp.sqrt(t - t_sim_batch) * jax.random.normal(rng, shape=state.shape) * training_noise
                    rng, _ = jax.random.split(rng)

                t_sim_batch = t

            return batch_loss, state

        return model_loss_grad

    def gradient_backward_fn_manual(self, probability_flow_correction_fn, training_noise):

        physics_value_fn = forward_step_custom(self.params['data_shape'])

        sim_time_to_network_fn = get_sim_time_to_network_fn(self.params['embedded_physics']['time_format'],
                                                            self.params['t0'], self.params['t1'])

        @jax.jit
        def model_loss_grad(params, t_sim_train, state, target, rng):

            t_sim_batch = t_sim_train[0]

            batch_loss = 0.0

            batch_size = state.shape[0]

            for target_state, t in zip(target[1:], t_sim_train[1:]):

                P = physics_value_fn(state, (t_sim_batch - t))

                if self.params['embedded_physics']['update_step'] == 1:
                    correction = probability_flow_correction_fn(params, jnp.expand_dims(state, axis=1),
                                                                jnp.tile(jnp.array(sim_time_to_network_fn(t_sim_batch)),
                                                                         batch_size))

                elif self.params['embedded_physics']['update_step'] == 2:

                    in_ = jnp.stack([state, P / (t_sim_batch - t)], axis=1)
                    correction = probability_flow_correction_fn(params, in_,
                                                                jnp.tile(jnp.array(sim_time_to_network_fn(t_sim_batch)),
                                                                         batch_size))

                else:

                    raise ValueError('Unknown update step ', self.params['embedded_physics']['update_step'])

                state_forward = state + P

                state = 2 * state - state_forward

                state = state - (t_sim_batch - t) * correction

                batch_loss += jnp.mean(jnp.square(state - target_state))

                if training_noise > 0:
                    state += jnp.sqrt(t_sim_batch - t) * jax.random.normal(rng, shape=state.shape) * training_noise
                    rng, _ = jax.random.split(rng)

                t_sim_batch = t

            return batch_loss, state

        return model_loss_grad

    def train_inner(self, log_dict, epoch, opt_state, current_rollout, rng, log_video_every=5):
        """
        Train for a single epoch on the training data with sliding window size <current_rollout>
        Parameters
        ----------
        log_dict : dictionary where to log
        epoch : current epoch
        opt_state : optimizer state
        current_rollout : current sliding window size
        rng : jr.PRNGKey
        log_video_every : log video every <log_video_every> epochs

        Returns
        -------

        """

        save_model_every = log_video_every

        video_key = self.params['train_key']

        t1 = self.params['t1']
        t0 = self.params['t0']

        forward_pass = self.params['embedded_physics']['forward_sim']
        backward_pass = self.params['embedded_physics']['backward_sim']

        simulations_per_epoch = int(self.params['simulations_per_epoch'])

        num_steps = self.params['embedded_physics']['num_steps']

        cut_gradient_after = min(self.params['embedded_physics']['CUT_GRADIENT'], current_rollout)

        gradient_cuts = int(jnp.ceil(current_rollout / cut_gradient_after))

        num_batches = min(len(self.train_generator), simulations_per_epoch)

        pbar = tqdm(range(num_batches))

        forward_loss_epoch_avg = 0.0
        backward_loss_epoch_avg = 0.0

        for batch_n in pbar:

            data = self.train_generator.__getitem__(batch_n, forward=False)[:, :, :, 0]

            if self.params['embedded_physics']['time_format'] == 'log':
                sim_times = get_rand_times_log(rng, t0, t1, num_steps)
            elif self.params['embedded_physics']['time_format'] == 'linear':
                sim_times = get_rand_times_linear(rng, t0, t1, num_steps)
            else:
                raise ValueError('Unknown time format ', self.params['embedded_physics']['time_format'])

            rng, _ = jax.random.split(rng)

            full_forward_simulation = self.forward_full_custom_fn(data, sim_times, rng=rng,
                                                                  noise=self.params['embedded_physics']['training'][
                                                                      'noise'])
            rng, _ = jax.random.split(rng)
            length_simulation = len(full_forward_simulation)

            forward_loss_batch_avg = 0.0

            if forward_pass:

                # just do an L2 regularization on the network outputs
                if self.params['embedded_physics']['zero_forward']:

                    target = full_forward_simulation

                    for i in range(length_simulation):
                        self.model_weights, opt_state, batch_loss = (
                            self.grad_update_forward_zero(self.model_weights, opt_state,
                                                          [sim_times[i:i + 2], target[i], target[i], rng], rng=rng))
                        rng, _ = jax.random.split(rng)
                        forward_loss_batch_avg += batch_loss

                #
                else:

                    for i in range(length_simulation):
                        state = None

                        for k in range(gradient_cuts):

                            min_q = i + k * cut_gradient_after
                            if min_q >= length_simulation - 1:
                                break

                            max_q = min(length_simulation - 1, i + (k + 1) * cut_gradient_after + 1)

                            target = full_forward_simulation[min_q:max_q]
                            sim_times_batch = jnp.array(sim_times[min_q:max_q])

                            if state is None:
                                state = target[0]

                            self.model_weights, opt_state, batch_loss, state = self.grad_update_forward(
                                self.model_weights, opt_state, [sim_times_batch, state, target, rng], rng=rng)
                            rng, _ = jax.random.split(rng)

                            forward_loss_batch_avg += batch_loss / len(target)

                # show loss in pbar
                pbar.set_description(f'Forward simulation loss: {forward_loss_batch_avg / length_simulation:.10f}')

            backward_loss_batch_avg = 0.0

            if backward_pass:

                for i in range(length_simulation):

                    state = None

                    for k in range(gradient_cuts):

                        min_q = i + k * cut_gradient_after

                        if min_q >= length_simulation - 1:
                            break

                        max_q = min(length_simulation - 1, i + (k + 1) * cut_gradient_after + 1)

                        target = full_forward_simulation[::-1][min_q:max_q]

                        sim_times_batch = jnp.array(sim_times[::-1][min_q:max_q])

                        if state is None:
                            state = target[0]

                        self.model_weights, opt_state, batch_loss, state = (
                            self.grad_update_backward(self.model_weights,
                                                      opt_state,
                                                      [sim_times_batch,
                                                       state, target,
                                                       rng], rng=rng))

                        rng, _ = jax.random.split(rng)

                        backward_loss_batch_avg += batch_loss / len(target)

                backward_loss_batch_avg = backward_loss_batch_avg / length_simulation
                backward_loss_epoch_avg += backward_loss_batch_avg
                pbar.set_description(f'Backward simulation loss: {backward_loss_batch_avg:.10f}')

        rng, _ = jax.random.split(rng)

        self.train_generator.on_epoch_end()

        log_dict['ROLLOUT'] = current_rollout
        log_dict['forward_loss'] = forward_loss_epoch_avg / num_batches
        log_dict['backward_loss'] = backward_loss_epoch_avg / num_batches

        if (epoch % log_video_every) == 0:

            if self.params['embedded_physics']['time_format'] == 'log':
                vid_times = get_times_log(t0, t1, num_steps)
                log_time = True
            elif self.params['embedded_physics']['time_format'] == 'linear':
                vid_times = get_times_linear(t0, t1, num_steps)
                log_time = False
            else:
                raise ValueError('Unknown time format ', self.params['embedded_physics']['time_format'])

            log_dict = self.log_eval_statistics(log_dict)

            video_elem = (self.test_generator.load((list(self.test_generator.h5files.keys())[0], '00000'),
                                                   transform=False)[..., 0] - self.params['data_mean']) / self.params[
                             'data_std']

            log_dict = save_simulation_video(log_dict, self.model_weights,
                                             self.inference_correction_probability_flow_fn,
                                             self.inference_correction_reverse_sde_fn, video_key, video_elem[None],
                                             vid_times, self.params['embedded_physics']['inference']['noise'],
                                             'diffusion_1',
                                             log_time=log_time,
                                             update_step=self.params['embedded_physics']['update_step'])

            log_dict = save_simulation_video(log_dict, self.model_weights,
                                             self.inference_correction_probability_flow_fn,
                                             self.inference_correction_probability_flow_fn, video_key, video_elem[None],
                                             vid_times, self.params['embedded_physics']['inference']['noise'],
                                             'diffusion_1_probability_flow', log_time=log_time,
                                             update_step=self.params['embedded_physics']['update_step'])

            video_elem = (self.test_generator.load((list(self.test_generator.h5files.keys())[0], '00001'),
                                                   transform=False)[..., 0] - self.params['data_mean']) / self.params[
                             'data_std']

            log_dict = save_simulation_video(log_dict, self.model_weights,
                                             self.inference_correction_probability_flow_fn,
                                             self.inference_correction_reverse_sde_fn, video_key, video_elem[None],
                                             vid_times, self.params['embedded_physics']['inference']['noise'],
                                             'diffusion_2',
                                             log_time=log_time,
                                             update_step=self.params['embedded_physics']['update_step'])

            log_dict = save_simulation_video(log_dict, self.model_weights,
                                             self.inference_correction_probability_flow_fn,
                                             self.inference_correction_probability_flow_fn, video_key, video_elem[None],
                                             vid_times, self.params['embedded_physics']['inference']['noise'],
                                             'diffusion_2_probability_flow', log_time=log_time,
                                             update_step=self.params['embedded_physics']['update_step'])

            keys = ['00000', '00001']

            prediction_data = self.predict(keys, self.params['sample_key'], probability_flow=False)

            log_dict['prediction'] = [wandb.Image(im) for im in plot_pictures(prediction_data)]

        if (epoch % save_model_every) == 0:
            save_model_weights(self.model_weights, f'weights/{wandb.run.id}_r{current_rollout:02d}_{epoch:04d}.p')

        wandb.log(log_dict)

        return opt_state, rng

    def train(self):

        self.params['embedded_physics']['training']['noise'] = self.params['embedded_physics']['training_noise']

        print('Noise for training set at ', self.params['embedded_physics']['training']['noise'])

        print('**************************')

        if self.params['embedded_physics']['inference_is_training']:
            print('Inference settings are the same as training settings!')
            self.params['embedded_physics']['inference'] = self.params['embedded_physics']['training']

        print('Noise for inference set at ', self.params['embedded_physics']['inference']['noise'])

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
            get_correction_term_probability_flow(self.forward_fn, self.params['embedded_physics']['training']['noise']))
        self.training_correction_reverse_sde_fn = jax.jit(
            get_correction_term_reverse_sde(self.forward_fn, self.params['embedded_physics']['training']['noise']))

        self.inference_correction_probability_flow_fn = jax.jit(
            get_correction_term_probability_flow(self.forward_fn,
                                                 self.params['embedded_physics']['inference']['noise']))
        self.inference_correction_reverse_sde_fn = jax.jit(
            get_correction_term_reverse_sde(self.forward_fn, self.params['embedded_physics']['inference']['noise']))

        training_noise = self.params['embedded_physics']['training']['noise'] if self.params['embedded_physics'][
            'noise_during_training'] else 0

        self.model_loss_forward_fn = self.gradient_forward_fn_manual(self.training_correction_probability_flow_fn,
                                                                     training_noise)
        self.model_loss_forward_zero_fn = self.gradient_forward_fn_manual_zero(
            self.training_correction_probability_flow_fn)
        self.model_loss_backward_fn = self.gradient_backward_fn_manual(self.training_correction_probability_flow_fn,
                                                                       training_noise)

        self.grad_update_forward = jax.jit(create_default_update_fn_aux(opt, self.model_loss_forward_fn))

        self.grad_update_forward_zero = jax.jit(create_default_update_fn(opt, self.model_loss_forward_zero_fn))

        self.grad_update_backward = jax.jit(create_default_update_fn_aux(opt, self.model_loss_backward_fn))

        _global_step = 0
        current_step = _global_step
        images = []
        nb_epochs = self.params['embedded_physics']['ROLLOUT_epochs']

        ROLLOUT = self.params['embedded_physics']['ROLLOUT_begin']
        ROLLOUT_ADD = self.params['embedded_physics']['ROLLOUT_add']
        ROLLOUT_NUM = self.params['embedded_physics']['ROLLOUT_increase']

        forward_pass = bool(self.params['embedded_physics']['forward_sim'])

        simulations_per_epoch = int(self.params['simulations_per_epoch'])

        print(f'Forward pass activated: {forward_pass}')

        shape = (self.params['resolution'], self.params['resolution'])
        self.forward_full_custom_fn = jax.jit(get_forward_full_custom(shape))

        rng = jax.random.PRNGKey(self.params['seed'])

        current_rollout = ROLLOUT

        log_video_every = 10

        if not self.params['test_only']:

            for rollout_n in range(ROLLOUT_NUM):

                print(f'Current rollout {current_rollout}')

                for epoch in range(nb_epochs):

                    epoch_global = epoch + rollout_n * nb_epochs

                    if epoch_global < self.params['start_epoch']:
                        continue

                    log_dict = {'epoch': epoch_global, 'lr': self.params['lr']}

                    opt_state, rng = self.train_inner(log_dict, epoch + 1, opt_state, current_rollout, rng,
                                                      log_video_every=log_video_every)

                current_rollout += ROLLOUT_ADD

            lr = self.params['lr']
            for epoch in range(1, self.params['epochs'] + 1):

                if epoch % self.params['finetune_lr_steps'] == 0:
                    lr = lr * self.params['finetune_lr_gamma']

                    print('Decreasing learning rate to ', lr)

                    opt_state.hyperparams['lr'] = lr

                epoch_global = epoch + ROLLOUT_NUM * nb_epochs

                if epoch_global < self.params['start_epoch']:
                    continue

                log_dict = {'epoch': epoch_global, 'lr': lr}

                opt_state, rng = self.train_inner(log_dict, epoch, opt_state, current_rollout, rng,
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
        wandb.finish()

    def build_model(self):

        if self.params['embedded_physics']['update_step'] == 2:
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
