import optax
from matplotlib.gridspec import GridSpec

from Optimizer import Optimizer

from models.ModelLoader import get_model
from physics import forward_full, forward_step
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import moviepy.editor as mp

import wandb

import jax
import jax.numpy as jnp

from utils.plots import make_axes_invisible, plot_pictures
from utils.utils import save_model_weights, create_default_update_fn, load_model_weights, count_weights


def eval_forward(initial_state, DT, STEPS, noise=0.0, rng=jax.random.PRNGKey(0)):
    _forward = [initial_state]

    time = 0

    shape = initial_state.shape

    physics_value_fn = forward_step(shape, DT)

    for i in range(int(STEPS)):
        state = _forward[-1]

        state = state + physics_value_fn(state)

        state = state + noise * jax.random.normal(rng, shape=state.shape)
        rng, _ = jax.random.split(rng)

        time += DT

        _forward.append(state)

    return _forward


def eval_backward(initial_state, params, correction_fn, STEPS, DT, correction_coefficient=1.0):
    time = DT * STEPS

    shape = initial_state.shape

    state = initial_state

    _forward = [initial_state]

    for i in range(int(STEPS)):
        state = _forward[-1]

        correction = correction_fn(params, jnp.expand_dims(state, axis=1), jnp.array(time).tile(state.shape[0]))

        # If correction_coefficient is 0, just do normal backwards simulation without any corrections
        state = state + correction_coefficient * DT * correction

        time -= DT

        _forward.append(state)

    return _forward


def eval_backward_rand(initial_value, params, correction_fn, DT, STEPS,
                       rng, noise):
    t0 = 0
    t1 = DT * STEPS
    args = None

    initial_shape = initial_value.shape

    y0 = initial_value

    physics_value_fn = forward_step(initial_shape, DT)

    tprev = jnp.array(t1 + DT)  # jnp.array(t1)
    tnext = jnp.array(t1)  # jnp.array(t1 - DT)

    y = y0

    y_list = [y]

    num_steps = int((t1 - t0) / DT)

    for i in tqdm(range(num_steps)):

        bm = noise * jax.random.normal(rng, shape=y.shape) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)

        y = y + DT * correction_fn(params, jnp.expand_dims(y, axis=1), tnext.tile(y.shape[0]))

        tprev = tnext
        tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)

        # see below
        # y = y + bm

        # THINK ABOUT (before or after score correction?)
        if i < num_steps - 1:
            y = y + bm

        y_list.append(y)

    return y_list


def get_score_fn(forward_fn):
    rng = jax.random.PRNGKey(0)

    def score_fn(params, state, t_sim_batch):
        return forward_fn(params, rng, state, t_sim_batch, rng)

    return jax.jit(score_fn)


def save_simulation_video(log_dict, model_weights, score_fn, rng, init_data, DT, NSTEPS, noise, savename):
    states_forward_0_coeff = eval_forward(init_data, DT, NSTEPS, rng=rng, noise=noise * jnp.sqrt(DT))
    rng, _ = jax.random.split(rng)

    ground_truth = states_forward_0_coeff

    vmin = np.min(states_forward_0_coeff)
    vmax = np.max(states_forward_0_coeff)

    data_dict = {}

    for i in range(9):
        backward_init = states_forward_0_coeff[-1]
        states_backward_rand = eval_backward_rand(backward_init, model_weights, score_fn, DT, NSTEPS, rng, noise)
        rng, _ = jax.random.split(rng)
        data_dict[i] = states_backward_rand

    backward_init = states_forward_0_coeff[-1]  # + DT * noise * jr.normal(rng, shape=states_forward_0_coeff[-1].shape)

    states_backward_0_coeff = eval_backward(backward_init, model_weights, score_fn, NSTEPS, DT,
                                            correction_coefficient=0.0)
    states_backward_1_coeff = eval_backward(backward_init, model_weights, score_fn, NSTEPS, DT,
                                            correction_coefficient=1.0)

    def get_image(idx, title, data_dict, no_correction, no_noise, ground_truth):

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

        ax_probability_flow.imshow(no_noise[idx][0], cmap='jet', vmin=vmin, vmax=vmax)
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

    print('save simulation video NSTEPS', NSTEPS)

    for i in range(NSTEPS + 1):
        images.append(
            get_image(i, f'Backward - {i * DT: .3f}', data_dict, states_backward_0_coeff, states_backward_1_coeff,
                      ground_truth))

    for i in range(10):
        images.append(images[-1])

    # Forward Simulation 

    data_dict_forward = {}

    for i in range(9):
        data_dict_forward[i] = eval_forward(data_dict[i][-1], DT, NSTEPS)

    states_backward_0_coeff_forward = eval_forward(states_backward_0_coeff[-1], DT, NSTEPS)
    states_backward_1_coeff_forward = eval_forward(states_backward_1_coeff[-1], DT, NSTEPS)

    for i in range(NSTEPS + 1):
        images.append(get_image(i, f'Forward - {NSTEPS * DT - i * DT: .3f}', data_dict_forward,
                                states_backward_0_coeff_forward, states_backward_1_coeff_forward, ground_truth))

    for i in range(10):
        images.append(images[-1])

    clip = mp.ImageSequenceClip(images, fps=8)

    clip.write_videofile(f'videos/{wandb.run.id}_{savename}.mp4', fps=8)

    log_dict[savename] = wandb.Video(f'videos/{wandb.run.id}_{savename}.mp4')

    return log_dict


class Autoregressive(Optimizer):

    def predict(self, data_keys, rng, realizations=8, **kwargs):

        DT = self.params['autoregressive']['step_size']

        score_fn = get_score_fn(self.forward_fn)

        noise = self.params['autoregressive']['inference']['noise']

        NSTEPS = int(self.params['t1'] / DT)

        return self.predict_(data_keys, rng, score_fn, DT, NSTEPS, noise, realizations=realizations, **kwargs)

    def predict_(self, data_keys, rng, score_fn, DT, NSTEPS, noise, sde=False, realizations=8):

        data_elems = [(self.test_generator.load((list(self.test_generator.h5files.keys())[0], key), transform=False)[
                           ..., 0] - self.params['data_mean']) / self.params['data_std'] for key in data_keys]

        forward_ = [forward_full(data, DT, self.params['t1'], rng=rng, noise=noise * jnp.sqrt(DT))[-1] for data in
                    data_elems]

        forward_noise = []

        forward_noise = jnp.stack(forward_)
        data_elems = jnp.stack(data_elems)

        return_ = {}

        for i in range(realizations):
            if sde:
                prediction = eval_backward_rand(forward_noise, self.model_weights, score_fn, DT, NSTEPS, rng, noise)[-1]
            else:
                prediction = eval_backward(forward_noise, self.model_weights, score_fn, NSTEPS, DT)[
                    -1]  # take final prediction

            rng, _ = jax.random.split(rng)

            return_[f'prediction_{i}'] = list(prediction)

        return_['ground_truth'] = list(data_elems)
        return_['input'] = list(forward_noise)
        return_['forward'] = forward_

        return return_

    def log_eval_statistics(self, log_dict):

        fft_normalized, fft_power = self.eval_power_spectrum(sde=True)

        log_dict['fft_normalized'] = [wandb.Image(im) for im in fft_normalized]

        log_dict['fft_power'] = [wandb.Image(im) for im in fft_power]

        fft_normalized, fft_power = self.eval_power_spectrum(sde=False)

        log_dict['fft_normalized_no_noise'] = [wandb.Image(im) for im in fft_normalized]

        log_dict['fft_power_no_noise'] = [wandb.Image(im) for im in fft_power]

        return log_dict

    def gradient_backward_fn_manual(self, score_fn, noise):

        DT = self.params['autoregressive']['step_size']

        def model_loss_grad(params, t_sim_train, state, target, rng):
            print('Tracing model loss grad...')

            t_sim_batch = t_sim_train

            batch_loss = 0.0
            backprop = []

            batch_size = state.shape[0]

            for target_state in target[1:]:
                state = state + DT * score_fn(params, jnp.expand_dims(state, axis=1),
                                              jnp.array(t_sim_batch).tile(batch_size))

                t_sim_batch -= DT

                batch_loss += jnp.mean(jnp.square(state - target_state))

                state = state + noise * jax.random.normal(rng, shape=state.shape) * jnp.sqrt(DT)
                rng, _ = jax.random.split(rng)

            return batch_loss

        return jax.jit(model_loss_grad)

    def train_inner(self, log_dict, epoch, opt, opt_state, CURRENT_ROLLOUT, rng, log_video_every=5):

        print_every_batch = 1
        save_model_every = log_video_every

        video_key = self.params['train_key']
        sample_key = self.params['train_key']
        train_key = self.params['train_key']
        loader_key = self.params['loader_key']

        DT = self.params['autoregressive']['step_size']
        t1 = self.params['t1']

        simulations_per_epoch = int(self.params['simulations_per_epoch'])

        for batch_n in tqdm(range(min(len(self.train_generator), simulations_per_epoch))):

            data = self.train_generator.__getitem__(batch_n, forward=False)[:, :, :, 0]

            full_forward_simulation = forward_full(data, DT, t1, rng=rng,
                                                   noise=self.params['autoregressive']['training']['noise'] * jnp.sqrt(DT))
            rng, _ = jax.random.split(rng)
            n_steps = len(full_forward_simulation) - CURRENT_ROLLOUT

            t_sim_train = jnp.array(t1)
            backward_loss_avg = 0.0

            batch_loss = 0

            rng, _ = jax.random.split(rng)

            for i in range(n_steps):
                target = full_forward_simulation[::-1][i:i + CURRENT_ROLLOUT + 1]

                state = target[0]

                self.model_weights, opt_state, batch_loss = self.grad_update_backward(self.model_weights, opt_state,
                                                                                      [t_sim_train, state, target, rng],
                                                                                      rng=rng)
                rng, _ = jax.random.split(rng)

                backward_loss_avg += batch_loss / CURRENT_ROLLOUT

                t_sim_train -= DT

            if batch_n % print_every_batch == 0:
                print(f'{epoch:3d} batch {batch_n:4d} backward loss: {jnp.abs(batch_loss / CURRENT_ROLLOUT):.10f}')

        if epoch % 1 == 0:
            print(f'{epoch:3d} backward loss: {backward_loss_avg / n_steps:.10f}')

        rng, _ = jax.random.split(rng)

        self.train_generator.on_epoch_end()

        log_dict['ROLLOUT'] = CURRENT_ROLLOUT

        log_dict['backward_loss'] = backward_loss_avg / n_steps

        if (epoch % log_video_every) == 0:
            NSTEPS = int(t1 / DT)

            log_dict = self.log_eval_statistics(log_dict)

            video_elem = (self.test_generator.load((list(self.test_generator.h5files.keys())[0], '00000'),
                                                   transform=False)[..., 0] - self.params['data_mean']) / self.params[
                             'data_std']

            log_dict = save_simulation_video(log_dict, self.model_weights, self.score_fn, video_key, video_elem[None],
                                             DT, NSTEPS, self.params['autoregressive']['inference']['noise'],
                                             'autoregressive_1')

            video_elem = (self.test_generator.load((list(self.test_generator.h5files.keys())[0], '00001'),
                                                   transform=False)[..., 0] - self.params['data_mean']) / self.params[
                             'data_std']

            log_dict = save_simulation_video(log_dict, self.model_weights, self.score_fn, video_key, video_elem[None],
                                             DT, NSTEPS, self.params['autoregressive']['inference']['noise'],
                                             'autoregressive_2')

            keys = ['00000', '00001']

            prediction_data = self.predict(keys, self.params['sample_key'], sde=True)

            log_dict['prediction'] = [wandb.Image(im) for im in plot_pictures(prediction_data)]

        if (epoch % save_model_every) == 0:
            save_model_weights(self.model_weights, f'weights/{wandb.run.id}_r{CURRENT_ROLLOUT:02d}_{epoch:04d}.p')

        wandb.log(log_dict)

        return opt_state, rng

    def train(self):

        print(self.params['autoregressive'])

        print('Noise for training set at ', self.params['autoregressive']['training']['noise'])

        print('**************************')

        if self.params['autoregressive']['inference_is_training']:
            print('Inference settings are same as the training settings!')
            self.params['autoregressive']['inference'] = self.params['autoregressive']['training']

        print('Noise for inference set at ', self.params['autoregressive']['inference']['noise'])

        opt_lr = lambda lr: optax.chain(
            optax.clip(10),
            optax.zero_nans(),
            optax.scale_by_adam(b1=0.9, b2=0.999),
            optax.scale(step_size=-lr))

        opt = optax.inject_hyperparams(opt_lr)(lr=self.params['lr'])

        opt_state = opt.init(self.model_weights)

        print('Optimizer hyperparams ', opt_state.hyperparams)

        print('Learning rate: ', opt_state.hyperparams['lr'])

        self.score_fn = get_score_fn(self.forward_fn)

        training_noise = self.params['autoregressive']['training']['noise'] if self.params['autoregressive'][
            'noise_during_training'] else 0

        self.model_loss_backward_fn = self.gradient_backward_fn_manual(self.score_fn, training_noise)
        self.grad_update_backward = jax.jit(create_default_update_fn(opt, self.model_loss_backward_fn))

        _global_step = 0

        nb_epochs = self.params['autoregressive']['ROLLOUT_epochs']

        ROLLOUT = self.params['autoregressive']['ROLLOUT_begin']
        ROLLOUT_ADD = self.params['autoregressive']['ROLLOUT_add']
        ROLLOUT_NUM = self.params['autoregressive']['ROLLOUT_increase']

        rng = jax.random.PRNGKey(self.params['seed'])

        current_rollout = ROLLOUT

        log_video_every = 5

        if not self.params['test_only']:

            for rollout_n in range(ROLLOUT_NUM):

                print(f'Current rollout {current_rollout}')

                for epoch in range(nb_epochs):
                    log_dict = {}
                    log_dict['epoch'] = epoch + rollout_n * nb_epochs
                    log_dict['lr'] = self.params['lr']

                    opt_state, rng = self.train_inner(log_dict, epoch, opt, opt_state, current_rollout, rng,
                                                      log_video_every=log_video_every)

                current_rollout += ROLLOUT_ADD

            lr = self.params['lr']
            for epoch in range(1, self.params['epochs'] + 1):

                log_dict = {'epoch': epoch + ROLLOUT_NUM * nb_epochs, 'lr': lr}

                if epoch % self.params['finetune_lr_steps'] == 0:
                    lr = lr * self.params['finetune_lr_gamma']

                    opt_state.hyperparams['lr'] = lr

                opt_state, rng = self.train_inner(log_dict, epoch, opt, opt_state, current_rollout, rng,
                                                  log_video_every=log_video_every)

        log_dict = {}

        log_dict = self.log_eval_statistics(log_dict)

        test_mse_no_noise = self.eval_reconstruction(sde=False)

        print(f'Test MSE no noise: {test_mse_no_noise}')

        wandb.summary['test_mse_no_noise'] = test_mse_no_noise

        test_mse = self.eval_reconstruction(sde=True)

        print(f'Test MSEe: {test_mse}')

        wandb.summary['test_mse'] = test_mse

        wandb.log(log_dict)

    def build_model(self):

        forward_fn, init_params = get_model(self.params['data_shape'], time_dim=True)

        if self.params['network_weights']:
            init_params = load_model_weights(self.params['network_weights'])
            print('Loading parameters from ', self.params['network_weights'])
        else:
            print('Starting from random initialization')

        print(f'model weights: {count_weights(init_params)}')
        self.model_weights = init_params
        self.forward_fn = forward_fn
