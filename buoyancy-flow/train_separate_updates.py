import array
import functools as ft
import gzip
import os
import struct
import urllib.request

from copy import deepcopy

import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import einops  # https://github.com/arogozhnikov/einops
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

import pickle

import equinox as eqx
import haiku as hk

import argparse

from matplotlib.gridspec import GridSpec
import moviepy.editor as mp

import wandb

import time
import h5py
from dataloader import *
from tqdm import tqdm
import numpy as np

from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree, \
    WeaklyDiagonalControlTerm

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from phi.jax.flow import *

phi.math.backend.set_global_default_backend(phi.jax.JAX)

from physics import *
from videos import *
from utils import *
from eval import *

from functools import partial

hyperparameter_defaults = {
    'api_key': None,
    'name': None,
    'gpu': None,
    'batch_size': 1,
    'architecture': 'dilated',  # 'dilated', 'encoder_decoder'
    'architecture_params': {
        'nBlocks': 4,
        'nFeatures': 48,
        'grid': False,
    },
    'training': {
        'noise': 1.0
    },
    'inference': {
        'noise': 1.0
    },
    'forward_sim': False,
    'seed': 12345,
    't1': 0.65,
    'DT': 0.01,
    'start_epoch': 0,
    'maxTime': 65,
    'update': 1,
    'test_only': False,
    'test_file': None,
    'rollout_noise': 0.0,
    'physics_backward': 2,  # 1 for reusing forward physics, 2 for backwards time integration
    'network_weights': None,
    'phase1': {
        'simulations_per_epoch': 1000,
        'learning_rate': 1e-4,
        'ROLLOUT_epochs': 30,
        'ROLLOUT_begin': 2,
        'ROLLOUT_add': 2,
        'ROLLOUT_increases': 11,
    },
    'phase2': {
        'simulations_per_epoch': 1000,
        'epochs': 250,
        'learning_rate': 1e-4,
        'gamma_steps': 50,
        'gamma_factor': 0.5,
        'ROLLOUT': 40,
        'full_simulation': False
    },
    'time_steps_per_batch': 65,
    'NOISE_INIT_SCALE': 3.0,
}


def gradient_forward_fn_manual(probability_flow_correction_fn, simulation_metadata, update=1, rollout_noise=0.0):
    DT = simulation_metadata['DT']
    physics_value_fn = physics_forward(simulation_metadata)

    def model_loss_grad_1(params, t_sim_train, state, target, obstacles, rng):

        state_smoke, state_vel_x, state_vel_y, state_mask = state
        target_smoke, target_vel_x, target_vel_y = target

        t_sim_batch = t_sim_train

        batch_loss = 0.0
        backprop = []

        for n in range(target_smoke.shape[0] - 1):

            state_smoke, state_vel_x, state_vel_y = physics_value_fn([state_smoke, state_vel_x, state_vel_y], obstacles,
                                                                     t_sim_batch)

            correction_smoke, correction_vel_x, correction_vel_y = probability_flow_correction_fn(params, [state_smoke,
                                                                                                           state_vel_x,
                                                                                                           state_vel_y,
                                                                                                           state_mask],
                                                                                                  jnp.array(
                                                                                                      t_sim_batch))

            state_smoke = state_smoke + DT * correction_smoke
            state_vel_x = state_vel_x + DT * correction_vel_x
            state_vel_y = state_vel_y + DT * correction_vel_y

            t_sim_batch += DT

            batch_loss += jnp.mean(jnp.square(state_smoke - target_smoke[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_x - target_vel_x[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_y - target_vel_y[n + 1]))

            if rollout_noise > 0:
                print('Rollout noise active..')

                state_smoke = state_smoke + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_smoke.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_x = state_vel_x + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_x.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_y = state_vel_y + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_y.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)

            state = [state_smoke, state_vel_x, state_vel_y]

        return batch_loss, [state_smoke, state_vel_x, state_vel_y]

    def model_loss_grad_2(params, t_sim_train, state, target, obstacles, rng):

        state_smoke, state_vel_x, state_vel_y, state_mask = state
        target_smoke, target_vel_x, target_vel_y = target

        t_sim_batch = t_sim_train

        batch_loss = 0.0
        backprop = []

        for n in range(target_smoke.shape[0] - 1):

            state_smoke_forward, state_vel_x_forward, state_vel_y_forward = physics_value_fn(
                [state_smoke, state_vel_x, state_vel_y], obstacles, t_sim_batch)

            state_smoke_delta = state_smoke_forward - state_smoke
            state_vel_x_delta = state_vel_x_forward - state_vel_x
            state_vel_y_delta = state_vel_y_forward - state_vel_y

            state_smoke = state_smoke + state_smoke_delta
            state_vel_x = state_vel_x + state_vel_x_delta
            state_vel_y = state_vel_y + state_vel_y_delta

            if rollout_noise > 0:
                print('Rollout noise active..')
                state_smoke = state_smoke + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_smoke.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_x = state_vel_x + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_x.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_y = state_vel_y + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_y.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)

            correction_smoke, correction_vel_x, correction_vel_y = probability_flow_correction_fn(params, [state_smoke,
                                                                                                           state_smoke_delta,
                                                                                                           state_vel_x,
                                                                                                           state_vel_x_delta,
                                                                                                           state_vel_y,
                                                                                                           state_vel_y_delta,
                                                                                                           state_mask],
                                                                                                  jnp.array(
                                                                                                      t_sim_batch))

            state_smoke = state_smoke + DT * correction_smoke
            state_vel_x = state_vel_x + DT * correction_vel_x
            state_vel_y = state_vel_y + DT * correction_vel_y

            t_sim_batch += DT

            batch_loss += jnp.mean(jnp.square(state_smoke - target_smoke[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_x - target_vel_x[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_y - target_vel_y[n + 1]))

            state = [state_smoke, state_vel_x, state_vel_y]

        return batch_loss, [state_smoke, state_vel_x, state_vel_y]

    print('jit compiling forward loss grad')
    if update == 1:
        return jax.jit(model_loss_grad_1)
    elif update == 2:
        return jax.jit(model_loss_grad_2)
    else:
        raise ValueError('Unknown forward update ', update)


def gradient_backward_fn_manual(probability_flow_correction_fn, simulation_metadata, type=2, update=1,
                                rollout_noise=0.0):
    DT = simulation_metadata['DT']
    physics_forward_fn = physics_forward(simulation_metadata)
    physics_backward_fn = physics_backwards(simulation_metadata)

    def delta_physics_1(state, obstacles, t_sim_batch):

        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_forward, state_vel_x_forward, state_vel_y_forward = physics_forward_fn(state, obstacles,
                                                                                           t_sim_batch)

        return state_smoke_forward - state_smoke, state_vel_x_forward - state_vel_x, state_vel_y_forward - state_vel_y

    def delta_physics_2(state, obstacles, t_sim_batch):

        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_backward, state_vel_x_backward, state_vel_y_backward = physics_backward_fn(state, obstacles,
                                                                                               t_sim_batch)

        return state_smoke - state_smoke_backward, state_vel_x - state_vel_x_backward, state_vel_y - state_vel_y_backward

    if type == 1:
        delta_physics = jax.jit(delta_physics_1)
    elif type == 2:
        delta_physics = jax.jit(delta_physics_2)
    else:
        raise ValueError(f'Loss type {type} not supported!')

    def model_loss_grad_1(params, t_sim_train, state, target, obstacles, rng):

        state_smoke, state_vel_x, state_vel_y, state_mask = state
        target_smoke, target_vel_x, target_vel_y = target

        t_sim_batch = t_sim_train

        batch_loss = 0.0
        backprop = []

        for n in range(target_smoke.shape[0] - 1):

            state = [state_smoke, state_vel_x, state_vel_y]

            state_smoke_delta, state_vel_x_delta, state_vel_y_delta = delta_physics(state, obstacles, t_sim_batch)

            state_smoke = state_smoke - state_smoke_delta
            state_vel_x = state_vel_x - state_vel_x_delta
            state_vel_y = state_vel_y - state_vel_y_delta

            if rollout_noise > 0:
                print('Rollout noise active..')
                state_smoke = state_smoke + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_smoke.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_x = state_vel_x + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_x.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_y = state_vel_y + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_y.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)

            correction_smoke, correction_vel_x, correction_vel_y = probability_flow_correction_fn(params, [state_smoke,
                                                                                                           state_vel_x,
                                                                                                           state_vel_y,
                                                                                                           state_mask],
                                                                                                  jnp.array(
                                                                                                      t_sim_batch))

            state_smoke = state_smoke - DT * correction_smoke
            state_vel_x = state_vel_x - DT * correction_vel_x
            state_vel_y = state_vel_y - DT * correction_vel_y

            t_sim_batch -= DT

            batch_loss += jnp.mean(jnp.square(state_smoke - target_smoke[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_x - target_vel_x[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_y - target_vel_y[n + 1]))

        return batch_loss, [state_smoke, state_vel_x, state_vel_y]

    def model_loss_grad_2(params, t_sim_train, state, target, obstacles, rng):

        state_smoke, state_vel_x, state_vel_y, state_mask = state
        target_smoke, target_vel_x, target_vel_y = target

        t_sim_batch = t_sim_train

        batch_loss = 0.0
        backprop = []

        for n in range(target_smoke.shape[0] - 1):

            state = [state_smoke, state_vel_x, state_vel_y]

            state_smoke_delta, state_vel_x_delta, state_vel_y_delta = delta_physics(state, obstacles, t_sim_batch)

            state_smoke = state_smoke - state_smoke_delta
            state_vel_x = state_vel_x - state_vel_x_delta
            state_vel_y = state_vel_y - state_vel_y_delta

            if rollout_noise > 0.0:
                print('Rollout noise active..')
                state_smoke = state_smoke + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_smoke.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_x = state_vel_x + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_x.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                state_vel_y = state_vel_y + rollout_noise * jnp.sqrt(DT) * jax.random.normal(rng,
                                                                                             shape=state_vel_y.shape,
                                                                                             dtype=jnp.float32)
                rng, _ = jax.random.split(rng)

            correction_smoke, correction_vel_x, correction_vel_y = probability_flow_correction_fn(params, [state_smoke,
                                                                                                           state_smoke_delta,
                                                                                                           state_vel_x,
                                                                                                           state_vel_x_delta,
                                                                                                           state_vel_y,
                                                                                                           state_vel_y_delta,
                                                                                                           state_mask],
                                                                                                  jnp.array(
                                                                                                      t_sim_batch))

            state_smoke = state_smoke - DT * correction_smoke
            state_vel_x = state_vel_x - DT * correction_vel_x
            state_vel_y = state_vel_y - DT * correction_vel_y

            t_sim_batch -= DT

            batch_loss += jnp.mean(jnp.square(state_smoke - target_smoke[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_x - target_vel_x[n + 1]))
            batch_loss += jnp.mean(jnp.square(state_vel_y - target_vel_y[n + 1]))

        return batch_loss, [state_smoke, state_vel_x, state_vel_y]

    if update == 1:
        model_loss_grad = model_loss_grad_1
    elif update == 2:
        model_loss_grad = model_loss_grad_2
    else:
        raise ValueError(f'Update type {update} not supported!')

    print('jit compiling backward loss grad')
    return jax.jit(model_loss_grad)


def log_videos(log_dict, params_, model_params, inference_dict, generator):
    t1 = params_['t1']
    DT = params_['DT']
    NSTEPS = int(t1 / DT)

    video_key = params_['train_key']

    video_elem = generator.load((list(generator.h5files.keys())[0], '0'), transform=False)

    log_dict = save_simulation_video_score_decoupled(log_dict, model_params, inference_dict, video_key, video_elem, DT,
                                                     NSTEPS, params_['inference']['noise'], f'diffusion_1')

    log_dict = save_simulation_video_score_decoupled(log_dict, model_params, inference_dict, video_key, video_elem, DT,
                                                     NSTEPS, params_['inference']['noise'],
                                                     f'diffusion_1_probability_flow')

    # video_elem = generator.load((list(generator.h5files.keys())[0], '1'), transform=False)

    # log_dict = save_simulation_video_score(log_dict, model_params, inference_dict, video_key, video_elem, DT, NSTEPS, params_['inference']['noise'], f'diffusion_2')

    # log_dict = save_simulation_video_score(log_dict, model_params, inference_dict, video_key, video_elem, DT, NSTEPS, params_['inference']['noise'], f'diffusion_2_probability_flow')

    return log_dict


def train_phase_1(params_, generator, model_params, opt_state, grad_update_dict, inference_dict, rng):
    forward_pass = bool(params_['forward_sim'])

    simulations_per_epoch = int(params_['phase1']['simulations_per_epoch'])

    backward_pass = True

    nb_epochs = params_['phase1']['ROLLOUT_epochs']
    ROLLOUT = params_['phase1']['ROLLOUT_begin']
    ROLLOUT_ADD = params_['phase1']['ROLLOUT_add']
    ROLLOUT_NUM = params_['phase1']['ROLLOUT_increases']

    CURRENT_ROLLOUT = ROLLOUT

    NOISE_INIT_SCALE_FACTOR = params_['NOISE_INIT_SCALE']
    DT = params_['DT']
    t1 = params_['t1']

    noise = jnp.sqrt(DT) * params_['training']['noise']

    print_every_batch = 10
    print_every = 1
    log_video_every = 1
    save_model_every = log_video_every

    print('***********************************************************************')
    print(f'Starting with phase 1 rollout...')
    print(f'Forward pass activated: {forward_pass}')
    print(
        f'START: {CURRENT_ROLLOUT}, NUMBER OF LOOPS: {ROLLOUT_NUM}, INCREASE ROLLOUT AFTER EVERY LOOP BY: {ROLLOUT_ADD}')
    print(f'NUMBER OF EPOCHS EACH ROLLOUT: {nb_epochs}')
    print(f'BATCHES WITHIN EACH EPOCH: {simulations_per_epoch}')
    print(f'BATCH SIZE: {params_["batch_size"]}')
    print('***********************************************************************')

    num_batches = min(len(generator), simulations_per_epoch)

    for rollout_n in range(ROLLOUT_NUM):

        print(f'Current rollout {CURRENT_ROLLOUT}')

        for epoch in range(nb_epochs):

            print(f'Epoch {epoch:3d}...')

            if params_['start_epoch'] > epoch + rollout_n * nb_epochs:
                continue

            for batch_n in range(min(len(generator), simulations_per_epoch)):

                data = generator.__getitem__(batch_n)

                obstacles = [item['obstacle_list'] for item in data]
                obstacles = batch_geometries_pre(obstacles)
                simulation_steps = data[0]['smoke'].shape[0]
                n_steps = data[0]['smoke'].shape[0] - CURRENT_ROLLOUT

                batch_loss = 0.0

                t_sim_train = jnp.array(0.0)
                forward_loss_avg = 0.0

                dummy = [0, 0, 0]  # dummy element

                if forward_pass:

                    # n_steps_random = [int(x) for x in list(jax.random.choice(rng, n_steps, shape=(params_['time_steps_per_batch'],)))]
                    # rng, _ = jax.random.split(rng)

                    for i in range(simulation_steps):  # tqdm(n_steps_random):

                        model_params_, opt_state, batch_loss, _, rng = train_inner_forward(i, CURRENT_ROLLOUT,
                                                                                           t_sim_train, data, obstacles,
                                                                                           grad_update_dict['forward'],
                                                                                           model_params,
                                                                                           opt_state, noise, rng, dummy)

                        if has_nan_weights(model_params_):
                            print(f'Epoch {epoch + epochs_phase_1:3d} !model has NAN.. resetting')
                            break
                        else:
                            model_params = model_params_

                        forward_loss_avg += batch_loss / CURRENT_ROLLOUT

                        t_sim_train += DT

                    if batch_n % print_every_batch == 0:
                        print(
                            f'Epoch {epoch:3d} Batch {batch_n:4d} Forward loss: {jnp.abs(batch_loss / CURRENT_ROLLOUT):.10f}')

                t_sim_train = jnp.array(t1)
                backward_loss_avg = 0.0

                if backward_pass:

                    # n_steps_random = [int(x) for x in list(jax.random.choice(rng, n_steps, shape=(params_['time_steps_per_batch'],)))]
                    # rng, _ = jax.random.split(rng)

                    for i in range(simulation_steps):  # tqdm(n_steps_random):

                        model_params_, opt_state, batch_loss, _, rng = train_inner_backward(i, CURRENT_ROLLOUT,
                                                                                            t_sim_train, data,
                                                                                            obstacles, grad_update_dict[
                                                                                                'backward'],
                                                                                            model_params,
                                                                                            opt_state, noise, rng,
                                                                                            dummy)

                        if has_nan_weights(model_params_):
                            print(f'Epoch {epoch + epochs_phase_1:3d} !model has NAN.. resetting')
                            break
                        else:
                            model_params = model_params_

                        backward_loss_avg += batch_loss / CURRENT_ROLLOUT

                        t_sim_train -= DT

                    if batch_n % print_every_batch == 0:
                        print(
                            f'Epoch {epoch:3d} Batch {batch_n:4d} Backward loss: {jnp.abs(batch_loss / CURRENT_ROLLOUT):.10f}')

            if epoch % 1 == 0:
                print(
                    f'Epoch {epoch:3d} Average forward loss: {jnp.abs(forward_loss_avg / (n_steps * num_batches)):.10f}')
                print(
                    f'Epoch {epoch:3d} Average backward loss: {jnp.abs(backward_loss_avg / (n_steps * num_batches)):.10f}')

            rng, _ = jax.random.split(rng)

            generator.on_epoch_end()

            if (epoch % print_every) == 0:

                log_dict = {}
                log_dict['epoch'] = epoch + rollout_n * nb_epochs
                log_dict['ROLLOUT'] = CURRENT_ROLLOUT
                log_dict['forward_loss'] = forward_loss_avg / (n_steps * num_batches)
                log_dict['backward_loss'] = backward_loss_avg / (n_steps * num_batches)
                log_dict['lr'] = params_['phase1']['learning_rate']

                if ((epoch + 1) % log_video_every) == 0 or epoch + 1 == nb_epochs:
                    log_dict = log_videos(log_dict, params_, model_params, inference_dict, generator)

                if (epoch % save_model_every) == 0:  # or epoch+1 == nb_epochs:

                    save_model_weights(model_params, f'weights/{wandb.run.id}_r{CURRENT_ROLLOUT:02d}_{epoch:04d}.p')

                wandb.log(log_dict)

        print(f'**********************************************************')
        print(f'Phase 1: Finished {nb_epochs} epochs in ROLLOUT {CURRENT_ROLLOUT}')
        print(f'**********************************************************')

        CURRENT_ROLLOUT += ROLLOUT_ADD

    print(f'**********************************************************')
    print(f'Phase 1: Finished all rollouts - Done')
    print(f'**********************************************************')

    return model_params, opt_state, rng


def train_phase_2(params_, generator, model_params, opt_state, grad_update_dict, inference_dict, rng):
    forward_pass = bool(params_['forward_sim'])
    backward_pass = True

    simulations_per_epoch = int(params_['phase2']['simulations_per_epoch'])

    # Calculate epochs in phase 1
    nb_epochs = params_['phase1']['ROLLOUT_epochs']
    ROLLOUT = params_['phase1']['ROLLOUT_begin']
    ROLLOUT_ADD = params_['phase1']['ROLLOUT_add']
    ROLLOUT_NUM = params_['phase1']['ROLLOUT_increases']

    epochs_phase_1 = ROLLOUT_NUM * nb_epochs

    ROLLOUT = params_['phase2']['ROLLOUT']
    num_epochs = params_['phase2']['epochs']

    NOISE_INIT_SCALE_FACTOR = params_['NOISE_INIT_SCALE']
    DT = params_['DT']
    t1 = params_['t1']

    noise = jnp.sqrt(DT) * params_['training']['noise']

    print_every_batch = 10
    print_every = 1
    log_video_every = 30
    save_model_every = log_video_every

    lr = params_['phase2']['learning_rate']
    opt_state.hyperparams['lr'] = lr

    gamma_steps = params_['phase2']['gamma_steps']
    gamma_factor = params_['phase2']['gamma_factor']

    full_simulation = params_['phase2']['full_simulation']

    print('***********************************************************************')
    print(f'Starting with phase 2 finetuning...')
    print(f'Forward pass activated: {forward_pass}')
    print(f'CUTTING GRADIENT AFTER {ROLLOUT} STEPS')
    print(f'FULL SIMULATION: {full_simulation}')
    print(f'NUMBER OF EPOCHS: {num_epochs}')
    print(f'BATCHES WITHIN EACH EPOCH: {simulations_per_epoch}')
    print(f'BATCH SIZE: {params_["batch_size"]}')
    print(f'INITIAL LEARNING RATE {lr}')
    print(f'DECREASING LEARNING RATE EVERY {gamma_steps} EPOCHS BY FACTOR {gamma_factor}')
    print('***********************************************************************')

    num_batches = min(len(generator), simulations_per_epoch)

    for epoch in range(params_['phase2']['epochs']):

        # Decrease learning rate
        if epoch + 1 % gamma_steps == 0:
            lr = lr * gamma_factor
            opt_state.hyperparams['lr'] = lr

            print(f'**********************************************************')
            print(f'Epoch {epoch:3d}: Decreasing learning rate from {lr / gamma_factor:.5f} to {lr:.5f}')
            print(f'**********************************************************')

        if params_['start_epoch'] > epoch + epochs_phase_1:
            continue

        for batch_n in range(num_batches):

            print(batch_n + 1, ' of ', num_batches)

            data = generator.__getitem__(batch_n)
            obstacles = [item['obstacle_list'] for item in data]
            obstacles = batch_geometries_pre(obstacles)
            num_steps = data[0]['smoke'].shape[0]

            # Alternate between starting at ROLLOUT / 2 or ROLLOUT
            if batch_n % 2 == 0:
                start_ = int(ROLLOUT / 2)
            else:
                start_ = ROLLOUT

            n_steps_list = [(0, start_)] + [(start_ + i * ROLLOUT, ROLLOUT) for i in
                                            range(int(np.ceil(num_steps / ROLLOUT - 0.5)))]

            num_segments = len(n_steps_list)

            batch_loss = 0.0

            t_sim_train = jnp.array(0.0)
            forward_loss_avg = 0.0
            count_stable_forward = 0

            if forward_pass:

                end_state = [0, 0, 0]  # dummy element
                useInit = False

                for segment in n_steps_list:

                    i, CURRENT_ROLLOUT = segment
                    model_params_, opt_state, batch_loss, end_state, rng = train_inner_forward(i, CURRENT_ROLLOUT,
                                                                                               t_sim_train, data,
                                                                                               obstacles,
                                                                                               grad_update_dict[
                                                                                                   'forward'],
                                                                                               model_params,
                                                                                               opt_state, noise, rng,
                                                                                               init=end_state)

                    if has_nan_weights(model_params_):
                        print(f'Epoch {epoch + epochs_phase_1:3d} !model has NAN.. resetting')
                        break
                    else:
                        model_params = model_params_

                    useInit = full_simulation

                    forward_loss_avg += batch_loss / CURRENT_ROLLOUT

                    t_sim_train += DT

                    # Do not continue rollout when states diverge
                    if max([jnp.max(s) for s in end_state]) > 10e5:
                        break
                    else:
                        count_stable_forward += 1

                if batch_n % print_every_batch == 0:
                    print(
                        f'Epoch {epoch + epochs_phase_1:3d} Batch {batch_n:4d} Forward loss: {jnp.abs(batch_loss / CURRENT_ROLLOUT):.10f}')

            t_sim_train = jnp.array(t1)
            backward_loss_avg = 0.0
            count_stable_backward = 0

            if backward_pass:

                end_state = [0, 0, 0]  # dummy element
                useInit = False

                for segment in n_steps_list:

                    i, CURRENT_ROLLOUT = segment
                    model_params_, opt_state, batch_loss, end_state, rng = train_inner_backward(i, CURRENT_ROLLOUT,
                                                                                                t_sim_train, data,
                                                                                                obstacles,
                                                                                                grad_update_dict[
                                                                                                    'backward'],
                                                                                                model_params, opt_state,
                                                                                                noise, rng,
                                                                                                init=end_state,
                                                                                                useInit=useInit)

                    if has_nan_weights(model_params_):
                        print(f'Epoch {epoch + epochs_phase_1:3d} !model has NAN.. resetting')
                        break
                    else:
                        model_params = model_params_

                    useInit = full_simulation

                    backward_loss_avg += batch_loss / CURRENT_ROLLOUT

                    t_sim_train -= DT

                    # Do not continue rollout when states diverge
                    if max([jnp.max(s) for s in end_state]) > 10e4:
                        continue
                    else:
                        count_stable_backward += 1

                if batch_n % print_every_batch == 0:
                    print(
                        f'Epoch {epoch + epochs_phase_1:3d} Batch {batch_n:4d} Backward loss: {jnp.abs(batch_loss / CURRENT_ROLLOUT):.10f}')

        if epoch % 1 == 0:
            print(
                f'Epoch {epoch + epochs_phase_1:3d} Average forward loss: {jnp.abs(forward_loss_avg / (num_batches * num_segments)):.10f}')
            print(
                f'Epoch {epoch + epochs_phase_1:3d} Average backward loss: {jnp.abs(backward_loss_avg / (num_batches * num_segments)):.10f}')

        rng, _ = jax.random.split(rng)

        generator.on_epoch_end()

        if (epoch % print_every) == 0:

            log_dict = {}
            log_dict['epoch'] = epoch + epochs_phase_1
            log_dict['ROLLOUT'] = CURRENT_ROLLOUT
            log_dict['forward_loss'] = forward_loss_avg / (num_batches * num_segments)
            log_dict['backward_loss'] = backward_loss_avg / (num_batches * num_segments)
            log_dict['lr'] = lr
            log_dict['percentage_stable_forward'] = count_stable_forward / (num_batches * num_segments)
            log_dict['percentage_stable_backward'] = count_stable_backward / (num_batches * num_segments)

            print(
                f'Epoch {epoch + epochs_phase_1:3d} Percentage stable forward: {count_stable_forward / (num_batches * num_segments):.3f}')
            print(
                f'Epoch {epoch + epochs_phase_1:3d} Percentage stable backward: {count_stable_backward / (num_batches * num_segments):.3f}')

            if ((epoch + 1) % log_video_every) == 0 or epoch + 1 == nb_epochs:
                log_dict = log_videos(log_dict, params_, model_params, inference_dict, generator)

            if (epoch % save_model_every) == 0 or epoch + 1 == nb_epochs:
                save_model_weights(model_params, f'weights/{wandb.run.id}_f_{epoch:04d}.p')

            wandb.log(log_dict)

    print(f'**********************************************************')
    print(f'Phase 2: Finished all epochs - Done')
    print(f'**********************************************************')

    return model_params, opt_state, rng


# @partial(jax.jit, static_argnames=['grad_update'])
def train_inner_forward(i, CURRENT_ROLLOUT, t_sim_train, data, obstacles, grad_update, model_params, opt_state, noise,
                        rng, init, useInit=0.0):
    target_smoke = [item['smoke'][i:i + CURRENT_ROLLOUT] for item in data]
    target_vel_x = [item['vel_x'][i:i + CURRENT_ROLLOUT] for item in data]
    target_vel_y = [item['vel_y'][i:i + CURRENT_ROLLOUT] for item in data]
    target_smoke = jnp.stack(target_smoke, axis=1)
    target_vel_x = jnp.stack(target_vel_x, axis=1)
    target_vel_y = jnp.stack(target_vel_y, axis=1)

    # target_smoke = target_smoke + noise * jr.normal(rng, shape=target_smoke.shape)
    # rng, _ = jax.random.split(rng)

    # target_vel_x = target_vel_x + noise * jr.normal(rng, shape=target_vel_x.shape)
    # rng, _ = jax.random.split(rng)

    # target_vel_y = target_vel_y + noise * jr.normal(rng, shape=target_vel_y.shape)
    # rng, _ = jax.random.split(rng)

    state_smoke = (1 - useInit) * target_smoke[0] + useInit * init[0]
    state_vel_x = (1 - useInit) * target_vel_x[0] + useInit * init[1]
    state_vel_y = (1 - useInit) * target_vel_y[0] + useInit * init[2]

    state_mask = [item['mask'][i] for item in data]
    state_mask = jnp.stack(state_mask, axis=0)

    state_smoke = state_smoke + noise * jr.normal(rng, shape=state_smoke.shape)
    rng, _ = jax.random.split(rng)

    state_vel_x = state_vel_x + noise * jr.normal(rng, shape=state_vel_x.shape)
    rng, _ = jax.random.split(rng)

    state_vel_y = state_vel_y + noise * jr.normal(rng, shape=state_vel_y.shape)
    rng, _ = jax.random.split(rng)

    # convert to float32
    state_smoke = jnp.array(state_smoke, dtype=jnp.float32)
    state_vel_x = jnp.array(state_vel_x, dtype=jnp.float32)
    state_vel_y = jnp.array(state_vel_y, dtype=jnp.float32)
    state_mask = jnp.array(state_mask, dtype=jnp.float32)

    target_smoke = jnp.array(target_smoke, dtype=jnp.float32)
    target_vel_x = jnp.array(target_vel_x, dtype=jnp.float32)
    target_vel_y = jnp.array(target_vel_y, dtype=jnp.float32)

    t_sim_train = jnp.array(t_sim_train, dtype=jnp.float32)

    state = [state_smoke, state_vel_x, state_vel_y, state_mask]
    target = [target_smoke, target_vel_x, target_vel_y]

    model_params, opt_state, batch_loss, end_state = grad_update(model_params, opt_state,
                                                                 [t_sim_train, state, target, obstacles, rng], rng=rng)

    return model_params, opt_state, batch_loss, end_state, rng


# @partial(jax.jit, static_argnames=['grad_update'])
def train_inner_backward(i, CURRENT_ROLLOUT, t_sim_train, data, obstacles, grad_update, model_params, opt_state, noise,
                         rng, init, useInit=0.0):
    target_smoke = [item['smoke'][::-1][i:i + CURRENT_ROLLOUT] for item in data]
    target_vel_x = [item['vel_x'][::-1][i:i + CURRENT_ROLLOUT] for item in data]
    target_vel_y = [item['vel_y'][::-1][i:i + CURRENT_ROLLOUT] for item in data]
    target_smoke = jnp.stack(target_smoke, axis=1)
    target_vel_x = jnp.stack(target_vel_x, axis=1)
    target_vel_y = jnp.stack(target_vel_y, axis=1)

    # target_smoke = target_smoke + noise * jr.normal(rng, shape=target_smoke.shape)
    # rng, _ = jax.random.split(rng)

    # target_vel_x = target_vel_x + noise * jr.normal(rng, shape=target_vel_x.shape)
    # rng, _ = jax.random.split(rng)

    # target_vel_y = target_vel_y + noise * jr.normal(rng, shape=target_vel_y.shape)
    # rng, _ = jax.random.split(rng)

    state_smoke = (1 - useInit) * target_smoke[0] + useInit * init[0]
    state_vel_x = (1 - useInit) * target_vel_x[0] + useInit * init[1]
    state_vel_y = (1 - useInit) * target_vel_y[0] + useInit * init[2]

    state_mask = [item['mask'][::-1][i] for item in data]
    state_mask = jnp.stack(state_mask, axis=0)

    state_smoke = state_smoke + noise * jr.normal(rng, shape=state_smoke.shape)
    rng, _ = jax.random.split(rng)

    state_vel_x = state_vel_x + noise * jr.normal(rng, shape=state_vel_x.shape)
    rng, _ = jax.random.split(rng)

    state_vel_y = state_vel_y + noise * jr.normal(rng, shape=state_vel_y.shape)
    rng, _ = jax.random.split(rng)

    # convert to float32
    state_smoke = jnp.array(state_smoke, dtype=jnp.float32)
    state_vel_x = jnp.array(state_vel_x, dtype=jnp.float32)
    state_vel_y = jnp.array(state_vel_y, dtype=jnp.float32)
    state_mask = jnp.array(state_mask, dtype=jnp.float32)

    target_smoke = jnp.array(target_smoke, dtype=jnp.float32)
    target_vel_x = jnp.array(target_vel_x, dtype=jnp.float32)
    target_vel_y = jnp.array(target_vel_y, dtype=jnp.float32)

    state = [state_smoke, state_vel_x, state_vel_y, state_mask]
    target = [target_smoke, target_vel_x, target_vel_y]

    t_sim_train = jnp.array(t_sim_train, dtype=jnp.float32)

    model_params, opt_state, batch_loss, end_state = grad_update(model_params, opt_state,
                                                                 [t_sim_train, state, target, obstacles, rng], rng=rng)

    return model_params, opt_state, batch_loss, end_state, rng


def eval_test(params, test_generator, model_weights, inference_dict, rng):
    NUM_ELEMS = 100
    keys = [key_[1] for key_ in test_generator.keys[:NUM_ELEMS]]

    data_init = test_generator.load((list(test_generator.h5files.keys())[0], keys[0]), transform=False)

    correction_probability_flow_fn_ = inference_dict['probability_flow']
    correction_reverse_sde_fn_ = inference_dict['reverse_sde']

    correction_probability_flow_fn = lambda state, t: correction_probability_flow_fn_(model_weights, state, t)
    correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(model_weights, state, t)

    simulation_metadata = {}

    inflow = data_init['INFLOW']

    center = math.tensor([(inflow['_center'][1], inflow['_center'][0])], batch('batch'), channel(vector='x,y'))

    simulation_metadata['INFLOW'] = Sphere(center=center, radius=inflow['_radius'])
    simulation_metadata['smoke_res'] = data_init['smoke_res']
    simulation_metadata['v_res'] = data_init['v_res']

    bounds = data_init['BOUNDS']

    simulation_metadata['BOUNDS'] = Box(x=(bounds['_lower'][0], bounds['_upper'][0]),
                                        y=(bounds['_lower'][1], bounds['_upper'][1]))

    noise = params['inference']['noise']

    update = params['update']
    type_ = params['physics_backward']

    data_dict = {}

    for DT in [0.01, 0.02, 0.005]:

        print(f'Evaluating at DT {DT}')

        data_dict[DT] = {}

        params['DT'] = DT
        simulation_metadata['DT'] = params['DT']
        simulation_metadata['NSTEPS'] = int(params['t1'] / params['DT'])

        physics_forward_fn = jax.jit(physics_forward(simulation_metadata))
        physics_backward_fn = jax.jit(physics_backwards(simulation_metadata))

        for key in keys:

            simulation_metadata['DT'] = params['DT']
            simulation_metadata['NSTEPS'] = int(params['t1'] / params['DT'])

            data_dict[DT][key] = {}
            data = test_generator.load((list(test_generator.h5files.keys())[0], key), transform=False)

            obstacles = [data['obstacle_list']]
            obstacles = batch_geometries_pre(obstacles)

            smoke_state = jnp.array(data['smoke'], dtype=jnp.float32)
            vel_x_state = jnp.array(data['vel_x'], dtype=jnp.float32)
            vel_y_state = jnp.array(data['vel_y'], dtype=jnp.float32)
            mask_state = jnp.array(data['mask'], dtype=jnp.float32)

            vmin_smoke = jnp.min(smoke_state)
            vmax_smoke = jnp.max(smoke_state)
            vmin_vel_x = jnp.min(vel_x_state)
            vmax_vel_x = jnp.max(vel_x_state)
            vmin_vel_y = jnp.min(vel_y_state)
            vmax_vel_y = jnp.max(vel_y_state)

            vmin_dict = {0: vmin_smoke, 1: np.minimum(vmin_vel_x, vmin_vel_y), 2: np.minimum(vmin_vel_x, vmin_vel_y)}
            vmax_dict = {0: vmax_smoke, 1: np.maximum(vmax_vel_x, vmax_vel_y), 2: np.maximum(vmax_vel_x, vmax_vel_y)}

            ground_truth = list(zip(jnp.expand_dims(smoke_state, axis=1), jnp.expand_dims(vel_x_state, axis=1),
                                    jnp.expand_dims(vel_y_state, axis=1), jnp.expand_dims(mask_state, axis=1)))

            for i in range(3):
                smoke_state_init = smoke_state[-1][None] + DT * noise * jr.normal(rng,
                                                                                  shape=smoke_state[-1][None].shape,
                                                                                  dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                vel_x_state_init = vel_x_state[-1][None] + DT * noise * jr.normal(rng,
                                                                                  shape=vel_x_state[-1][None].shape,
                                                                                  dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                vel_y_state_init = vel_y_state[-1][None] + DT * noise * jr.normal(rng,
                                                                                  shape=vel_y_state[-1][None].shape,
                                                                                  dtype=jnp.float32)
                rng, _ = jax.random.split(rng)
                mask_state_init = mask_state[-1][None]

                # states_backward_rand = eval_backward_score([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_reverse_sde_fn, rng, noise, type=type_, update=update, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)

                # rng, _ = jax.random.split(rng)

                # states_backward_rand_2 = eval_backward_score_decoupled([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, noise, type=type_, update=update, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)

                # rng, _ = jax.random.split(rng)

                states_backward_rand_2 = eval_backward_score_decoupled(
                    [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles,
                    simulation_metadata, correction_reverse_sde_fn, rng, noise, type=type_, update=update,
                    physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)

                rng, _ = jax.random.split(rng)

                # states_backward_rand_pc = eval_backward_score([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_reverse_sde_fn, rng, noise, type=type_, update=update, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn, corrector_steps=40)

                # rng, _ = jax.random.split(rng)

                # states_backward_rand_pc_2 = eval_backward_score_decoupled([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, noise, type=type_, update=update, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn, corrector_steps=40)

                # rng, _ = jax.random.split(rng)

                states_backward_rand_pc_2 = eval_backward_score_decoupled(
                    [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles,
                    simulation_metadata, correction_reverse_sde_fn, rng, noise, type=type_, update=update,
                    physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn, corrector_steps=40)

                rng, _ = jax.random.split(rng)

                states_backward_no_noise = eval_backward_score_decoupled(
                    [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles,
                    simulation_metadata, correction_reverse_sde_fn, rng, 0.0, type=type_, update=update,
                    physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)

                rng, _ = jax.random.split(rng)

                data_dict[DT][key][f'reverse_sde_{i}'] = states_backward_rand_2
                data_dict[DT][key][f'reverse_sde_{i}_pc'] = states_backward_rand_pc_2
                data_dict[DT][key][f'reverse_sde_no_noise'] = states_backward_no_noise

            data_dict[DT][key]['ground_truth'] = ground_truth[::-1]

            data_dict[DT][key]['no_correction'] = eval_backward_score_decoupled(
                [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata,
                correction_probability_flow_fn, rng, 0, correction_coefficient=0.0, type=type_, update=update,
                physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
            data_dict[DT][key]['probability_flow_noise'] = eval_backward_score_decoupled(
                [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata,
                correction_probability_flow_fn, rng, noise, correction_coefficient=1.0, type=type_, update=update,
                physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
            data_dict[DT][key]['probability_flow'] = eval_backward_score_decoupled(
                [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata,
                correction_probability_flow_fn, rng, 0, correction_coefficient=1.0, type=type_, update=update,
                physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)

            data_dict[DT][key]['forward'] = {}

            for t in [5, 10, 20, 30, 40]:

                t_adjusted = int(0.01 / DT)
                t_adjusted = t * t_adjusted

                print(f'Evaluating forward from timestep {t}')

                data_dict[DT][key]['forward'][t] = {}

                for k in data_dict[DT][key]:

                    if k == 'forward' or k == 'ground_truth':
                        continue

                    data_dict[DT][key]['forward'][t][k] = {}

                    simulation_metadata['DT'] = DT
                    simulation_metadata['NSTEPS'] = t_adjusted

                    init_state = data_dict[DT][key][k][t_adjusted]

                    data_dict[DT][key]['forward'][t][k] = eval_forward(init_state, obstacles, simulation_metadata,
                                                                       physics_forward_fn=physics_forward_fn,
                                                                       t0=params['t1'] - t_adjusted * DT)

    data_dict['params'] = params

    savename = f'test_saves/{params["name"]}.p'

    with open(savename, 'wb') as file:
        pickle.dump(data_dict, file)

    return data_dict


def train(
        forward_fn,
        model_params,
        generator,
        test_generator,
        params_,
        simulation_metadata,
        seed=56789,
):
    print('Noise for training set at ', params_['training']['noise'])

    print('**************************')

    print('Noise for inference set at ', params_['inference']['noise'])

    # SETUP OPTIMIZER

    opt_lr = lambda lr: optax.chain(
        optax.clip(10),
        optax.zero_nans(),
        optax.scale_by_adam(b1=0.9, b2=0.999),
        optax.scale(step_size=-lr))

    opt = optax.inject_hyperparams(opt_lr)(lr=params_['phase1']['learning_rate'])

    opt_state = opt.init(model_params)

    _global_step = 0
    current_step = _global_step
    DT = simulation_metadata['DT']

    # SETUP NETWORK FUNCTIONS

    training_correction_probability_flow_fn = get_correction_term_probability_flow_score(forward_fn,
                                                                                         params_['training']['noise'])
    training_correction_reverse_sde_fn = get_correction_term_reverse_sde_score(forward_fn, params_['training']['noise'])

    if bool(params_['forward_sim']):
        inference_correction_probability_flow_fn = jax.jit(
            get_correction_term_probability_flow_score(forward_fn, params_['inference']['noise']))
        inference_correction_reverse_sde_fn = jax.jit(
            get_correction_term_reverse_sde_score(forward_fn, params_['inference']['noise']))
    else:
        # if no forward sim, then network learns score instead of 1/2 score
        inference_correction_probability_flow_fn = jax.jit(
            get_correction_term_probability_flow_score(forward_fn, params_['inference']['noise'] / jnp.sqrt(2)))
        inference_correction_reverse_sde_fn = jax.jit(
            get_correction_term_reverse_sde_score(forward_fn, params_['inference']['noise'] / jnp.sqrt(2)))

    # physics_forward, physics_backwards
    simulation_metadata_physics = deepcopy(simulation_metadata)
    simulation_metadata_physics['INFLOW'] = simulation_metadata_physics['INFLOW_1b']
    physics_forward_fn = physics_forward(simulation_metadata_physics)
    physics_backward_fn = physics_backwards(simulation_metadata_physics)

    model_loss_forward_fn = jax.jit(
        gradient_forward_fn_manual(training_correction_probability_flow_fn, simulation_metadata,
                                   update=params_['update'], rollout_noise=params_['rollout_noise']))

    grad_update_forward = jax.jit(create_default_update_fn(opt, model_loss_forward_fn))

    model_loss_backward_fn = jax.jit(
        gradient_backward_fn_manual(training_correction_probability_flow_fn, simulation_metadata,
                                    type=params_['physics_backward'], update=params_['update'],
                                    rollout_noise=params_['rollout_noise']))

    grad_update_backward = jax.jit(create_default_update_fn(opt, model_loss_backward_fn))

    # SETUP random KEY

    rng = jax.random.PRNGKey(seed)
    test_rng = jax.random.PRNGKey(2022)

    # Phase 1

    grad_update_dict = {'forward': grad_update_forward, 'backward': grad_update_backward}
    inference_dict = {'probability_flow': inference_correction_probability_flow_fn,
                      'reverse_sde': inference_correction_reverse_sde_fn,
                      'physics_forward': jax.jit(physics_forward_fn), 'physics_backward': jax.jit(physics_backward_fn),
                      'physics_backward_type': params_['physics_backward'], 'update': params_['update']}

    if not params_['test_only']:
        model_params, opt_state, rng = train_phase_1(params_, generator, model_params, opt_state, grad_update_dict,
                                                     inference_dict, rng)

        # Phase 2

        model_params, opt_state, rng = train_phase_2(params_, generator, model_params, opt_state, grad_update_dict,
                                                     inference_dict, rng)

    # Running tests

    if not test_generator is None:
        eval_test(params_, test_generator, model_params, inference_dict, test_rng)

    return model_params


def main(params):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".7"

    # params['api_key'] = '10f2ddff5bfdb03ddfa296889625f5d943443af1'

    if (not 'api_key' in params) or (params['api_key'] is None):
        os.environ["WANDB_MODE"] = 'dryrun'
    else:
        os.environ["WANDB_API_KEY"] = params['api_key']

    # DRYRUN
    os.environ["WANDB_MODE"] = 'dryrun'

    os.environ["WANDB_MODE"] = 'online'

    if (not 'name' in params) or (params['name'] is None):
        name = 'Buoyancy-driven flow'
    else:
        name = params['name']

    if 'gpu' in params and not (params['gpu'] is None):
        os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']

    if params['continue_id']:
        wandb.init(id=params['continue_id'], resume='allow', config=params, project='bouyancy-flow', name=name)
    else:
        wandb.init(config=params, project='buoyancy-flow', name=name)

    key = jr.PRNGKey(params['seed'])

    with h5py.File(params['file'], 'r') as f:
        train_keys = list(f.keys())
        train_keys = list(zip([params['file']] * len(train_keys), train_keys))

    params_ = {}

    params_['name'] = params['name']

    params_['t1'] = float(params['t1'])
    params_['t0'] = 0.0
    params_['DT'] = params['DT']
    params_['maxTime'] = params['maxTime']

    params_['physics_backward'] = params['physics_backward']
    params_['update'] = params['update']

    params_['forward_sim'] = bool(params['forward_sim'])

    params_['batch_size'] = int(params['batch_size'])
    params_['rollout_noise'] = params['rollout_noise']

    params_['test_only'] = bool(params['test_only'])

    generator = DataLoader([params['file']], train_keys, name='train', batchSize=params_['batch_size'],
                           maxTime=params_['maxTime'])

    if not params['test_file'] is None:

        with h5py.File(params['test_file'], 'r') as f:
            test_keys = list(f.keys())
            test_keys = list(zip([params['test_file']] * len(test_keys), test_keys))

        test_generator = DataLoader([params['test_file']], test_keys, name='test', batchSize=params_['batch_size'],
                                    maxTime=params_['maxTime'], shuffle=False)

    else:

        test_generator = None

    item_init = generator.__getitem__(0)

    simulation_metadata = {}

    simulation_metadata['INFLOW'] = batch_inflow(item_init[0]['INFLOW'], batchSize=params_['batch_size'])
    simulation_metadata['INFLOW_1b'] = batch_inflow(item_init[0]['INFLOW'], batchSize=1)
    bounds = item_init[0]['BOUNDS']

    simulation_metadata['BOUNDS'] = Box(x=(bounds['_lower'][0], bounds['_upper'][0]),
                                        y=(bounds['_lower'][1], bounds['_upper'][1]))

    simulation_metadata['smoke_res'] = item_init[0]['smoke_res']
    simulation_metadata['v_res'] = item_init[0]['v_res']
    simulation_metadata['DT'] = params['DT']

    resolution = item_init[0]['smoke'].shape[1]

    data_shape = (resolution, resolution)

    params_['resolution'] = resolution
    params_['data_shape'] = data_shape

    params_['start_epoch'] = int(params['start_epoch'])

    params_['training'] = {}
    params_['training']['noise'] = float(params['training_noise'])

    if bool(params['inference_is_training']):

        print('inference_is_training is true')

        params_['inference'] = params_['training']

    else:

        print('inference_is_training is false')

        params_['inference'] = {}
        params_['inference']['noise'] = float(params['inference_noise'])

    params_['phase1'] = params['phase1']
    params_['phase2'] = params['phase2']

    params_['time_steps_per_batch'] = params['time_steps_per_batch']
    params_['NOISE_INIT_SCALE'] = params['NOISE_INIT_SCALE']
    model_key, train_key, loader_key, sample_key = jr.split(key, 4)

    params_['model_key'] = model_key
    params_['train_key'] = model_key
    params_['loader_key'] = model_key
    params_['sample_key'] = model_key

    params_['forward_sim'] = params['forward_sim']
    params_['batch_size'] = params['batch_size']

    if params['architecture'] == 'dilated':

        from models.DilatedConv import get_model
        forward_fn, init_params = get_model(params_['data_shape'], nBlocks=params['architecture_params']['nBlocks'],
                                            nFeatures=params['architecture_params']['nFeatures'],
                                            update_type=params['update'])

    elif params['architecture'] == 'encoder_decoder':

        from models.EncoderDecoder import get_model
        forward_fn, init_params = get_model(params_['data_shape'], update_type=params['update'])

    elif params['architecture'] == 'unet':

        from models.UNet import get_model
        forward_fn, init_params = get_model(params_['data_shape'], use_grid=params['architecture_params']['grid'],
                                            update_type=params['update'])

    if params['network_weights']:
        init_params = load_model_weights(params['network_weights'])
        print('Loading parameters from ', params['network_weights'])
    else:
        print('Starting from random initialization')

    print(f'model weights: {count_weights(init_params)}')

    model_params = train(forward_fn, init_params, generator, test_generator, params_, simulation_metadata)

    save_model_weights(model_params, f'weights/{wandb.run.id}.p')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Parser',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', default=None, help='Name of experiment')
    parser.add_argument('--file', required=True, help='Training data')
    parser.add_argument('--continue-id', default=None, help='ID of run to continue')
    parser.add_argument('--start-epoch', default=hyperparameter_defaults['start_epoch'], type=int,
                        help='Epoch to begin training')
    parser.add_argument('--gpu', default=None, help='Visible GPUs')
    parser.add_argument('--batch-size', default=hyperparameter_defaults['batch_size'], type=int, help='Batch size')
    parser.add_argument('--network-weights', default=hyperparameter_defaults['network_weights'],
                        help='File with weights used for initialization')
    parser.add_argument('--architecture', default=hyperparameter_defaults['architecture'], help='Network architecture')
    parser.add_argument('--training-noise', default=hyperparameter_defaults['training']['noise'], type=float,
                        help='Coefficient for noise during training')
    parser.add_argument('--inference-noise', default=hyperparameter_defaults['inference']['noise'], type=float,
                        help='Coefficient for noise during inference')
    parser.add_argument('--inference-is-training',
                        help='Flag if noise scales are the same for inference as for training', action='store_true')
    parser.add_argument('--forward-sim', default=hyperparameter_defaults['forward_sim'],
                        help='Flag whether to learn probability flow ODE during forward simulation',
                        action='store_true')
    parser.add_argument('--rollout-noise', default=hyperparameter_defaults['rollout_noise'],
                        help='Flag whether to include noise in rollouts', type=float)
    parser.add_argument('--test-only', default=hyperparameter_defaults['test_only'],
                        help='Flag whether to only run test', action='store_true')
    parser.add_argument('--test-file', default=hyperparameter_defaults['test_file'], help='Test file')
    parser.add_argument('--t1', default=hyperparameter_defaults['t1'], type=float, help='End time of simulation')
    parser.add_argument('--api-key', default=hyperparameter_defaults['api_key'], help='Wanbb API key')
    parser.add_argument('--update', default=hyperparameter_defaults['update'], type=int,
                        help='1 for only states as network inputs (default); 2 for including physics in network inputs')
    parser.add_argument('--physics-backward', default=hyperparameter_defaults['physics_backward'], type=int,
                        help='What backward physics to use; either 1 for reusing negative forward physics or 2 for time integration')

    args, unknown = parser.parse_known_args()

    hyperparameter_defaults.update(vars(args))

    main(hyperparameter_defaults)
