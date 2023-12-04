from copy import deepcopy

import jax.random as jr

import wandb
from tqdm import tqdm

from dataset import get_dataset, iterate_batches


import jax
import functools
import jax.numpy as jnp

from jax import jacfwd

def gradient_fn(forward_fn, *args):
    """
    Create gradient function for sliced score matching
    Parameters
    ----------
    forward_fn: model evaluation function
    args: additional arguments, not used right now

    Returns
    -------

    """

    @jax.jit
    def model_loss(params, rng, x_train, y_train):
        def mul_v(params_, rng_, v_, x_train_, y_train_):

            n = forward_fn(params_, rng_, x_train_, y_train_)

            s = jnp.einsum('b, b ->', n, v_)

            return s

        @functools.partial(jax.vmap, in_axes=(0, 0))
        def score_loss(x, y):

            v = jr.rademacher(rng, (1,))

            grad_x = jacfwd(mul_v, argnums=(3,))(params, rng, v, x, y)

            reduced_x = grad_x[0] * v

            u = forward_fn(params, rng, x, y)

            return reduced_x + 0.5 * jnp.sum(jnp.square(u))

        loss_score = jnp.mean(score_loss(x_train, y_train))

        return loss_score

    return model_loss


def update_network_epoch(params, grad_update, opt_state, iterator, key, stopping_criterion, logs=None, log_wandb=False):
    """
    Update network for one epoch (one pass through the data set)
    Parameters
    ----------
    params : network parameters
    grad_update : gradient update function
    opt_state : optimizer state
    iterator: iterator function for dataset
    key : jr.PRNGKey
    logs : dict of logs (from previous epochs) to continue logging
    log_wandb: whether to log to wandb
    stopping_criterion: stopping criterion for training

    Returns
    -------


    """

    generator = iterator(key)
    key = jr.split(key)[0]

    if logs is None:
        logs = {}
    logs_epoch = deepcopy(logs)
    current_step = logs_epoch["step"]
    grad_updates = logs_epoch["grad_updates"]

    loss = 0.0
    i = 0

    for batch, _ in generator:

        x_train_sub = batch[:, 1]
        t_train_sub = batch[:, 0]

        # Compute loss for current part of trajectory
        params, opt_state, batch_loss = grad_update(params, opt_state,
                                                    [x_train_sub, t_train_sub], rng=key)
        key = jr.split(key)[0]
        grad_updates += 1

        current_step += 1

        logs_epoch["step"] = current_step
        logs_epoch["grad_updates"] = grad_updates
        logs_epoch["batch_loss"] = batch_loss

        loss += batch_loss
        i += 1

        if log_wandb:
            wandb.log(logs_epoch)

        if stopping_criterion(logs['epoch_acc'], logs['grad_updates']):
            print('Stopping criterion reached. Stopping training.')
            return params, opt_state, logs_epoch, key

    logs_epoch["loss"] = loss / i

    return params, opt_state, logs_epoch, key


def update_network(model_dict, dataset, opt_state, logs, training_config,
                   training_callback, key, stopping_criterion, log_wandb=False):
    """
    Setup training loop and update network weights for training_config["epochs"] epochs
    Parameters
    ----------
    model_dict : dict containing forward_fn, model_params, grad_update
    dataset : dataset
    opt_state : optimizer state
    logs : dict of logs (from previous epochs) to continue logging
    training_config : dict containing training parameters
    training_callback : callback function for training
    key : jr.PRNGKey
    stopping_criterion : stopping criterion for training
    log_wandb : bool whether to log to wandb

    Returns
    -------

    """

    if logs is None:
        logs = {"step": 0, "grad_updates": 0, "epoch": 0, "epoch_acc": 0}
    logs = deepcopy(logs)

    dataset = jnp.transpose(dataset, (0, 2, 1))
    dataset = dataset.reshape(-1, 2)

    iterator = lambda key_: iterate_batches(dataset, training_config['batch_size'],
                                            shuffle=training_config['shuffle'], key=key_)

    model_params = model_dict['model_params']

    pbar = tqdm(range(training_config['epochs']))

    logs["epoch"] = 0

    # Iterate through all epochs
    for n in pbar:

        if logs["epoch"] > n:
            continue

        logs["epoch"] = n

        # Update network for one epoch
        model_params, opt_state, logs_epoch, key = update_network_epoch(
            model_params, model_dict['grad_update'], opt_state, iterator, key, stopping_criterion,
            log_wandb=log_wandb, logs=logs)
        key = jr.split(key)[0]
        logs["step"] = logs_epoch["step"]
        logs["grad_updates"] = logs_epoch["grad_updates"]

        pbar.set_description(f'loss: {logs_epoch["loss"]:.5f} grad updates: {logs_epoch["grad_updates"]}')

        # Call callback function
        if training_callback is not None:
            logs = training_callback(model_params, logs)

        logs["epoch_acc"] += 1

        if stopping_criterion(logs['epoch_acc'], logs['grad_updates']):
            print('Stopping criterion reached. Stopping training.')
            return model_params, logs

    return model_params, logs
