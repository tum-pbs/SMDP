
from copy import deepcopy

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm

from dataset import iterate_batches, prepare_batch, get_dataset

import wandb


def gradient_fn(forward_fn, physics_operator):
    """
    Returns jitted model loss.
    Parameters
    ----------
    forward_fn : model evaluation function
    physics_operator : physics operator

    Returns
    -------

    """

    @jax.jit
    def model_loss(model_weights, rng, x_train, t_train):
        """
        Computes model loss based on l2 distance for one trajectory.
        Parameters
        ----------
        model_weights : network parameters
        rng : random number generator
        x_train : trajectory positions
        t_train : trajectory times

        Returns
        -------

        """
        x = x_train[:, 0]
        i = 1

        loss = 0.0

        t_train = jnp.transpose(t_train)
        for t1, t0 in zip(t_train, t_train[1:]):
            delta_t = t1 - t0

            physics_update = physics_operator(x)

            # note that we absorb g**2 (constant) in the definition of forward_fn here
            score_update = - 0.5 * forward_fn(model_weights, rng, jnp.expand_dims(x, axis=1),
                                              jnp.expand_dims(t1, axis=1))[:, 0]

            x = x - delta_t * (physics_update + score_update)

            x_true = x_train[:, i]

            loss += jnp.mean(jnp.square(x - x_true))
            i += 1

        return loss

    return model_loss


def update_network_epoch(params, grad_update, opt_state, iterator, key, rollout,
                         subsample, bidirectional, stopping_criterion, logs=None, log_wandb=False):
    """
    Update network for one epoch (one pass through the data set)
    Parameters
    ----------
    params : network parameters
    grad_update : gradient update function
    opt_state : optimizer state
    iterator : iterator function for dataset
    key : jr.PRNGKey
    rollout : rollout length (sliding window size)
    subsample : subsampling rate (sample points on trajectory)
    bidirectional : bool use forward and backward time direction
    logs : dict of logs (from previous epochs) to continue logging
    log_wandb: whether to log to wandb
    stopping_criterion: stopping criterion for training

    Returns
    -------


    """
    # Iterate through data set
    generator = iterator(key)
    key = jr.split(key)[0]

    loss_forward = 0
    loss_backward = 0

    if logs is None:
        logs = {}
    logs_epoch = deepcopy(logs)
    current_step = logs_epoch["step"]
    grad_updates = logs_epoch["grad_updates"]

    for batch in generator:

        batch = prepare_batch(batch[0])
        x_train = batch[:, 1]
        t_train = batch[:, 0]

        # Use subsampling to reduce number of points on trajectory
        x_train = x_train[:, ::subsample]
        t_train = t_train[:, ::subsample]

        # Iterate through trajectory

        for t in range(x_train.shape[1]):

            # Select values based on position and window size
            x_train_sub = x_train[:, t:t + rollout]
            t_train_sub = t_train[:, t:t + rollout]

            # Compute loss for current part of trajectory
            params, opt_state, batch_loss_backward = grad_update(params, opt_state,
                                                                 [x_train_sub, t_train_sub], rng=key)
            key = jr.split(key)[0]
            grad_updates += 1

            # Reverse values and time discretization for the forward time direction
            batch_loss_forward = 0
            if bidirectional:
                params, opt_state, batch_loss_forward = grad_update(params, opt_state,
                                                                    [x_train_sub[:, ::-1], t_train_sub[::-1]],
                                                                    rng=key)
                key = jr.split(key)[0]
                grad_updates += 1

            loss_backward += jax.device_get(batch_loss_backward)

            if bidirectional:
                loss_forward += jax.device_get(batch_loss_forward)

        current_step += 1

        logs_epoch["step"] = current_step
        logs_epoch["grad_updates"] = grad_updates
        logs_epoch["loss_backward"] = loss_backward
        logs_epoch["loss_forward"] = loss_forward

        if log_wandb:
            wandb.log(logs_epoch)

        if stopping_criterion(logs['epoch_acc'], logs['grad_updates']):
            print('Stopping criterion reached. Stopping training.')
            return params, opt_state, logs_epoch, key

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

    rollout = training_config['sw_start']

    iterator = lambda key_: iterate_batches(dataset, training_config['batch_size'],
                                                shuffle=training_config['shuffle'], key=key_)

    if logs is None:
        logs = {"step": 0, "grad_updates": 0, "epoch": 0, "epoch_acc": 0}
    logs = deepcopy(logs)

    model_params = model_dict['model_params']

    # Iterate through all sliding window sizes
    for _ in range(training_config['sw_steps'] + 1):

        if "rollout" in logs and logs["rollout"] > rollout:
            continue

        logs["rollout"] = rollout
        print('window size: ', rollout)

        pbar = tqdm(range(training_config['epochs']))

        logs["epoch"] = 0

        # Iterate through all epochs
        for n in pbar:

            if logs["epoch"] > n:
                continue

            logs["epoch"] = n

            # Update network for one epoch
            model_params, opt_state, logs_epoch, key = update_network_epoch(
                model_params, model_dict['grad_update'], opt_state, iterator, key, rollout,
                training_config['subsample'], training_config['bidirectional'], stopping_criterion,
                log_wandb=log_wandb, logs=logs)
            key = jr.split(key)[0]
            logs["step"] = logs_epoch["step"]
            logs["grad_updates"] = logs_epoch["grad_updates"]

            pbar.set_description(f'loss: {logs_epoch["loss_forward"]:.5f}; '
                                 f'{logs_epoch["loss_backward"]:.5f} grad updates: {logs_epoch["grad_updates"]}')

            # Call callback function
            if training_callback is not None:
                logs = training_callback(model_params, logs)

            logs["epoch_acc"] += 1

            if stopping_criterion(logs['epoch_acc'], logs['grad_updates']):
                print('Stopping criterion reached. Stopping training.')
                return model_params, logs

        rollout += training_config['sw_increase']

    return model_params, logs


