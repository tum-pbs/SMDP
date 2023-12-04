
from copy import deepcopy

import jax
import jax.random as jr
import jax.numpy as jnp

from tqdm import tqdm

from dataset import iterate_batches, prepare_batch

import wandb


def gradient_fn(forward_fn, physics_operator):
    """
    Returns jitted model loss. In contrast to differentiable_physics.py this version only supports single steps
    for the trajectory, but it includes a regularization term. Adding his regularization term ensures that the network
    learns the score exactly and not only something that is proportional to the score (basically the regularization
    term represents the forward direction in learning the probability flow ODE).
    The regularization is equivalent to the expected forward direction loss.
    Parameters
    ----------
    forward_fn : model evaluation function
    physics_operator : physics operator

    Returns
    -------

    """
    @jax.jit
    def model_loss(model_weights, rng, x_train, t_train):
        x = x_train[:, 0]
        i = 1

        loss = 0.0

        t_train = jnp.transpose(t_train)
        for t1, t0 in zip(t_train, t_train[1:]):
            delta_t = t1 - t0

            physics_update = physics_operator(x)

            nn_output = forward_fn(model_weights, rng, jnp.expand_dims(x, axis=1),
                                              jnp.expand_dims(t1, axis=1))[:, 0]

            # note that we absorb g**2 (constant) in the definition of forward_fn here
            score_update = - 0.5 * nn_output

            regularization_term = 0.25 * (delta_t ** 2) * jnp.square(nn_output)

            x = x - delta_t * (physics_update + score_update)

            x_true = x_train[:, i]

            loss += jnp.mean(jnp.square(x - x_true)) + jnp.mean(regularization_term)
            i += 1

        return loss

    return model_loss


def update_network_epoch(params, grad_update, opt_state, iterator, key, subsample,
                         stopping_criterion, logs=None, log_wandb=False):
    """
    Update network for one epoch (one pass through the data set)
    Parameters
    ----------
    params : network parameters
    grad_update : gradient update function
    opt_state : optimizer state
    iterator : iterator function for dataset
    key : jr.PRNGKey
    subsample : subsampling rate (sample points on trajectory)
    logs : dict of logs (from previous epochs) to continue logging
    log_wandb: whether to log to wandb
    stopping_criterion: stopping criterion for training

    Returns
    -------
    """

    rollout = 2
    # Iterate through data set
    generator = iterator(key)
    key = jr.split(key)[0]

    if logs is None:
        logs = {}
    logs_epoch = deepcopy(logs)
    current_step = logs_epoch["step"]
    grad_updates = logs_epoch["grad_updates"]

    loss = 0

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
            params, opt_state, batch_loss = grad_update(params, opt_state,
                                                                 [x_train_sub, t_train_sub], rng=key)
            key = jr.split(key)[0]
            grad_updates += 1
            loss += jax.device_get(batch_loss)

        current_step += 1

        logs_epoch["step"] = current_step
        logs_epoch["grad_updates"] = grad_updates
        logs_epoch["loss"] = loss

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
    model_dict : dictionary containing forward_fn, model_params and grad_update
    dataset : dataset
    opt_state : optimizer state
    logs : dict of logs (from previous epochs) to continue logging
    training_config : training configuration
    training_callback : callback function for training
    key : jr.PRNGKey
    stopping_criterion : stopping criterion for training
    log_wandb : whether to log to wandb

    Returns
    -------

    """

    iterator = lambda key_: iterate_batches(dataset, training_config['batch_size'],
                                                shuffle=training_config['shuffle'], key=key_)

    if logs is None:
        logs = {"step": 0, "grad_updates": 0, "epoch": 0, "epoch_acc": 0}
    logs = deepcopy(logs)

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
            model_params, model_dict['grad_update'], opt_state, iterator, key, training_config['subsample'],
            stopping_criterion, log_wandb=log_wandb, logs=logs)
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


