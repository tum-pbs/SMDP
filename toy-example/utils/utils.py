from typing import Callable, Tuple

import jax
import optax
import haiku as hk
import jax.numpy as jnp

import pickle
import os


def get_batches(l, batch_size):
    """
    Get batches of list
    Parameters
    ----------
    l: list
    batch_size: batch size

    Returns
    -------
    batches: list of batches
    """

    batches = []

    for i in range(0, len(l), batch_size):
        batches.append(l[i:i + batch_size])

    return batches


def save_network(model_params, logs, directory, opt_state=None):
    """
    Save model parameters and logs to disk
    Parameters
    ----------
    model_params: model parameters
    logs: logs
    directory: directory to save to
    :param opt_state:
    Returns
    -------


    """

    # create directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, f'model_params_{logs["grad_updates"]}.pkl'), 'wb') as f:
        pickle.dump(model_params, f)

    with open(os.path.join(directory, 'logs.pkl'), 'wb') as f:
        pickle.dump(logs, f)

    if opt_state is not None:
        with open(os.path.join(directory, f'opt_state_{logs["grad_updates"]}.pkl'), 'wb') as f:
            pickle.dump(opt_state, f)


def load_network(directory):
    """
    Load model parameters and logs from disk
    Parameters
    ----------
    directory: directory to load from

    Returns
    -------

    """

    with open(os.path.join(directory, 'logs.pkl'), 'rb') as f:
        logs = pickle.load(f)

    with open(os.path.join(directory, f'model_params_{logs["grad_updates"]}.pkl'), 'rb') as f:
        model_params = pickle.load(f)

    if os.path.exists(os.path.join(directory, f'opt_state_{logs["grad_updates"]}.pkl')):
        with open(os.path.join(directory, f'opt_state_{logs["grad_updates"]}.pkl'), 'rb') as f:
            opt_state = pickle.load(f)
    else:
        opt_state = None

    return model_params, opt_state, logs


def create_eval_fn(forward_fn, params):
    """
    Create evaluation function for model
    Parameters
    ----------
    forward_fn: model forward function
    params: model parameters

    Returns
    -------

    """

    @jax.jit
    def eval_model(x, t, key=None):
        return forward_fn(params, key, x, t)

    return eval_model


def create_default_update_fn(optimizer: optax.GradientTransformation,
                             model_loss: Callable):
    """
    Create default update function for model
    Parameters
    ----------
    optimizer: optax optimizer
    model_loss: model loss function

    Returns
    -------

    """

    @jax.jit
    def update(params, opt_state, batch, rng) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        batch_loss, grads = jax.value_and_grad(model_loss)(params, rng, *batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, batch_loss

    return update


def get_random_id():
    """
    Get random id based on timestamp
    Returns
    -------

    """
    import random
    import string

    from datetime import datetime

    random.seed(datetime.now().timestamp())
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
