from enum import Enum
from typing import List

import jax.numpy as jnp
import jax.random as jr
import jax.nn
import haiku as hk

EPSILON = 1e-5

modelType = Enum('modelType', 'mlp grid')


def mlp_model(output_sizes: List[int], activation, log_t=True, key=None):
    """
    MLP model for learning the score function
    Parameters
    ----------
    output_sizes : List[int] list of hidden layer sizes
    activation : activation function
    log_t : bool, if True, log transform time
    key : jr.PRNGKey, random key for initialization
    -------

    """

    def f(x, t):
        if log_t:
            t = jnp.log(t + EPSILON)
        x = jnp.hstack([x, t])
        net = hk.nets.MLP(output_sizes=output_sizes,
                          activation=activation)
        return net(x)

    init_params, forward_fn = hk.transform(f)

    if key is None:
        key = jr.PRNGKey(0)

    x_init = jnp.ones((10, 1))
    t_init = x_init
    params = init_params(key, x_init, t_init)

    return forward_fn, params


class GridModel(hk.Module):
    def __init__(self, x_size, y_size, order):
        super(GridModel, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.order = order

        self.weight = hk.get_parameter("weight", shape=(y_size, x_size), init=jnp.zeros)

    def __call__(self, x, t):
        interp = jax.scipy.ndimage.map_coordinates(self.weight, [x, t], order=self.order)

        return interp


def grid_model(order: int, t0=0.0, t1=10.0, y0=-1.25, y1=1.25, dt=0.02, dy=0.01, key=None):
    y_size = int((y1 - y0) / dy)
    x_size = int((t1 - t0) / dt)

    def f(x, t):
        # transform to grid coordinates
        t = (t - t0) / dt
        x = (x - y0) / dy

        net = GridModel(x_size, y_size, order)

        return net(x, t)

    init_params, forward_fn = hk.transform(f)

    if key is None:
        key = jr.PRNGKey(0)

    x_init = jnp.ones((10, 1))
    t_init = x_init
    params = init_params(key, x_init, t_init)

    return forward_fn, params


def get_paper_grid_model(key=None):
    """
    Get grid model from paper
    Parameters
    ----------
    key : jr.PRNGKey, random key for initialization
    -------

    """
    return grid_model(1, t0=0.0, t1=10.0, key=key)


def get_paper_mlp_model(key=None):
    """
    Get MLP model from paper
    Parameters
    ----------
    key : jr.PRNGKey, random key for initialization
    -------

    """
    activation = jax.nn.elu
    output_sizes = [30, 30, 25, 20, 10, 1]
    return mlp_model(output_sizes, activation, log_t=True, key=key)


def get_model(model_type: modelType, key, **kwargs):
    if model_type == modelType.mlp:
        return mlp_model(**kwargs, key=key)
    elif model_type == modelType.grid:
        return grid_model(**kwargs, key=key)
    else:
        raise ValueError("Unknown model type")
