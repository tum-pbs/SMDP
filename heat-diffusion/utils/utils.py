import copy
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk
from typing import Callable, Tuple
import pickle
import numpy as np


def loguniform(rng, low=0.01, high=0.2, size=None):
    a = jnp.log(low)
    b = jnp.log(high + low)
    return jnp.exp(jr.uniform(rng, size) * (b - a) + a) - low


def sim_time_to_network_log(low, high, t):
    a = jnp.log(low)
    b = jnp.log(high + low)
    return (jnp.log(low + t) - a) / (b - a)


def log_time_to_network(t, t0, t1, offset=0.1):
    t_ = t * (t0 - jnp.log(t0 + offset) + jnp.log(t1 + offset)) / t1
    t_ = t_ + jnp.log(offset + t0) - t0
    t_ = jnp.exp(t_) - offset
    return t_


def get_times_linear(t0, t1, num_steps):
    return jnp.linspace(t0, t1, num_steps + 1)


def get_times_log(t0, t1, num_steps, offset=0.1):
    samples_linear = get_times_linear(t0, t1, num_steps)
    samples_log = jnp.log(samples_linear + offset)
    samples_log = samples_log - jnp.log(offset + t0) + t0
    samples_log = samples_log * (t1 / (t0 - jnp.log(t0 + offset) + jnp.log(t1 + offset)))
    return samples_log


def get_rand_times_linear(rng, t0, t1, num_steps):
    grid = jnp.linspace(t0, t1, num_steps + 1)
    width = (t1 - t0) / num_steps
    grid_start = grid[1:-1] - 0.5 * width
    rand = jr.uniform(rng, (num_steps - 1,)) * width
    positions = jnp.concatenate([jnp.array((t0,)), grid_start + rand, jnp.array((t1,))])
    return positions


def get_rand_times_log(rng, t0, t1, num_steps, offset=0.1):
    samples_linear = get_rand_times_linear(rng, t0, t1, num_steps)
    samples_log = jnp.log(samples_linear + offset)
    samples_log = samples_log - jnp.log(offset + t0) + t0
    samples_log = samples_log * (t1 / (t0 - jnp.log(t0 + offset) + jnp.log(t1 + offset)))
    # samples_log = jnp.flip(jnp.abs(samples_log-t1), axis=0)
    return samples_log


# https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def parse_arguments(args, default):
    default = copy.deepcopy(default)

    def subparse(dict_1, dict_2, prefix=''):

        m = {}

        for key in dict_1:

            key_u = prefix + key

            if type(dict_1[key]) is dict:

                if key_u in dict_2:

                    m[key] = subparse(dict_1[key], dict_2[key_u])

                else:

                    if key in ['embedded_physics', 'autoregressive', 'bayesian']:

                        m[key] = subparse(dict_1[key], dict_2)

                    else:

                        m[key] = subparse(dict_1[key], dict_2, prefix=f'{key_u}_')

            else:

                if key_u in dict_2:

                    if dict_2[key_u] is None:

                        m[key] = dict_1[key]

                    else:

                        if dict_1[key] is None:

                            m[key] = str(dict_2[key_u])

                        else:

                            m[key] = type(dict_1[key])(dict_2[key_u])

                else:

                    m[key] = dict_1[key]

        return m

    params = subparse(default, args)

    return params


def load_model_weights(savename):
    with open(savename, 'rb') as file:
        params = pickle.load(file)

    return params


def count_weights(params):
    params_tree = jax.tree_util.tree_map(jnp.size, params)
    leaves = jax.tree_leaves(params_tree)
    return sum(leaves)


def save_model_weights(model, savename):
    with open(savename, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def create_default_update_fn_aux(optimizer: optax.GradientTransformation,
                                 model_loss: Callable):
    """
    This function calls the update function, to implement the backpropagation
    """

    @jax.jit
    def update(params, opt_state, batch, rng) -> Tuple[hk.Params, optax.OptState, jnp.float32, jnp.ndarray]:

        print('Tracing update function...')
        out_, grads = jax.value_and_grad(model_loss, has_aux=True)(params, *batch)
        batch_loss, end_state = out_
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, batch_loss, end_state

    return update


def create_default_update_fn(optimizer: optax.GradientTransformation,
                             model_loss: Callable):
    """
    This function calls the update function, to implement the backpropagation
    """

    # @jax.jit
    def update(params, opt_state, batch, rng) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        print('Tracing update function...')

        batch_loss, grads = jax.value_and_grad(model_loss)(params, *batch)
        grads = jax.tree_util.tree_map(jnp.conj, grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, batch_loss

    return update
