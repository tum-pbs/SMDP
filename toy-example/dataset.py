import os
import os.path

import pickle

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from processes import toy_sde
from utils.utils import get_batches

DATA_DIR = "data"


def build_dataset(samples: int, drift, diffusion: float, dt: float = 0.02, seed: int = 0, t0: float = 0.0,
                  t1: float = 10.0) -> jnp.ndarray:
    """
    Build dataset of samples of toy SDEs
    Parameters
    ----------
    samples : int number of samples in the dataset
    drift function (independent of time); takes x as input and returns float
    diffusion : float diffusion coefficient of brownian motion
    dt : float time step
    seed : int seed for jax random number generator
    t0 : float start time, usually 0
    t1 : float end time, usually 10
    -------

    """

    #  dataset = []
    x = jnp.linspace(t0, t1, int((t1 - t0) / dt))

    key = jr.PRNGKey(seed)

    def run_sde(x0, key):
        return jnp.stack([x, jnp.diag(toy_sde(drift, x0, diffusion, key, t0, t1).evaluate(x))], axis=0)

    x0_list = []

    for batch in get_batches(list(range(samples)), 1000):

        key_list = []

        for _ in range(len(batch)):
            key_list.append(key)
            key = jr.split(key)[0]

        key_list = jnp.array(key_list)
        order_ = jnp.power(jnp.ones(len(batch)) * (-1), jnp.arange(len(batch)))

        x0_list.append(vmap(run_sde, in_axes=(0, 0))(order_, key_list))

    dataset = jnp.concatenate(x0_list)

    return dataset

    #  for n in tqdm(range(samples)):
    #    x0 = (-1) ** n
    #    solution = toy_sde(drift, x0, diffusion, key, t0, t1)
    #    dataset.append((x, jnp.diag(solution.evaluate(x))))
    #    key = jr.split(key)[0]

    #  return jnp.array(dataset)


def get_dataset(name: str, rebuild=False, **kwargs) -> jnp.ndarray:
    """
    Get dataset from file or build it
    Parameters
    ----------
    name : str dataset name
    rebuild : bool rebuild dataset
    kwargs : dict parameters for build_dataset
    -------

    """
    file_path = os.path.join(DATA_DIR, f'{name}.p')

    if os.path.isfile(file_path) and not rebuild:

        with open(file_path, 'rb') as file:

            dataset = pickle.load(file)

            return dataset
    else:

        dataset = build_dataset(**kwargs)

        with open(file_path, 'wb') as file:

            pickle.dump(dataset, file)

        return dataset


def prepare_batch(batch: jnp.ndarray) -> jnp.ndarray:
    """
    Prepare batch for training. Reverse time axis
    Parameters
    ----------
    batch : jnp.ndarray (batch_size, n_steps, 2)
    -------
    """
    return batch[:, :, ::-1]


def iterate_batches(dataset: jnp.ndarray, batch_size: int, shuffle=False, key=None):
    """
    Iterate in batches over dataset
    Parameters
    ----------
    dataset: dataset (n_samples, n_steps, 2)
    batch_size: batch size
    shuffle: shuffle dataset
    key: jr.PRNGKey
    """

    n_samples = dataset.shape[0]

    ids = jnp.arange(n_samples)
    sample_perm = jnp.arange(n_samples)

    if batch_size is None:
        batch_size = n_samples
    if shuffle:
        if key is None:
            key = jr.PRNGKey(0)
        sample_perm = jr.permutation(key, n_samples)

    batch_idx = 0
    num_batches = jnp.ceil(n_samples / batch_size)

    while batch_idx < num_batches:
        start = batch_idx * batch_size
        end = min(n_samples, (batch_idx + 1) * batch_size)

        indices = list(range(start, end))
        perm_indices = sample_perm[jnp.array(indices)]
        dataset_batch = dataset[perm_indices]
        ids_batch = ids[perm_indices]

        batch_idx += 1
        yield dataset_batch, ids_batch
