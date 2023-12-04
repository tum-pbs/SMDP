from random import shuffle as shuffleList
from random import choice
import jax.numpy as jnp
from skimage.transform import resize

import numpy as np
from copy import deepcopy
import os
import jax.random as jr
from scipy import fftpack

import h5py


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def listdir_nohidden(path, suffix=None):
    list_ = []
    for f in os.listdir(path):
        if (not f.startswith('.')):
            if suffix is None:
                list_.append(f)
            else:
                if f.endswith(suffix):
                    list_.append(f)
    return list_


def make_generator(dl):
    def f():
        for x, y in dl:
            for n in range(x.shape[0]):
                yield x[n], y[n]

    return f


class DataLoader(object):

    def __init__(self, files, keys, batch_size, rng=jr.PRNGKey(2022), shuffle=True, resolution=32, name='',
                 augmentation=None, t1=10.0):

        self.augmentation = augmentation
        self.name = name
        self.t1 = t1
        self.resolution = resolution
        self.rng = rng
        self.keys = deepcopy(keys)
        self.shuffle = shuffle
        self.batchSize = batch_size
        self.transform = []

        self.map_ = None

        self._open_h5file(files)
        # self.keys = list(self.h5file.keys())

        self.numSamples = len(self.keys)

        self.normalize = True
        self.mean = jnp.array([0., 0., 0.])
        self.std = jnp.array([1., 1., 1.])

        if self.shuffle:
            shuffleList(self.keys)

        self.batches = list(chunks(self.keys, self.batchSize))
        self.numBatches = len(self.batches)

        print("Length: %d" % self.numSamples)

    def _open_h5file(self, paths):

        self.h5files = {}
        for path in paths:
            self.h5files[path] = h5py.File(path, 'r')

        return self.h5files

    def __del__(self):
        try:
            if self.h5files is not None:
                for file in self.h5files:
                    self.h5files[file].close()

        finally:
            self.h5files = None

    def load(self, key, transform=True, augmentation=None):

        file, sample_id = key
        data = self.h5files[file][str(sample_id)][:]

        data = np.expand_dims(data, axis=-1)

        data = jnp.array(resize(data, (self.resolution, self.resolution, 1)))

        if not augmentation is None:
            data = augmentation(data)

        if transform:
            flip_axes = []

            if choice([0, 1]) == 1:
                flip_axes.append(0)
            if choice([0, 1]) == 1:
                flip_axes.append(1)
            data = jnp.flip(data, axis=flip_axes)

            rotation = choice([0, 1, 2, 3])
            data = jnp.rot90(data, k=rotation, axes=(0, 1))

        return data

    def __len__(self):

        return self.numBatches

    def set_mean_and_std(self, mean, std):

        self.normalize = True
        self.mean = mean
        self.std = std

    def __getitem__(self, index, forward=False):

        data = []
        data_fft = []

        for sample in self.batches[index]:

            item = self.load(sample, augmentation=self.augmentation)
            data.append(item)

            if forward:
                field_fft = fftpack.fft2(np.array(item[..., 0]))
                field_shiftfft = jnp.fft.fftshift(field_fft)
                data_fft.append(jnp.array(field_shiftfft))

        item = jnp.stack(data)

        item_forward = 0
        t = 0
        if forward:
            item_fft = jnp.stack(data_fft)
            batch_size = item.shape[0]
            t = jr.uniform(self.rng, (batch_size,), minval=0, maxval=self.t1 / batch_size)
            self.rng, _ = jr.split(self.rng, 2)
            t = t + (self.t1 / batch_size) * jnp.arange(batch_size)

            data_shape = item.shape
            if self.map_ is None:
                self.map_ = np.zeros(shape=data_shape[1:3])
                for i in range(data_shape[1]):
                    for j in range(data_shape[2]):
                        self.map_[i, j] = np.sqrt((i - data_shape[1] / 2) ** 2 + (j - data_shape[2] / 2) ** 2)
                self.map_ = jnp.stack([self.map_] * self.batchSize)
                self.map_ = -jnp.square(self.map_)

            map_stacked = self.map_[:item.shape[0]]
            map_stacked = jnp.einsum('ijk,i->ijk', map_stacked, t)
            map_stacked = jnp.exp(map_stacked)

            item_forward = jnp.multiply(map_stacked.astype(jnp.complex64), item_fft)

            item_forward = fftpack.ifft2(np.fft.ifftshift(item_forward, axes=(1, 2)), axes=(1, 2))
            item_forward = jnp.real(item_forward)

        if self.normalize:

            item = (item - self.mean) / self.std

            if forward:
                item_forward = (item_forward - self.mean) / self.std

        if forward:
            return item, item_forward, t

        return item

    def on_epoch_end(self):

        if self.shuffle:
            shuffleList(self.keys)
            self.batches = list(chunks(self.keys, self.batchSize))

    def setTransformations(self, transformations):

        self.transform = transformations

    def get_norm(self, mean=None, std=None):

        mean = 0
        std = 0
        max_ = - jnp.inf
        min_ = jnp.inf

        count_elems = 0

        normalize_ = self.normalize
        self.normalize = False

        for i in range(self.__len__()):
            batch = self.__getitem__(i, forward=False)

            count_elems += batch.shape[0] * batch.shape[1] * batch.shape[2]
            mean += jnp.sum(batch, axis=(0, 1, 2))
            max_ = jnp.maximum(max_, jnp.max(batch))
            min_ = jnp.minimum(min_, jnp.min(batch))

        mean = mean / count_elems

        for i in range(self.__len__()):
            batch = self.__getitem__(i)

            std += jnp.sum(jnp.abs(batch - mean) ** 2, axis=(0, 1, 2))

        std = std / count_elems
        std = jnp.sqrt(std)

        self.normalize = normalize_

        return mean, std, min_, max_

    def __iter__(self):
        """Create a generator that iterates over the sequence"""
        for item in (self[i] for i in range(len(self))):
            yield item
