import haiku as hk
import jax
import jax.numpy as jnp


# https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
class SpectralConv2d(hk.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))

        def scale_initializer(key):
            k1, k2 = jax.random.split(key, 2)

            def f_(shape, dtype):
                real_ = jax.random.uniform(k1, shape=shape)

                complex_ = jax.random.uniform(k2, shape=shape)

                i_ = real_ + complex_ * 1j

                return self.scale * i_

            return f_

        self.weights1 = hk.get_parameter("weights1",
                                         [in_channels, out_channels, modes1, modes2],
                                         init=scale_initializer(jax.random.PRNGKey(1)),
                                         dtype=jnp.complex64)

        self.weights2 = hk.get_parameter("weights2",
                                         [in_channels, out_channels, modes1, modes2],
                                         init=scale_initializer(jax.random.PRNGKey(2)),
                                         dtype=jnp.complex64)

    def compl_mul2d(self, input, weights):
        return jnp.einsum("bixy,ioxy->boxy", input, weights)

    def __call__(self, x):
        batch_size = x.shape[0]

        x_ft = jnp.fft.rfft2(x)

        out_ft = jnp.zeros(shape=(batch_size, self.out_channels, x.shape[-2], int(x.shape[-1] // 2) + 1),
                           dtype=jnp.complex64)

        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].set(
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],
                             self.weights1))

        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].set(
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                             self.weights2))

        x = jnp.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))

        return x


class FNO2d(hk.Module):
    def __init__(self, modes1, modes2, width, is_training=True):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2

        self.fc0 = hk.Linear(width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = hk.Conv2D(width, 1, data_format='NCHW')
        self.w1 = hk.Conv2D(width, 1, data_format='NCHW')
        self.w2 = hk.Conv2D(width, 1, data_format='NCHW')
        self.w3 = hk.Conv2D(width, 1, data_format='NCHW')

        self.bn0 = hk.BatchNorm(True, True, 0.9)
        self.bn1 = hk.BatchNorm(True, True, 0.9)
        self.bn2 = hk.BatchNorm(True, True, 0.9)
        self.bn3 = hk.BatchNorm(True, True, 0.9)

        self.fc1 = hk.Linear(128)
        self.fc2 = hk.Linear(1)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = jnp.linspace(0, 1, size_x)
        gridx = jnp.reshape(gridx, (1, size_x, 1, 1))
        gridx = jnp.tile(gridx, [batchsize, 1, size_y, 1])
        gridy = jnp.linspace(0, 1, size_y)
        gridy = jnp.reshape(gridy, (1, 1, size_y, 1))
        gridy = jnp.tile(gridy, [batchsize, size_x, 1, 1])
        grid = jnp.concatenate([gridx, gridy], axis=-1)

        return grid

    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))

        grid = self.get_grid(x.shape)

        x = jnp.concatenate([x, grid], axis=-1)
        x = self.fc0(x)
        x = jnp.transpose(x, (0, 3, 1, 2))

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = jax.nn.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = jax.nn.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = jax.nn.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = jnp.transpose(x, (0, 2, 3, 1))

        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)

        x = jnp.transpose(x, (0, 3, 1, 2))

        return x


def get_model(modes, width, init=True):
    def model(x):
        fno = FNO2d(modes, modes, width)

        return fno(x)

    if not init:
        return model

    init_params, forward_fn = hk.transform(model)

    x = jnp.ones(shape=(16, 1, 64, 64))

    key = jax.random.PRNGKey(0)
    model_weights = init_params(key, x)

    return forward_fn, model_weights
