import einops
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax


def get_model(data_shape, time_dim=True, two_step=False, dropout_rate=0.5, rng=None):
    def f(x, key):

        print('Tracing encoder decoder model...')

        key = hk.PRNGSequence(key)

        # print(f'f: {x.shape}')

        batch_size, channels, height, width = x.shape
        # t = einops.repeat(t, "-> 1 h w", h=height, w=width)
        # x = jnp.concatenate([x, t])

        padding = 16

        x = jnp.tile(x, [1, 1, 3, 3])

        x = x[:, :, width - padding: 2 * width + padding, width - padding: 2 * width + padding]

        x = hk.Conv2D(32, 4, data_format='NCHW')(x)

        x_block1 = hk.Conv2D(32, 4, data_format='NCHW')(x)
        x_block1 = jax.nn.leaky_relu(x_block1)
        x_block1 = hk.Conv2D(32, 4, data_format='NCHW')(x_block1)
        x_block1 = jax.nn.leaky_relu(x_block1)

        x_block1 = x_block1 + x

        x_block2 = hk.Conv2D(32, 4, data_format='NCHW')(x_block1)
        x_block2 = jax.nn.leaky_relu(x_block2)
        x_block2 = hk.Conv2D(32, 4, data_format='NCHW')(x_block2)
        x_block2 = jax.nn.leaky_relu(x_block2)

        x_block2 = x_block2 + x_block1

        x_block3 = hk.Conv2D(32, 4, data_format='NCHW')(x_block2)
        x_block3 = jax.nn.leaky_relu(x_block3)
        x_block3 = hk.Conv2D(32, 4, data_format='NCHW')(x_block3)
        x_block3 = jax.nn.leaky_relu(x_block3)

        x_block3 = x_block3 + x_block2

        x_block4 = hk.Conv2D(32, 4, data_format='NCHW')(x_block3)
        x_block4 = jax.nn.leaky_relu(x_block4)
        x_block4 = hk.Conv2D(32, 4, data_format='NCHW')(x_block4)
        x_block4 = jax.nn.leaky_relu(x_block4)

        x_block4 = x_block4 + x_block3

        x_block5 = hk.Conv2D(32, 4, data_format='NCHW')(x_block4)
        x_block5 = jax.nn.leaky_relu(x_block5)
        x_block5 = hk.Conv2D(32, 4, data_format='NCHW')(x_block5)
        x_block5 = jax.nn.leaky_relu(x_block5)

        x_block5 = x_block5 + x_block4

        encoder = hk.Conv2D(1, 1, data_format='NCHW')(x_block5)

        dec_x_init = hk.Conv2DTranspose(32, 4, data_format='NCHW')(encoder)

        dec_x_block1 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_init)
        dec_x_block1 = jax.nn.leaky_relu(dec_x_block1)
        dec_x_block1 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block1)
        dec_x_block1 = jax.nn.leaky_relu(dec_x_block1)
        dec_x_block1 = hk.dropout(next(key), dropout_rate, dec_x_block1)

        dec_x_block1 = dec_x_block1 + dec_x_init

        dec_x_block2 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block1)
        dec_x_block2 = jax.nn.leaky_relu(dec_x_block2)
        dec_x_block2 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block2)
        dec_x_block2 = jax.nn.leaky_relu(dec_x_block2)
        dec_x_block2 = hk.dropout(next(key), dropout_rate, dec_x_block2)

        dec_x_block2 = dec_x_block2 + dec_x_block1

        dec_x_block3 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block2)
        dec_x_block3 = jax.nn.leaky_relu(dec_x_block3)
        dec_x_block3 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block3)
        dec_x_block3 = jax.nn.leaky_relu(dec_x_block3)
        dec_x_block3 = hk.dropout(next(key), dropout_rate, dec_x_block3)

        dec_x_block3 = x_block3 + dec_x_block2

        dec_x_block4 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block3)
        dec_x_block4 = jax.nn.leaky_relu(dec_x_block4)
        dec_x_block4 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block4)
        dec_x_block4 = jax.nn.leaky_relu(dec_x_block4)
        dec_x_block4 = hk.dropout(next(key), dropout_rate, dec_x_block4)

        dec_x_block4 = dec_x_block4 + dec_x_block3

        dec_x_block5 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block4)
        dec_x_block5 = jax.nn.leaky_relu(dec_x_block5)
        dec_x_block5 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block5)
        dec_x_block5 = jax.nn.leaky_relu(dec_x_block5)

        net = hk.Conv2D(1, 5, data_format='NCHW')(dec_x_block5)

        return net[:, 0, padding:-padding, padding:-padding]

    def f_(x, t, key):

        batch_size, _, height, width = x.shape

        t = einops.repeat(t, "c -> c 1 h w", h=height, w=width)

        x = jnp.concatenate([x, t], axis=1)

        x = f(x, key)

        return x

    if two_step:
        x_init = jnp.ones(data_shape)[None]
        x_init = jnp.concatenate([x_init, x_init], axis=1)
    else:
        x_init = jnp.ones((1,) + data_shape)

    if rng is None:
        rng = jr.PRNGKey(0)

    if time_dim:

        t_init = jnp.array([0.0])
        init_params, forward_fn = hk.transform(f_)
        params = init_params(rng, x_init, t_init, rng)

    else:

        init_params, forward_fn = hk.transform(f)
        params = init_params(rng, x_init, rng)

    return forward_fn, params
