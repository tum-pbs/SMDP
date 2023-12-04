import jax.numpy as jnp


def get_grid(shape):
    batch_size, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = jnp.linspace(0, 1, size_x)
    gridx = jnp.reshape(gridx, (1, size_x, 1))
    gridx = jnp.tile(gridx, [batch_size, 1, size_y])
    gridy = jnp.linspace(0, 1, size_y)
    gridy = jnp.reshape(gridy, (1, 1, size_y))
    gridy = jnp.tile(gridy, [batch_size, size_x, 1])
    grid = jnp.stack([gridx, gridy], axis=1)

    return grid
