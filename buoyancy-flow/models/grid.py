import jax.numpy as jnp

def get_grid(shape):
    """Returns a grid of size shape, with values in [0, 1]
    :param shape: shape of the grid
    """
        
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = jnp.linspace(0, 1, size_x)
    gridx = jnp.reshape(gridx, (1, size_x, 1))
    gridx = jnp.tile(gridx, [batchsize, 1, size_y])
    gridy = jnp.linspace(0, 1, size_y)
    gridy = jnp.reshape(gridy, (1, 1, size_y))
    gridy = jnp.tile(gridy, [batchsize, size_x, 1])
    grid = jnp.stack([gridx, gridy], axis=1)

    return jnp.array(grid, dtype=jnp.float32)