import einops
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax

from .grid import get_grid

def get_model(data_shape, update_type=1, rng=jax.random.PRNGKey(2022)):
    """Returns a function that computes the forward pass of the model.
    :param data_shape: channels and number of dimensions of the input data
    :param update_type: If 1, only smoke, velocities and mask are inputs. If 2,
                        smoke_delta, velocities_delta are also inputs.
    :param rng: random number generator
    :return: function with forward pass of model and model parameters
    """
    
    grid = get_grid((1,) + data_shape)

    def f(x, t):

        smoke, vel_x, vel_y, mask = x
        
        batch_size, height, width = smoke.shape
        
        if update_type==1:
            smoke, vel_x, vel_y, mask = x
        
            batch_size, height, width = smoke.shape
        
            vel_x = jnp.pad(vel_x, ((0,0), (0,0), (1,0)), mode='edge')
            vel_y = jnp.pad(vel_y, ((0,0), (1,0), (0,0)), mode='edge')
        
            x = jnp.stack([smoke, vel_x, vel_y, mask], axis=1)
            
        elif update_type==2:
            smoke, smoke_delta, vel_x, vel_x_delta, vel_y, vel_y_delta, mask = x
        
            batch_size, height, width = smoke.shape
        
            vel_x = jnp.pad(vel_x, ((0,0), (0,0), (1,0)), mode='edge')
            vel_y = jnp.pad(vel_y, ((0,0), (1,0), (0,0)), mode='edge')
            vel_x_delta = jnp.pad(vel_x_delta, ((0,0), (0,0), (1,0)), mode='edge')
            vel_y_delta = jnp.pad(vel_y_delta, ((0,0), (1,0), (0,0)), mode='edge')
            
            x = jnp.stack([smoke, smoke_delta, vel_x, vel_x_delta, vel_y, vel_y_delta, mask], axis=1)
            
        else:
            raise ValueError(f'Update type {update_type}')
        
        t = einops.repeat(t, "->b 1 h w", b=batch_size, h=height, w=width)
        grid_tiled = jnp.tile(grid, [batch_size, 1, 1, 1])
        
        x = jnp.concatenate([x, t, grid_tiled], axis=1)
        
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

        dec_x_block1 = dec_x_block1 + dec_x_init

        dec_x_block2 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block1)
        dec_x_block2 = jax.nn.leaky_relu(dec_x_block2)
        dec_x_block2 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block2)
        dec_x_block2 = jax.nn.leaky_relu(dec_x_block2)

        dec_x_block2 = dec_x_block2 + dec_x_block1

        dec_x_block3 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block2)
        dec_x_block3 = jax.nn.leaky_relu(dec_x_block3)
        dec_x_block3 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block3)
        dec_x_block3 = jax.nn.leaky_relu(dec_x_block3)

        dec_x_block3 = x_block3 + dec_x_block2

        dec_x_block4 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block3)
        dec_x_block4 = jax.nn.leaky_relu(dec_x_block4)
        dec_x_block4 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block4)
        dec_x_block4 = jax.nn.leaky_relu(dec_x_block4)

        dec_x_block4 = dec_x_block4 + dec_x_block3
        
        dec_x_block5 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block4)
        dec_x_block5 = jax.nn.leaky_relu(dec_x_block5)
        dec_x_block5 = hk.Conv2DTranspose(32, 4, data_format='NCHW')(dec_x_block5)
        dec_x_block5 = jax.nn.leaky_relu(dec_x_block5)
        
        net = hk.Conv2D(3, 5, data_format='NCHW')(dec_x_block5)

        # smoke, vel_x, vel_y
        return net[:, 0], net[:, 1, :, 1:], net[:, 2, 1:, :]
        # return net
        
    init_params, forward_fn = hk.transform(f)
    
    t_init = jnp.array(0.0)
    vel_x_init = jnp.ones((1, data_shape[0], data_shape[1]-1))
    vel_y_init = jnp.ones((1, data_shape[0]-1, data_shape[1]))
    smoke_init = jnp.ones((1,) + data_shape)
    mask_init = jnp.ones((1,) + data_shape)
    
    params = init_params(rng, [smoke_init, vel_x_init, vel_y_init, mask_init], t_init)
    
    return forward_fn, params