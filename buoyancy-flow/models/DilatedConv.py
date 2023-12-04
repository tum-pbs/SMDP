import einops
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax
from .grid import get_grid

def get_model(data_shape, nBlocks=4, nFeatures=48, update_type=1, rng=jr.PRNGKey(2022)):
    """
    Returns a function that computes the forward pass of the model and model initialization parameters.
    :param data_shape: channels and number of dimensions of the input data
    :param nBlocks: number of processing blocks in the model
    :param nFeatures: number of channels/filters in the model
    :param update_type: If 1, only smoke, velocities and mask are inputs. If 2,
                        smoke_delta, velocities_delta are also inputs.
    :param rng: random number generator
    :return: function with forward pass of model and model parameters
    """
    
    grid = get_grid((1,) + data_shape)

    def f(x, t):
        
        print('tracing dilated conv model..')
       
        if update_type == 1:
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
      
        x = jnp.array(x, dtype=jnp.float64)
    
        # Encoder
        x = hk.Conv2D(nFeatures, 3, data_format='NCHW')(x)

        x_res = x
        
        for _ in range(nBlocks):
            
            x = hk.Conv2D(nFeatures, 3, rate=1, data_format='NCHW')(x)
            x = jax.nn.relu(x)
            x = hk.Conv2D(nFeatures, 3, rate=2, data_format='NCHW')(x)
            x = jax.nn.relu(x)

            # Commented out to reduce number of model parameters
            # x = hk.Conv2D(nFeatures, 3, rate=4, data_format='NCHW')(x)
            # x = jax.nn.relu(x)
            # x = hk.Conv2D(nFeatures, 3, rate=8, data_format='NCHW')(x)
            # x = jax.nn.relu(x)
            # x = hk.Conv2D(nFeatures, 3, rate=4, data_format='NCHW')(x)
            # x = jax.nn.relu(x)


            x = hk.Conv2D(nFeatures, 3, rate=2, data_format='NCHW')(x)
            x = jax.nn.relu(x)
            x = hk.Conv2D(nFeatures, 3, rate=1, data_format='NCHW')(x)
            x = jax.nn.relu(x)
            
            x = x + x_res
            x_res = x
        
        # Decoder
        x = hk.Conv2D(3, 3, data_format='NCHW')(x)
        
        # smoke, vel_x, vel_y
        return x[:, 0], x[:, 1, :, 1:], x[:, 2, 1:, :]
        
    init_params, forward_fn = hk.transform(f)
    
    t_init = jnp.array(0.0, dtype=jnp.float32)
    vel_x_init = jnp.ones((1, data_shape[0], data_shape[1]-1), dtype=jnp.float32)
    vel_y_init = jnp.ones((1, data_shape[0]-1, data_shape[1]), dtype=jnp.float32)
    smoke_init = jnp.ones((1,) + data_shape, dtype=jnp.float32)
    mask_init = jnp.ones((1,) + data_shape, dtype=jnp.float32)

    if update_type == 1:
        params = init_params(rng, [smoke_init, vel_x_init, vel_y_init, mask_init], t_init)
    elif update_type == 2:
        params = init_params(rng, [smoke_init, smoke_init, vel_x_init, vel_x_init,
                                   vel_y_init, vel_y_init, mask_init], t_init)
    else:
        raise ValueError(f'Update type {update_type} not implemented')
    
    
    return forward_fn, params