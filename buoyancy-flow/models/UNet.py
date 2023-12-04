import einops
import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax

from .grid import get_grid

def get_model(data_shape, use_grid=True, update_type=1, rng=jr.PRNGKey(2022)):
    """
    Returns a function that computes the forward pass of the model and model initialization parameters.
    :param data_shape: channels and number of dimensions of the input data
    :param use_grid: if True, grid (get_grid()) is used as input to the model
    :param update_type: If 1, only smoke, velocities and mask are inputs. If 2,
                        smoke_delta, velocities_delta are also inputs.
    :param rng: random number generator
    :return: function with forward pass of model and model parameters
    """
    
    grid = get_grid((1,) + data_shape)

    def f(x, t):
        
        print('tracing UNet')
        
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
       
        
        if use_grid:
            x = jnp.concatenate([x, t, grid_tiled], axis=1)
        else:
            x = jnp.concatenate([x, t], axis=1)
            
        x = jnp.array(x, dtype=jnp.float64)
        
        
        # Encoded - 64 x 64
        x_l1 = hk.Conv2D(16, 3, data_format='NCHW')(x)
        x_l1 = hk.Conv2D(16, 3, data_format='NCHW')(x_l1)
        x_l2 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l1)
        
        # 32 x 32
        x_l2 = hk.Conv2D(32, 3, data_format='NCHW')(x_l2)
        x_l2 = hk.Conv2D(32, 3, data_format='NCHW')(x_l2)
        x_l3 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l2)
        
        # 16 x 16
        x_l3 = hk.Conv2D(48, 3, data_format='NCHW')(x_l3)
        x_l3 = hk.Conv2D(48, 3, data_format='NCHW')(x_l3)
        x_l4 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l3)

        # 8 x 8
        x_l4 = hk.Conv2D(64, 3, data_format='NCHW')(x_l4)
        x_l4 = hk.Conv2D(64, 3, data_format='NCHW')(x_l4)
        x_l5 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l4)
        
        # 4 x 4
        x_l5 = hk.Conv2D(80, 3, data_format='NCHW')(x_l5)
        x_l5 = hk.Conv2D(80, 3, data_format='NCHW')(x_l5)
        x_l6 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l5)
        
        # 2 x 2
        x_l6 = hk.Conv2D(96, 3, data_format='NCHW')(x_l6)
        x_l6 = hk.Conv2D(96, 3, data_format='NCHW')(x_l6)
        x_l7 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l6)
        
        # 1 x 1
        x_l7 = hk.Conv2D(112, 1, data_format='NCHW')(x_l7)
        x_l7 = hk.Conv2D(112, 1, data_format='NCHW')(x_l7)
    
        x_u6 = hk.Conv2DTranspose(96, 3, stride=2, data_format='NCHW')(x_l7)
    
        # 2 x 2
        x_u6 = jnp.concatenate([x_l6, x_u6], axis=1)
        x_u6 = hk.Conv2D(96, 1, data_format='NCHW')(x_u6)
        
        x_u5 = hk.Conv2DTranspose(80, 3, stride=2, data_format='NCHW')(x_u6)
        
        # 4 x 4
        
        x_u5 = jnp.concatenate([x_l5, x_u5], axis=1)
        x_u5 = hk.Conv2D(80, 1, data_format='NCHW')(x_u5)
        
        x_u4 = hk.Conv2DTranspose(64, 3, stride=2, data_format='NCHW')(x_u5)
        
        # 8 x 8 
        
        x_u4 = jnp.concatenate([x_l4, x_u4], axis=1)
        x_u4 = hk.Conv2D(64, 1, data_format='NCHW')(x_u4)
        
        x_u3 = hk.Conv2DTranspose(48, 3, stride=2, data_format='NCHW')(x_u4)
        
        # 16 x 16
        
        x_u3 = jnp.concatenate([x_l3, x_u3], axis=1)
        x_u3 = hk.Conv2D(48, 1, data_format='NCHW')(x_u3)
        
        x_u2 = hk.Conv2DTranspose(32, 3, stride=2, data_format='NCHW')(x_u3)
        
        # 32 x 32
        
        x_u2 = jnp.concatenate([x_l2, x_u2], axis=1)
        x_u2 = hk.Conv2D(32, 1, data_format='NCHW')(x_u2)
        
        x_u1 = hk.Conv2DTranspose(16, 3, stride=2, data_format='NCHW')(x_u2)
        
        # 64 x 64
        
        x_u1 = jnp.concatenate([x_l1, x_u1], axis=1)
        x_u1 = hk.Conv2D(16, 3, data_format='NCHW')(x_u1)
        
        x = hk.Conv2D(3, 1, data_format='NCHW')(x_u1)
        
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
        params = init_params(rng, [smoke_init, smoke_init, vel_x_init, vel_x_init, vel_y_init, vel_y_init, mask_init], t_init)
    else:
        raise ValueError(f'Update type {update_type} not implemented')
    
    return forward_fn, params