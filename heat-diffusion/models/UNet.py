import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import jax

from grid import get_grid


def spatial_dropout(rng: jr.PRNGKey, rate: float, x: jnp.ndarray) -> jnp.ndarray:
    try:
        if rate < 0 or rate >= 1:
            raise ValueError("rate must be in [0, 1).")

        if rate == 0.0:
            return x, rng
    except jax.errors.ConcretizationTypeError:
        pass

    keep_rate = 1.0 - rate
    keep = jax.random.bernoulli(rng, keep_rate, shape=x.shape[:2])
    rng, _ = jax.random.split(rng)
    x_ = jnp.einsum("bixy,bi->bixy", x, keep)
    
    return x_ / keep_rate, rng

def get_model(data_shape, use_grid=False, rng = None, dropout_rate=0.0):
    
    grid = get_grid((1,) + data_shape)

    def f(x, rng):

        batch_size, channels, height, width = x.shape
        # t = einops.repeat(t, "-> 1 h w", h=height, w=width)
        # x = jnp.concatenate([x, t])
        
        padding = 16
        
        x = jnp.tile(x, [1,1,3,3])
    
        x = x[:, :, width-padding : 2*width + padding, width-padding : 2*width + padding]
        
        grid_tiled = jnp.tile(grid, [batch_size, 1, 1, 1])
        
        if use_grid:
            x = jnp.concatenate([x, grid_tiled], axis=1)
        
        # Encoded - 64 x 64
        x_l1 = hk.Conv2D(16, 3, data_format='NCHW')(x)
        x_l1, rng = spatial_dropout(rng, dropout_rate, x_l1)
        x_l1 = hk.Conv2D(16, 3, data_format='NCHW')(x_l1)
        x_l1, rng = spatial_dropout(rng, dropout_rate, x_l1)
        x_l2 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l1)
        
        # 32 x 32
        x_l2 = hk.Conv2D(32, 3, data_format='NCHW')(x_l2)
        x_l2, rng = spatial_dropout(rng, dropout_rate, x_l2)
        x_l2 = hk.Conv2D(32, 3, data_format='NCHW')(x_l2)
        x_l2, rng = spatial_dropout(rng, dropout_rate, x_l2)
        x_l3 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l2)
        
        # 16 x 16
        x_l3 = hk.Conv2D(48, 3, data_format='NCHW')(x_l3)
        x_l3, rng = spatial_dropout(rng, dropout_rate, x_l3)
        x_l3 = hk.Conv2D(48, 3, data_format='NCHW')(x_l3)
        x_l3, rng = spatial_dropout(rng, dropout_rate, x_l3)
        x_l4 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l3)

        # 8 x 8
        x_l4 = hk.Conv2D(64, 3, data_format='NCHW')(x_l4)
        x_l4, rng = spatial_dropout(rng, dropout_rate, x_l4)
        x_l4 = hk.Conv2D(64, 3, data_format='NCHW')(x_l4)
        x_l4, rng = spatial_dropout(rng, dropout_rate, x_l4)
        x_l5 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l4)
        
        # 4 x 4
        x_l5 = hk.Conv2D(80, 3, data_format='NCHW')(x_l5)
        x_l5, rng = spatial_dropout(rng, dropout_rate, x_l5)
        x_l5 = hk.Conv2D(80, 3, data_format='NCHW')(x_l5)
        x_l5, rng = spatial_dropout(rng, dropout_rate, x_l5)
        x_l6 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l5)
        
        # 2 x 2
        x_l6 = hk.Conv2D(96, 3, data_format='NCHW')(x_l6)
        x_l6, rng = spatial_dropout(rng, dropout_rate, x_l6)
        x_l6 = hk.Conv2D(96, 3, data_format='NCHW')(x_l6)
        x_l6, rng = spatial_dropout(rng, dropout_rate, x_l6)
        x_l7 =  hk.MaxPool((2,2), (2,2), 'SAME')(x_l6)
        
        # 1 x 1
        x_l7 = hk.Conv2D(112, 1, data_format='NCHW')(x_l7)
        x_l7 = hk.Conv2D(112, 1, data_format='NCHW')(x_l7)
    
        x_u6 = hk.Conv2DTranspose(96, 3, stride=2, data_format='NCHW')(x_l7)
        x_u6, rng = spatial_dropout(rng, dropout_rate, x_u6)
    
        # 2 x 2
        x_u6 = jnp.concatenate([x_l6, x_u6], axis=1)
        x_u6 = hk.Conv2D(96, 1, data_format='NCHW')(x_u6)
        x_u6, rng = spatial_dropout(rng, dropout_rate, x_u6)
        x_u5 = hk.Conv2DTranspose(80, 3, stride=2, data_format='NCHW')(x_u6)
        x_u5, rng = spatial_dropout(rng, dropout_rate, x_u5)
        
        # 4 x 4
        
        x_u5 = jnp.concatenate([x_l5, x_u5], axis=1)
        x_u5 = hk.Conv2D(80, 1, data_format='NCHW')(x_u5)
        x_u5, rng = spatial_dropout(rng, dropout_rate, x_u5)
        x_u4 = hk.Conv2DTranspose(64, 3, stride=2, data_format='NCHW')(x_u5)
        x_u4, rng = spatial_dropout(rng, dropout_rate, x_u4)
        # 8 x 8 
        
        x_u4 = jnp.concatenate([x_l4, x_u4], axis=1)
        x_u4 = hk.Conv2D(64, 1, data_format='NCHW')(x_u4)
        x_u4, rng = spatial_dropout(rng, dropout_rate, x_u4)
        x_u3 = hk.Conv2DTranspose(48, 3, stride=2, data_format='NCHW')(x_u4)
        x_u3, rng = spatial_dropout(rng, dropout_rate, x_u3)
        # 16 x 16
        
        x_u3 = jnp.concatenate([x_l3, x_u3], axis=1)
        x_u3 = hk.Conv2D(48, 1, data_format='NCHW')(x_u3)
        x_u3, rng = spatial_dropout(rng, dropout_rate, x_u3)
        x_u2 = hk.Conv2DTranspose(32, 3, stride=2, data_format='NCHW')(x_u3)
        x_u2, rng = spatial_dropout(rng, dropout_rate, x_u2)
        # 32 x 32
        
        x_u2 = jnp.concatenate([x_l2, x_u2], axis=1)
        x_u2 = hk.Conv2D(32, 1, data_format='NCHW')(x_u2)
        x_u2, rng = spatial_dropout(rng, dropout_rate, x_u2)
        x_u1 = hk.Conv2DTranspose(16, 3, stride=2, data_format='NCHW')(x_u2)
        x_u1, rng = spatial_dropout(rng, dropout_rate, x_u1)
        
        # 64 x 64
        
        x_u1 = jnp.concatenate([x_l1, x_u1], axis=1)
        x_u1 = hk.Conv2D(16, 3, data_format='NCHW')(x_u1)
        x_u1, rng = spatial_dropout(rng, dropout_rate, x_u1)
        
        x = hk.Conv2D(3, 5, data_format='NCHW')(x_u1)
        
        return x[:, 0, padding:-padding, padding:-padding]
        
    init_params, forward_fn = hk.transform(f)
    
    t_init = jnp.array(0.0)
    state = jnp.ones(shape=data_shape)[None]
    
    if rng is None:
        rng = jax.random.PRNGKey(2022)
    
    params = init_params(rng, state, rng)
    
    return forward_fn, params