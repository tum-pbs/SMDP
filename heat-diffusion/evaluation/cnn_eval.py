import sys
sys.path.append('..')

import jax.random as jr
import jax.numpy as jnp
import jax

from optimizers.Bayesian import *

def predict(generator, params, forward_fn, model_weights , data_keys, rng, realizations = 8):
        
        data_elems = [(generator.load((list(generator.h5files.keys())[0], key), transform=False)[...,0] - params['data_mean']) / params['data_std'] for key in data_keys] 
        
        forward_ = [forward_full(data, params['t1'], params['t1'])[-1] for data in data_elems]
        
        forward_noise = []
        
        for elem in forward_:
            elem = elem + params['bayesian']['noise'] * jr.normal(rng, shape=elem.shape)
            forward_noise.append(elem)
            rng, _ = jax.random.split(rng)
          
        
        forward_noise = jnp.stack(forward_noise)
        data_elems = jnp.stack(data_elems)
        
        return_ = {}
        
        pred_fn = lambda weights, data, rng: forward_fn(weights, rng, data, rng)
        pred_fn = jax.jit(pred_fn)
        
        for i in range(realizations):
          
            prediction = forward_noise + pred_fn(model_weights, jnp.expand_dims(forward_noise, axis=1), rng)
            rng, _ = jax.random.split(rng)
           
            return_[f'prediction_{i}'] = list(prediction)
        
        return_['ground_truth'] = list(data_elems)
        return_['input']= list(forward_noise)
        return_['forward'] = forward_
        
        return return_