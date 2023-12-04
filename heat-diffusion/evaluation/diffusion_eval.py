import sys
sys.path.append('..')

import jax.random as jr
import jax.numpy as jnp
import jax

from optimizers.Autoregressive import *

def predict(generator, params, forward_fn, model_weights, data_keys, rng, realizations = 8, version=0):
        
    DT = params['step_size']

    score_fn = get_score_fn(forward_fn)

    noise = params['diffusion']['inference']['noise']

    NSTEPS = int(params['t1'] / DT)
    
    data_elems = [(generator.load((list(generator.h5files.keys())[0], key), transform=False)[...,0] - params['data_mean']) / params['data_std'] for key in data_keys] 

    forward_ = [forward_full(data, params['t1'], params['t1'])[-1] for data in data_elems]

    forward_noise = []

    for elem in forward_:

        elem = elem + DT * noise * jr.normal(rng, shape=elem.shape)
        forward_noise.append(elem)
        rng, _ = jax.random.split(rng)


    forward_noise = jnp.stack(forward_noise)
    data_elems = jnp.stack(data_elems)

    return_ = {}

    for i in range(realizations):
        
        if version == 0: # Standard solution with noise
            print('Evaluating version 0')
            prediction = eval_backward_rand(forward_noise, model_weights, score_fn, DT, NSTEPS, rng, noise)[-1] # take final prediction

        elif version == 3: # Solution with no noise
            print('Evaluating version 3')
            prediction = eval_backward_rand(forward_noise, model_weights, score_fn, DT, NSTEPS, rng, 0.0)[-1]
            
        else:
            
            raise ValueError('Not implemented')
            
            
        rng, _ = jax.random.split(rng)

        return_[f'prediction_{i}'] = list(prediction)

    return_['ground_truth'] = list(data_elems)
    return_['input']= list(forward_noise)
    return_['forward']= list(forward_)

    return return_

def eval_backward_rand(initial_value, params, correction_fn, DT, STEPS, rng, noise):

    
    t0 = 0.0
    t1 = DT * STEPS
    args = None

    initial_shape = initial_value.shape
    
    y0 = initial_value
    
    tprev = jnp.array(t1 + DT) 
    tnext = jnp.array(t1) 
    
    y = y0

    y_list = [y]
    
    num_steps = int((t1-t0) / DT)
    
    print('num steps ', num_steps)
    
    # Predicor steps
    for i in tqdm(range(num_steps)):
        
        bm = noise * jax.random.normal(rng, shape=y.shape) * DT
        rng, _ = jax.random.split(rng)
        
        y = y + DT * correction_fn(params, jnp.expand_dims(y, axis=1), tnext.tile(y.shape[0]))
        
        tprev = tnext
        tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)
        
        if i < num_steps - 1:
            y = y + bm
            
        y_list.append(y)    
    
    return y_list