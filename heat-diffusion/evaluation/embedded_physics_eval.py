import sys
sys.path.append('..')

import jax.random as jr
import jax.numpy as jnp
import jax

from optimizers.EmbeddedPhysics import *

def predict(generator, params, forward_fn, model_weights, data_keys, rng, realizations = 8, version=0, noise_start=0.01, update_step=1):
        
    inference_correction_probability_flow_fn = get_correction_term_probability_flow(forward_fn, params['inference']['noise'])
    inference_correction_reverse_sde_fn = get_correction_term_reverse_sde(forward_fn, params['inference']['noise'])

    noise = params['inference']['noise']
    
    correction_reverse_sde_fn = lambda state, t: inference_correction_reverse_sde_fn(model_weights, state, t)
    
    correction_probability_flow_fn = lambda state, t: inference_correction_probability_flow_fn(model_weights, state, t)
    
    data_elems = [(generator.load((list(generator.h5files.keys())[0], key), transform=False)[...,0] - params['data_mean']) / params['data_std'] for key in data_keys] 

    t0 = params['t0']
    t1 = params['t1']
    num_steps = params['num_steps']
    if params['time_format'] == 'log':
        times = get_times_log(t0, t1, num_steps)[::-1]
        log_time = True
    elif params['time_format'] == 'linear':
        times = get_times_linear(t0, t1, num_steps)[::-1]
        log_time = False
    else:
        raise ValueError('Unknown time format ', self.params['time_format'])
    
    forward_ = [forward_full(data, params['t1'], params['t1'])[-1] for data in data_elems]   
    forward_noise = []

    # for data in data_elems:
    #     forward_noise.append(eval_forward(data, times[::-1], rng = rng, noise = noise)[-1])
    #     rng, _ = jax.random.split(rng)
    
    for elem in forward_:

        elem = elem + noise_start * jr.normal(rng, shape=elem.shape)
        forward_noise.append(elem)
        rng, _ = jax.random.split(rng)

    forward_noise = jnp.stack(forward_noise)
    data_elems = jnp.stack(data_elems)

    return_ = {}
    
     
    

    for i in range(realizations):
        
        if version == 0: # Reverse-sde solution

            prediction = eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, noise, log_time=log_time, update_step=update_step)[-1] # take final prediction

        elif version == 1: # Probability flow solution + noise
            
            prediction = eval_backward_rand_pc(forward_noise, correction_probability_flow_fn, times, rng, noise, log_time=log_time, update_step=update_step)[-1]
            
        elif version == 2: # Reverse-sde solution + PC
            
            prediction = eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, noise, corrector_steps = 100, log_time=log_time, update_step=update_step)[-1]
            
        elif version == 3: # Probability flow solution 
            
            prediction = eval_backward_rand_pc(forward_noise, correction_probability_flow_fn, times, rng, 0.0, log_time=log_time, update_step=update_step)[-1]
            
        elif version == 4: # Reverse-sde solution with increased noise
            
            prediction = eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, 2 * noise, log_time=log_time, update_step=update_step)[-1]
            
        elif version == 5: # Reverse-sde solution with decreased noise
            
            prediction = eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, 0.5 * noise, log_time=log_time, update_step=update_step)[-1]
            
        elif version == 7: # Reverse-sde solution with no noise
            
            prediction = eval_backward_rand_pc(forward_noise, correction_reverse_sde_fn, times, rng, 0, log_time=log_time, update_step=update_step)[-1]
            
        else:
            
            raise ValueError('Not implemented')
            
            
        rng, _ = jax.random.split(rng)

        return_[f'prediction_{i}'] = list(prediction)

    return_['ground_truth'] = list(data_elems)
    return_['input']= list(forward_noise)
    return_['forward']= list(forward_)

    return return_

def eval_backward_rand_pc(initial_value, correction_fn, times, rng, noise, corrector_steps=0, EPSILON=2e-7, log_time=False, update_step=1):

    if log_time:
        time_to_network_fn = lambda t : log_time_to_network(t, times[-1], times[0])
    else:
        time_to_network_fn = lambda t : t

    initial_shape = initial_value.shape
    
    y = initial_value
    
    physics_value_fn = forward_step_custom(initial_shape)
    
    y_list = [y]
    
    # Predicor steps
    for t1, t0 in tqdm(zip(times, times[1:])):
        
        delta_t = t1-t0
       
        if update_step == 1:
    
            P = physics_value_fn(y, delta_t)
            C = correction_fn(jnp.expand_dims(y, axis=1), jnp.tile(time_to_network_fn(t1), y.shape[0]))
            
            y = y - P - delta_t * C
        
        elif update_step == 2:
            
            P = physics_value_fn(y, delta_t) 
            
            in_ = jnp.stack([y, P/delta_t], axis=1)
            
            C = correction_fn(in_, jnp.tile(time_to_network_fn(t1), y.shape[0]))
        
            y = y - P - delta_t * C
        
        # Corrector step
        
        if t0 > times[-1]:
        
            for _ in range(corrector_steps):

                bm = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=y.shape) 
                rng, _ = jax.random.split(rng)
                
                if update_step == 1:
                    C = correction_fn(jnp.expand_dims(y, axis=1), jnp.tile(time_to_network_fn(t0), y.shape[0]))
                elif update_step == 2:
                    in_ = jnp.stack([y, P/delta_t], axis=1)
                    C = correction_fn(in_, jnp.tile(time_to_network_fn(t0), y.shape[0]))
                    
                y = y - EPSILON * C
                y = y + bm
                
            bm = noise * jax.random.normal(rng, shape=y.shape) * jnp.sqrt(delta_t)
            rng, _ = jax.random.split(rng)
            y = y + bm
            
        y_list.append(y)    
    
    return y_list