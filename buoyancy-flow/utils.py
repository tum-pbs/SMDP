import jax

import optax
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import haiku as hk
import jax.numpy as jnp
import phi.math
import pickle
import os

def get_correction_term_reverse_sde_score(forward_fn, noise):
    
    rng = jax.random.PRNGKey(0)

    def correction_term_reverse_sde(params, state, t_sim_batch):

        correction = forward_fn(params, rng, state, t_sim_batch) 

        r_ = []
        for e in correction: 
        
            correction_ = - (noise ** 2) * e 
            r_.append(correction_)

        return r_
    
    return correction_term_reverse_sde

def get_correction_term_probability_flow_score(forward_fn, noise):
    
    correction_term = get_correction_term_reverse_sde_score(forward_fn, noise)
    f = lambda params, state, t: [0.5 * e for e in correction_term(params, state, t)]
    
    return f

def get_correction_term_reverse_sde(forward_fn, noise, score_noise, DT):
    
    rng = jax.random.PRNGKey(0)

    def correction_term_reverse_sde(params, state, t_sim_batch):

        correction = forward_fn(params, rng, state, t_sim_batch) 

        r_ = []
        for e in correction: 
        
            correction_ = - 2 * score_noise * noise 
            correction_ = correction_ - 2 * (score_noise + 0.5 * noise) * e
            correction_ = correction_ - 2 * score_noise * noise * jnp.power(e, 2)
            correction_ = correction_ - (score_noise ** 2) * jnp.power(e, 3)
            correction_ = correction_ * DT
            r_.append(correction_)

        return r_
    
    return correction_term_reverse_sde

def get_correction_term_probability_flow(forward_fn, noise, score_noise, DT):
    
    correction_term = get_correction_term_reverse_sde(forward_fn, noise, score_noise, DT)
    f = lambda params, state, t: [0.5 * e for e in correction_term(params, state, t)]
    
    return f

def get_score_fn(forward_fn):
    
    rng = jax.random.PRNGKey(0)
    
    def score_fn(params, state, t_sim_batch):
        
        return [- e for e in forward_fn(params, rng, state, t_sim_batch)]
    
    return score_fn

def count_weights(params):
    params_tree = jax.tree_util.tree_map(jnp.size, params)
    leaves = jax.tree_leaves(params_tree)
    return sum(leaves)

def has_nan_weights(params):
    f = lambda x: jnp.any(jnp.isnan(x))
    params_tree = jax.tree_util.tree_map(f, params)
    leaves = jax.tree_leaves(params_tree)
    return any(leaves)
    
    
def create_default_update_fn(optimizer: optax.GradientTransformation,
                             model_loss: Callable):
    """
    This function calls the update function, to implement the backpropagation
    """

    def update(params, opt_state, batch, rng) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
     
        print('Tracing update...')
        # print(f'Tracing update... with signature ROLLOUT: {batch[2][0].shape[0]}
        # SPHERES: {len(batch[3][0])} BOXES: {len(batch[3][1])}')
    
        out_, grads = jax.value_and_grad(model_loss, has_aux=True)(params, *batch)
        batch_loss, end_state = out_
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, batch_loss, end_state
    
    return update
            

def save_model_weights(model, savename):

    # if directory of savename does not exist, create it
    if not os.path.exists(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))

    with open(savename, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_model_weights(savename):
        
    with open(savename, 'rb') as file:
        params = pickle.load(file)

    return params