import phi.math
from phi.jax.flow import *

from physics import *
from tqdm import tqdm

import jax.numpy as jnp

def eval_backward_score_decoupled(state, obstacles, simulation_metadata, correction_fn, rng, noise,
                                  correction_coefficient=1.0, type=2, update=1, physics_forward_fn=None,
                                  physics_backward_fn=None, corrector_steps=0):
    
    DT = simulation_metadata['DT']
    STEPS = simulation_metadata['NSTEPS']
    
    def delta_physics_1(state, obstacles, t_sim_batch):
        
        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_forward, state_vel_x_forward, state_vel_y_forward = physics_forward_fn(state, obstacles, t_sim_batch) 
        
        return state_smoke_forward - state_smoke, state_vel_x_forward - state_vel_x, state_vel_y_forward - state_vel_y
        
    def delta_physics_2(state, obstacles, t_sim_batch):
        
        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_backward, state_vel_x_backward, state_vel_y_backward = physics_backward_fn(state, obstacles, t_sim_batch) 
        
        return state_smoke - state_smoke_backward, state_vel_x - state_vel_x_backward, state_vel_y - state_vel_y_backward
        
    if type==1:
        delta_physics = jax.jit(delta_physics_1)
    elif type==2:
        delta_physics = jax.jit(delta_physics_2)
    else:
        raise ValueError(f'Loss type {type} not supported!')
    
    t0 = 0
    t1 = DT * STEPS
    
    tprev = jnp.array(t1 + DT, dtype=jnp.float32) 
    tnext = jnp.array(t1, dtype=jnp.float32) 

    state_smoke, state_vel_x, state_vel_y, state_mask = state
    
    state_smoke = jnp.array(state_smoke, dtype=jnp.float32)
    state_vel_x = jnp.array(state_vel_x, dtype=jnp.float32)
    state_vel_y = jnp.array(state_vel_y, dtype=jnp.float32)
    state_mask = jnp.array(state_mask, dtype=jnp.float32)
    
    
    y_list = [state]
        
    for i in tqdm(range(int((t1-t0) / DT))):

        state = [state_smoke, state_vel_x, state_vel_y]

        state_smoke_delta, state_vel_x_delta, state_vel_y_delta = delta_physics(state, obstacles, tnext) 
        
        state_smoke = state_smoke - state_smoke_delta
        state_vel_x = state_vel_x - state_vel_x_delta 
        state_vel_y = state_vel_y - state_vel_y_delta 

        if update==1:
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_vel_x, state_vel_y, state_mask], jnp.array(tnext)))

        elif update==2:
           
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_smoke_delta, state_vel_x, state_vel_x_delta,
                               state_vel_y, state_vel_y_delta, state_mask], jnp.array(tnext, dtype=jnp.float32)))

        else:
            raise ValueError(f'Update {update} not supported.')

        state_smoke = state_smoke - correction_coefficient * DT * correction_smoke
        state_vel_x = state_vel_x - correction_coefficient * DT * correction_vel_x
        state_vel_y = state_vel_y - correction_coefficient * DT * correction_vel_y

       
        EPSILON = 1e-5
        for _ in range(corrector_steps):
            bm_smoke = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_smoke.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            bm_vel_x = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_vel_x.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            bm_vel_y = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_vel_y.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            
            if update==1:
                correction_smoke, correction_vel_x, correction_vel_y = correction_fn([state_smoke,
                                                                                      state_vel_x, state_vel_y,
                                                                                      state_mask], jnp.array(tnext))

            elif update==2:
                correction_smoke, correction_vel_x, correction_vel_y = correction_fn([state_smoke,
                                                                                      state_smoke_delta,
                                                                                      state_vel_x,
                                                                                      state_vel_x_delta,
                                                                                      state_vel_y,
                                                                                      state_vel_y_delta,
                                                                                      state_mask], jnp.array(tnext))

            state_smoke = state_smoke - EPSILON * correction_smoke + bm_smoke
            state_vel_x = state_vel_x - EPSILON * correction_vel_x + bm_vel_x
            state_vel_y = state_vel_y - EPSILON * correction_vel_y + bm_vel_y
        
        if noise > 0:

            state_smoke += noise * jax.random.normal(rng, shape=state_smoke.shape,
                                                     dtype=jnp.float32) * jnp.sqrt(DT)
            rng, _ = jax.random.split(rng)
            state_vel_x += noise * jax.random.normal(rng, shape=state_vel_x.shape,
                                                     dtype=jnp.float32) * jnp.sqrt(DT)
            rng, _ = jax.random.split(rng)
            state_vel_y += noise * jax.random.normal(rng, shape=state_vel_y.shape,
                                                     dtype=jnp.float32) * jnp.sqrt(DT)
            rng, _ = jax.random.split(rng)
            
            if update==1:
                correction_smoke, correction_vel_x, correction_vel_y = correction_fn([state_smoke,
                                                                                      state_vel_x,
                                                                                      state_vel_y,
                                                                                      state_mask],
                                                                                     jnp.array(tnext))

            elif update==2:

                correction_smoke, correction_vel_x, correction_vel_y = correction_fn([state_smoke,
                                                                                      state_smoke_delta,
                                                                                      state_vel_x,
                                                                                      state_vel_x_delta,
                                                                                      state_vel_y,
                                                                                      state_vel_y_delta,
                                                                                      state_mask],
                                                                                     jnp.array(tnext,
                                                                                               dtype=jnp.float32))

            else:
                raise ValueError(f'Update {update} not supported.')

            state_smoke = state_smoke - correction_coefficient * DT * correction_smoke
            state_vel_x = state_vel_x - correction_coefficient * DT * correction_vel_x
            state_vel_y = state_vel_y - correction_coefficient * DT * correction_vel_y
            

        tprev = tnext
        tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)

        state = state_smoke, state_vel_x, state_vel_y, state_mask

        y_list.append(state)  

    return y_list

def eval_backward_score(state, obstacles, simulation_metadata, correction_fn, rng, noise,
                        correction_coefficient=1.0, type=2, update=1, physics_forward_fn=None,
                        physics_backward_fn=None, corrector_steps=0):
    
    DT = simulation_metadata['DT']
    STEPS = simulation_metadata['NSTEPS']
    
    def delta_physics_1(state, obstacles, t_sim_batch):
        
        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_forward, state_vel_x_forward, state_vel_y_forward = physics_forward_fn(state, obstacles,
                                                                                           t_sim_batch)
        
        return (state_smoke_forward - state_smoke, state_vel_x_forward - state_vel_x,
                state_vel_y_forward - state_vel_y)
        
    def delta_physics_2(state, obstacles, t_sim_batch):
        
        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_backward, state_vel_x_backward, state_vel_y_backward = (
            physics_backward_fn(state, obstacles, t_sim_batch))
        
        return (state_smoke - state_smoke_backward, state_vel_x - state_vel_x_backward,
                state_vel_y - state_vel_y_backward)
        
    if type==1:
        delta_physics = jax.jit(delta_physics_1)
    elif type==2:
        delta_physics = jax.jit(delta_physics_2)
    else:
        raise ValueError(f'Loss type {type} not supported!')
    
    t0 = 0
    t1 = DT * STEPS
    
    tprev = jnp.array(t1 + DT, dtype=jnp.float32) 
    tnext = jnp.array(t1, dtype=jnp.float32) 

    state_smoke, state_vel_x, state_vel_y, state_mask = state
    
    state_smoke = jnp.array(state_smoke, dtype=jnp.float32)
    state_vel_x = jnp.array(state_vel_x, dtype=jnp.float32)
    state_vel_y = jnp.array(state_vel_y, dtype=jnp.float32)
    state_mask = jnp.array(state_mask, dtype=jnp.float32)
    
    
    y_list = [state]
        
    for i in tqdm(range(int((t1-t0) / DT))):

        state = [state_smoke, state_vel_x, state_vel_y]

        state_smoke_delta, state_vel_x_delta, state_vel_y_delta = delta_physics(state, obstacles, tnext) 

        if update==1:
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_vel_x, state_vel_y, state_mask], jnp.array(tnext)))

        elif update==2:
           
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_smoke_delta, state_vel_x, state_vel_x_delta,
                               state_vel_y, state_vel_y_delta, state_mask], jnp.array(tnext, dtype=jnp.float32)))

        else:
            raise ValueError(f'Update {update} not supported.')

        state_smoke = state_smoke - state_smoke_delta - correction_coefficient * DT * correction_smoke
        state_vel_x = state_vel_x - state_vel_x_delta - correction_coefficient * DT * correction_vel_x
        state_vel_y = state_vel_y - state_vel_y_delta - correction_coefficient * DT * correction_vel_y

       
        EPSILON = 1e-5
        for _ in range(corrector_steps):
            bm_smoke = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_smoke.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            bm_vel_x = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_vel_x.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            bm_vel_y = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_vel_y.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            
            if update==1:
                correction_smoke, correction_vel_x, correction_vel_y = (
                    correction_fn([state_smoke, state_vel_x, state_vel_y, state_mask], jnp.array(tnext)))

            elif update==2:
                correction_smoke, correction_vel_x, correction_vel_y = (
                    correction_fn([state_smoke, state_smoke_delta, state_vel_x, state_vel_x_delta,
                                   state_vel_y, state_vel_y_delta, state_mask], jnp.array(tnext)))

            state_smoke = state_smoke - EPSILON * correction_smoke + bm_smoke
            state_vel_x = state_vel_x - EPSILON * correction_vel_x + bm_vel_x
            state_vel_y = state_vel_y - EPSILON * correction_vel_y + bm_vel_y
        

        state_smoke += noise * jax.random.normal(rng, shape=state_smoke.shape, dtype=jnp.float32) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)
        state_vel_x += noise * jax.random.normal(rng, shape=state_vel_x.shape, dtype=jnp.float32) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)
        state_vel_y += noise * jax.random.normal(rng, shape=state_vel_y.shape, dtype=jnp.float32) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)

        tprev = tnext
        tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)

        state = state_smoke, state_vel_x, state_vel_y, state_mask

        y_list.append(state)  

    return y_list

def eval_backward_score_2(state, obstacles, simulation_metadata, correction_fn,
                          rng, noise, correction_coefficient=1.0, type=2, update=1,
                          physics_forward_fn=None, physics_backward_fn=None, corrector_steps=0):
    
    DT = simulation_metadata['DT']
    STEPS = simulation_metadata['NSTEPS']
    
    def delta_physics_1(state, obstacles, t_sim_batch):
        
        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_forward, state_vel_x_forward, state_vel_y_forward = (
            physics_forward_fn(state, obstacles, t_sim_batch))
        
        return (state_smoke_forward - state_smoke, state_vel_x_forward - state_vel_x,
                state_vel_y_forward - state_vel_y)
        
    def delta_physics_2(state, obstacles, t_sim_batch):
        
        state_smoke, state_vel_x, state_vel_y = state
        state_smoke_backward, state_vel_x_backward, state_vel_y_backward = (
            physics_backward_fn(state, obstacles, t_sim_batch))
        
        return (state_smoke - state_smoke_backward, state_vel_x - state_vel_x_backward,
                state_vel_y - state_vel_y_backward)
        
    if type==1:
        delta_physics = jax.jit(delta_physics_1)
    elif type==2:
        delta_physics = jax.jit(delta_physics_2)
    else:
        raise ValueError(f'Loss type {type} not supported!')
    
    t0 = 0
    t1 = DT * STEPS
    
    tprev = jnp.array(t1 + DT, dtype=jnp.float32) 
    tnext = jnp.array(t1, dtype=jnp.float32) 

    state_smoke, state_vel_x, state_vel_y, state_mask = state
    
    state_smoke = jnp.array(state_smoke, dtype=jnp.float32)
    state_vel_x = jnp.array(state_vel_x, dtype=jnp.float32)
    state_vel_y = jnp.array(state_vel_y, dtype=jnp.float32)
    state_mask = jnp.array(state_mask, dtype=jnp.float32)
    
    
    y_list = [state]
        
    for i in tqdm(range(int((t1-t0) / DT))):

        state = [state_smoke, state_vel_x, state_vel_y]

        state_smoke_delta, state_vel_x_delta, state_vel_y_delta = delta_physics(state, obstacles, tnext) 

        if update==1:
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_vel_x, state_vel_y, state_mask], jnp.array(tnext)))

        elif update==2:
           
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_smoke_delta, state_vel_x, state_vel_x_delta,
                               state_vel_y, state_vel_y_delta, state_mask],
                              jnp.array(tnext, dtype=jnp.float32)))

        else:
            raise ValueError(f'Update {update} not supported.')

        state_smoke = state_smoke - state_smoke_delta - correction_coefficient * DT * correction_smoke
        state_vel_x = state_vel_x - state_vel_x_delta - correction_coefficient * DT * correction_vel_x
        state_vel_y = state_vel_y - state_vel_y_delta - correction_coefficient * DT * correction_vel_y

       
        EPSILON = 1e-5
        for _ in range(corrector_steps):
            bm_smoke = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_smoke.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            bm_vel_x = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_vel_x.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            bm_vel_y = jnp.sqrt(2 * EPSILON) * jax.random.normal(rng, shape=state_vel_y.shape, dtype=jnp.float32) 
            rng, _ = jax.random.split(rng)
            
            if update==1:
                correction_smoke, correction_vel_x, correction_vel_y = (
                    correction_fn([state_smoke, state_vel_x, state_vel_y, state_mask], jnp.array(tnext)))

            elif update==2:
                correction_smoke, correction_vel_x, correction_vel_y = (
                    correction_fn([state_smoke, state_smoke_delta, state_vel_x, state_vel_x_delta,
                                   state_vel_y, state_vel_y_delta, state_mask], jnp.array(tnext)))

            state_smoke = state_smoke - EPSILON * correction_smoke + bm_smoke
            state_vel_x = state_vel_x - EPSILON * correction_vel_x + bm_vel_x
            state_vel_y = state_vel_y - EPSILON * correction_vel_y + bm_vel_y
        

        state_smoke += noise * jax.random.normal(rng, shape=state_smoke.shape, dtype=jnp.float32) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)
        state_vel_x += noise * jax.random.normal(rng, shape=state_vel_x.shape, dtype=jnp.float32) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)
        state_vel_y += noise * jax.random.normal(rng, shape=state_vel_y.shape, dtype=jnp.float32) * jnp.sqrt(DT)
        rng, _ = jax.random.split(rng)
        
        if update==1:
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_vel_x, state_vel_y, state_mask], jnp.array(tnext)))

        elif update==2:
           
            correction_smoke, correction_vel_x, correction_vel_y = (
                correction_fn([state_smoke, state_smoke_delta, state_vel_x,
                               state_vel_x_delta, state_vel_y, state_vel_y_delta, state_mask],
                              jnp.array(tnext, dtype=jnp.float32)))
            
        state_smoke = state_smoke - correction_coefficient * DT * correction_smoke
        state_vel_x = state_vel_x - correction_coefficient * DT * correction_vel_x
        state_vel_y = state_vel_y - correction_coefficient * DT * correction_vel_y

        tprev = tnext
        tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)

        state = state_smoke, state_vel_x, state_vel_y, state_mask

        y_list.append(state)  

    return y_list

def eval_backward_rand(state, obstacles, simulation_metadata, correction_fn, score_fn,
                       rng, score_drift, score_noise, noise, correction_coefficient=1.0,
                       type='2', physics_forward_fn=None, physics_backward_fn=None):
    
    DT = simulation_metadata['DT']
    STEPS = simulation_metadata['NSTEPS']
    
    t0 = 0
    t1 = DT * STEPS
    
    tprev = jnp.array(t1 + DT) 
    tnext = jnp.array(t1) 

    smoke_state, vel_x_state, vel_y_state, mask_state = state
    
    y_list = [state]
    
    if type=='1':
        
        if physics_forward_fn:
            physics_value_fn = physics_forward_fn
        else:
            physics_value_fn = physics_forward(simulation_metadata)

        for i in tqdm(range(int((t1-t0) / DT))):

            smoke_bm = noise * jax.random.normal(rng, shape=smoke_state.shape) * DT
            rng, _ = jax.random.split(rng)
            vel_x_bm = noise * jax.random.normal(rng, shape=vel_x_state.shape) * DT
            rng, _ = jax.random.split(rng)
            vel_y_bm = noise * jax.random.normal(rng, shape=vel_y_state.shape) * DT
            rng, _ = jax.random.split(rng)

            score_smoke, score_vel_x, score_vel_y = score_fn(state, tnext)

            smoke_bm = (smoke_bm + correction_coefficient * score_noise * score_smoke *
                        jax.random.normal(rng, shape=smoke_bm.shape) * DT)
            rng, _ = jax.random.split(rng)
            vel_x_bm = (vel_x_bm + correction_coefficient * score_noise * score_vel_x *
                        jax.random.normal(rng, shape=vel_x_bm.shape) * DT)
            rng, _ = jax.random.split(rng)
            vel_y_bm = (vel_y_bm + correction_coefficient * score_noise * score_vel_y *
                        jax.random.normal(rng, shape=vel_y_bm.shape) * DT)
            rng, _ = jax.random.split(rng)

            smoke_forward, vel_x_forward, vel_y_forward = physics_value_fn([smoke_state, vel_x_state,
                                                                            vel_y_state], obstacles, tnext)

            smoke_state = 2 * smoke_state - smoke_forward 
            vel_x_state = 2 * vel_x_state - vel_x_forward 
            vel_y_state = 2 * vel_y_state - vel_y_forward 

            state = smoke_state, vel_x_state, vel_y_state, mask_state

            score_smoke, score_vel_x, score_vel_y = score_fn(state, tnext)
            correction_smoke, correction_vel_x, correction_vel_y = correction_fn(state, tnext)

            smoke_state = (smoke_state - correction_coefficient *
                           (DT * score_drift * score_smoke + correction_smoke))
            vel_x_state = (vel_x_state - correction_coefficient *
                           (DT * score_drift * score_vel_x + correction_vel_x))
            vel_y_state = (vel_y_state - correction_coefficient *
                           (DT * score_drift * score_vel_y + correction_vel_y))

            smoke_state = smoke_state + smoke_bm
            vel_x_state = vel_x_state + vel_x_bm
            vel_y_state = vel_y_state + vel_y_bm

            tprev = tnext
            tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)

            state = smoke_state, vel_x_state, vel_y_state, mask_state

            y_list.append(state)  
            
        return y_list
        
    elif type=='2':

        if physics_backward_fn:
            physics_backward_fn = physics_backward_fn
        else:
            physics_backward_fn = physics_backwards(simulation_metadata)
        
        for i in tqdm(range(int((t1-t0) / DT))):

            smoke_bm = noise * jax.random.normal(rng, shape=smoke_state.shape) * DT
            rng, _ = jax.random.split(rng)
            vel_x_bm = noise * jax.random.normal(rng, shape=vel_x_state.shape) * DT
            rng, _ = jax.random.split(rng)
            vel_y_bm = noise * jax.random.normal(rng, shape=vel_y_state.shape) * DT
            rng, _ = jax.random.split(rng)

            score_smoke, score_vel_x, score_vel_y = score_fn(state, tnext)

            smoke_bm = (smoke_bm + correction_coefficient *
                        score_noise * score_smoke * jax.random.normal(rng, shape=smoke_bm.shape) * DT)
            rng, _ = jax.random.split(rng)
            vel_x_bm = (vel_x_bm + correction_coefficient *
                        score_noise * score_vel_x * jax.random.normal(rng, shape=vel_x_bm.shape) * DT)
            rng, _ = jax.random.split(rng)
            vel_y_bm = (vel_y_bm + correction_coefficient *
                        score_noise * score_vel_y * jax.random.normal(rng, shape=vel_y_bm.shape) * DT)
            rng, _ = jax.random.split(rng)

       
            
            smoke_state, vel_x_state, vel_y_state = physics_backward_fn([smoke_state, vel_x_state, vel_y_state],
                                                                        obstacles, tnext)

            # smoke_forward, vel_x_forward, vel_y_forward = physics_value_fn([smoke_state, vel_x_state, vel_y_state], obstacles, tnext) 

            # smoke_state = 2 * smoke_state - smoke_forward 
            # vel_x_state = 2 * vel_x_state - vel_x_forward 
            # vel_y_state = 2 * vel_y_state - vel_y_forward 

            state = smoke_state, vel_x_state, vel_y_state, mask_state

            score_smoke, score_vel_x, score_vel_y = score_fn(state, tnext)
            correction_smoke, correction_vel_x, correction_vel_y = correction_fn(state, tnext)

            smoke_state = smoke_state - correction_coefficient * (DT * score_drift * score_smoke + correction_smoke)
            vel_x_state = vel_x_state - correction_coefficient * (DT * score_drift * score_vel_x + correction_vel_x)
            vel_y_state = vel_y_state - correction_coefficient * (DT * score_drift * score_vel_y + correction_vel_y)

            smoke_state = smoke_state + smoke_bm
            vel_x_state = vel_x_state + vel_x_bm
            vel_y_state = vel_y_state + vel_y_bm

            tprev = tnext
            tnext = jnp.maximum(jnp.array(tprev - DT), 0.0)

            state = smoke_state, vel_x_state, vel_y_state, mask_state

            y_list.append(state)  

        return y_list
        
    else:
        raise ValueError(f'Type {type} not supported.')
        
def eval_backward_conv(params, state, simulation_metadata, network_fn):
    
    smoke_state, vel_x_state, vel_y_state, mask_state = state
    DT = simulation_metadata['DT']
    STEPS = simulation_metadata['NSTEPS']
    t0 = 0
    t1 = DT * STEPS
    
    rng = jax.random.PRNGKey(0)
    
    backward_ = [state]
    
    tprev = jnp.array(t1 + DT) 
    tnext = jnp.array(t1) 
    
    for i in tqdm(range(int((t1-t0) / DT)-1)):
        
        net_fn_smoke, net_fn_vel_x, net_fn_vel_y = (
            network_fn(params, rng, [smoke_state, vel_x_state, vel_y_state, mask_state], jnp.array(tnext)))
            
        smoke_state = smoke_state - DT * net_fn_smoke
        vel_x_state = vel_x_state - DT * net_fn_vel_x
        vel_y_state = vel_y_state - DT * net_fn_vel_y
            
        tprev = tnext
        tnext = tnext - DT
        
        backward_.append([smoke_state, vel_x_state, vel_y_state, mask_state])
   
    return backward_
    
def eval_forward(state, obstacles, simulation_metadata, physics_forward_fn = None, t0=0):
    
    DT = simulation_metadata['DT']
    STEPS = simulation_metadata['NSTEPS']
    
    t1 = DT * STEPS
    
    # tprev = jnp.array(t0)

    tprev = phi.math.tensor(t0)
    tnext = phi.math.tensor(t0+DT)

    # tnext = jnp.array(t0+DT)

    smoke_state, vel_x_state, vel_y_state, mask_state = state
    
    _forward = [state]
    
    if physics_forward_fn:
        physics_value_fn = physics_forward_fn
    else:
        physics_value_fn = physics_forward(simulation_metadata)
    
    for i in range(int(STEPS)):
  
        smoke_state, vel_x_state, vel_y_state, mask_state = state

        smoke_forward, vel_x_forward, vel_y_forward = (
            physics_value_fn([smoke_state, vel_x_state, vel_y_state], obstacles, tnext))

        tnext += DT

        state = [smoke_forward, vel_x_forward, vel_y_forward, mask_state]
        
        _forward.append(state)
        
    return _forward
    