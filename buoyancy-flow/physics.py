import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import phi.math.backend

from phi.jax.flow import *

def batch_inflow(inflow, batchSize=1):
    
    init = [(inflow['_center'][1], inflow['_center'][0])] * batchSize
    center = tensor(init, batch('batch'), channel(vector='x,y'))
    geometry_ = Sphere(center=center, radius=inflow['_radius'])
    
    return geometry_


def batch_geometries_pre(geometries):
    batchSize = len(geometries)

    sphere_batch = {}
    box_batch = {}

    spheres = []
    boxes = []

    for n, geom in zip(range(batchSize), geometries):

        sphere_batch[n] = []
        box_batch[n] = []

        for o in geom:

            if '_center' in o:

                sphere_batch[n].append(o)

            elif '_lower' in o:

                box_batch[n].append(o)

            else:
                raise ValueError(f"Unkown obstacle {o}")

    for sphere_i in range(3):

        init_radius = [0] * batchSize
        init_center = [(-20, -20)] * batchSize

        active = False

        for n in range(batchSize):

            if len(sphere_batch[n]) > sphere_i:
                active = True
                o = sphere_batch[n][sphere_i]
                init_radius[n] = o['_radius']
                init_center[n] = (o['_center'][1], o['_center'][0])

        if active:
            spheres.append((jnp.array(init_center), jnp.array(init_radius)))

    for box_i in range(3):

        init_lower = [(-20, -20)] * batchSize
        init_upper = [(-20, -20)] * batchSize

        active = False

        for n in range(batchSize):

            if len(box_batch[n]) > box_i:
                active = True
                o = box_batch[n][box_i]
                init_lower[n] = (o['_lower'][1], o['_lower'][0])
                init_upper[n] = (o['_upper'][1], o['_upper'][0])

        if active:
            boxes.append((jnp.array(init_lower), jnp.array(init_upper)))

    return spheres, boxes

def batch_geometries_pre_phiflow(geometries):
    
    batchSize = len(geometries)
    
    sphere_batch = {}
    box_batch = {}
    
    spheres = []
    boxes = []
    
    for n, geom in zip(range(batchSize), geometries):
        
        sphere_batch[n] = []
        box_batch[n] = []
        
        for o in geom:
            
            if '_center' in o:
                
                sphere_batch[n].append(o)
            
            elif '_lower' in o:
                
                box_batch[n].append(o)
             
            else:
                raise ValueError(f"Unkown obstacle {o}")
                
    for sphere_i in range(3):
        
        init_radius = [0] * batchSize
        init_center = [(-20, -20)] * batchSize
        
        active = False
        
        for n in range(batchSize):
            
            if len(sphere_batch[n]) > sphere_i:
                active = True
                o = sphere_batch[n][sphere_i]
                init_radius[n] = o['_radius']
                init_center[n] = (o['_center'][1], o['_center'][0])
                
        if active:
    
            spheres.append((math.tensor(np.asarray(init_center), instance('spheres'), channel(vector='x,y')),
                            math.tensor(init_radius)))
                    
    for box_i in range(3):
        
        init_lower = [(-20, -20)] * batchSize
        init_upper = [(-20, -20)] * batchSize
        
        active = False
        
        for n in range(batchSize):
            
            if len(box_batch[n]) > box_i:
                active = True
                o = box_batch[n][box_i]
                init_lower[n] = (o['_lower'][1], o['_lower'][0])
                init_upper[n] = (o['_upper'][1], o['_upper'][0])
                
        if active:
            
            boxes.append((math.tensor(init_lower, instance('boxes'), channel(vector='x,y')),
                          math.tensor(init_upper, instance('boxes'), channel(vector='x,y'))))
        
    return spheres, boxes
    
def batch_geometries_post(geometries):
    
    batched_geometries = []
    
    for elem in geometries[0]:
        
        center = math.tensor(elem[0], batch('batch'), channel(vector='x,y'))
        radius = math.tensor(elem[1], batch('batch'))
        geometry_ = Sphere(center=center, radius=radius)
        batched_geometries.append(Obstacle(geometry_))
        
    for elem in geometries[1]:
            lower = math.tensor(elem[0], batch('batch'), channel(vector='x,y'))
            upper = math.tensor(elem[1], batch('batch'), channel(vector='x,y'))
            geometry_ = Box(lower=lower, upper=upper)
            batched_geometries.append(Obstacle(geometry_))
       
    return batched_geometries

def batch_geometries(geometries):
    
    batchSize = len(geometries)
    
    batched_geometries = []
    
    sphere_batch = {}
    box_batch = {}
    
    for n, geom in zip(range(batchSize), geometries):
        
        sphere_batch[n] = []
        box_batch[n] = []
        
        for o in geom:
            
            if '_center' in o:
                
                sphere_batch[n].append(o)
            
            elif '_lower' in o:
                
                box_batch[n].append(o)
             
            else:
                raise ValueError(f"Unkown obstacle {o}")
                
    for sphere_i in range(3):
        
        init_radius = [0] * batchSize
        init_center = [(-20, -20)] * batchSize
        
        active = False
        
        for n in range(batchSize):
            
            if len(sphere_batch[n]) > sphere_i:
                active = True
                o = sphere_batch[n][sphere_i]
                init_radius[n] = o['_radius']
                init_center[n] = (o['_center'][1], o['_center'][0])
                
        if active:
            center = math.tensor(init_center, batch('batch'), channel(vector='x,y'))
            radius = math.tensor(init_radius, batch('batch'))
            geometry_ = Sphere(center=center, radius=radius)
            batched_geometries.append(Obstacle(geometry_))
            
    for box_i in range(3):
        
        init_lower = [(-20, -20)] * batchSize
        init_upper = [(-20, -20)] * batchSize
        
        active = False
        
        for n in range(batchSize):
            
            if len(box_batch[n]) > box_i:
                active = True
                o = box_batch[n][box_i]
                init_lower[n] = (o['_lower'][1], o['_lower'][0])
                init_upper[n] = (o['_upper'][1], o['_upper'][0])
                
        if active:
            lower = math.tensor(init_lower, batch('batch'), channel(vector='x,y'))
            upper = math.tensor(init_upper, batch('batch'), channel(vector='x,y'))
            geometry_ = Box(lower=lower, upper=upper)
            batched_geometries.append(Obstacle(geometry_))
            
    return batched_geometries

def physics_backwards(simulation_metadata):
    
    INFLOW = simulation_metadata['INFLOW']
    inflow_geometry = SoftGeometryMask(INFLOW)
    smoke_res = simulation_metadata['smoke_res']
    v_res = simulation_metadata['v_res']
    BOUNDS = simulation_metadata['BOUNDS']
    
    DT = simulation_metadata['DT']
    orig_DT = 0.01
    time_step = DT / orig_DT 
    
    # we have to do this, because otherwise we get a tracer in inflow_geometry (seems like bug in phiflow 2.2)
    _ = inflow_geometry @ CenteredGrid(0, extrapolation.BOUNDARY, x=smoke_res ** 2,
                                            y=smoke_res ** 2, bounds=BOUNDS)
    
    pressure_solver = 'auto'
    
    def physics_(state, obstacle_list, timestep):
        
            print('tracing physics backwards...')
        
            obstacle_list = batch_geometries_post(obstacle_list)
        
            smoke_state, velocity_x_state, velocity_y_state = state
        
            # Init smoke and velocity with data

            vel_x_phi = math.tensor(velocity_x_state, batch('batch'), spatial('x,y'))
            vel_y_phi = math.tensor(velocity_y_state, batch('batch'), spatial('x,y'))
            
            stacked_vel = math.stack((vel_y_phi, vel_x_phi), dim=math.channel('vector'))

            velocity = StaggeredGrid(stacked_vel, extrapolation.ZERO, x=v_res ** 2, y=v_res ** 2, bounds=BOUNDS)

            smoke = CenteredGrid(math.tensor(smoke_state, batch('batch'), spatial('x,y')), extrapolation.BOUNDARY, x=smoke_res ** 2, y=smoke_res ** 2, bounds=BOUNDS)

            # hard coded in data generation
            inflow_active = jnp.int32(timestep < 0.2)
            
            inflow = SoftGeometryMask(INFLOW) @ CenteredGrid(0, smoke.extrapolation, x=smoke_res ** 2,
                                                              y=smoke_res ** 2, bounds=BOUNDS)
            
            velocity = velocity @ StaggeredGrid(0, velocity.extrapolation, x=v_res ** 2, y=v_res ** 2,
                                                     bounds=BOUNDS)
            
            
            smoke = (1 - inflow_active) * smoke + inflow_active * smoke @ inflow
            
            smoke = advect.mac_cormack(smoke, velocity, -time_step) - time_step * inflow_active * inflow

            buoyancy_force = smoke * (0.1, 0) @ velocity  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, -time_step) - time_step * buoyancy_force
            
            try:
                with math.SolveTape() as solves:
                    velocity, pressure = fluid.make_incompressible(velocity, obstacle_list,
                                                                   Solve(pressure_solver, 1e-5, 0))
               
                # print(f"Pressure solve {v_res ** 2}x{v_res ** 2} with {solves[0].method}: {solves[0].solve_time * 1000:.0f} ms ({solves[0].iterations} iterations)")

            except ConvergenceException as err:
            
                print(f"Pressure solve {v_res ** 2}x{v_res ** 2} with {err.result.method}: {err}\nMax residual: {math.max(abs(err.result.residual.values))}")


                save_sample = False

                velocity -= field.spatial_gradient(err.result.x, velocity.extrapolation, type=type(velocity))

            velocity_x_state = jnp.asarray(velocity.vector[1].values.native(order='batch,x,y'), jnp.float64)
            velocity_y_state = jnp.asarray(velocity.vector[0].values.native(order='batch,x,y'), jnp.float64)
            
            smoke_state = jnp.asarray(smoke.values.native(order='batch,x,y'), jnp.float64)
            
            return smoke_state, velocity_x_state, velocity_y_state
    
    print('jit compile physics')
    return math.jit_compile(physics_)

def physics_forward(simulation_metadata):
    
    INFLOW = simulation_metadata['INFLOW']
    inflow_geometry = SoftGeometryMask(INFLOW)
    smoke_res = simulation_metadata['smoke_res']
    v_res = simulation_metadata['v_res']
    BOUNDS = simulation_metadata['BOUNDS']
    
    # we have to do this, because otherwise we get a tracer in inflow_geometry (seems like bug in phiflow 2.2)
    _ = inflow_geometry @ CenteredGrid(0, extrapolation.BOUNDARY, x=smoke_res ** 2,
                                            y=smoke_res ** 2, bounds=BOUNDS)
    
    # obstacle_list = simulation_metadata['obstacle_list']
    
    DT = simulation_metadata['DT']
    orig_DT = 0.01
    time_step = DT / orig_DT 
    
    pressure_solver = 'auto'
    
    def physics_forward(state, obstacle_list, timestep):
        
            print('tracing physics forwards...')

            obstacle_list = batch_geometries_post(obstacle_list)
        
            smoke_state, velocity_x_state, velocity_y_state = state
        
            # Init smoke and velocity with data

            vel_x_phi = math.tensor(velocity_x_state, batch('batch'), spatial('x,y'))
            vel_y_phi = math.tensor(velocity_y_state, batch('batch'), spatial('x,y'))
            
            stacked_vel = math.stack((vel_y_phi, vel_x_phi), dim=math.channel('vector'))
 
            velocity = StaggeredGrid(stacked_vel, extrapolation.ZERO, x=v_res ** 2, y=v_res ** 2, bounds=BOUNDS)

            smoke = CenteredGrid(math.tensor(smoke_state, batch('batch'), spatial('x,y')), extrapolation.BOUNDARY, x=smoke_res ** 2, y=smoke_res ** 2, bounds=BOUNDS)

            # hard coded in data generation
            # inflow_active = jnp.int32(timestep < 0.2)

            inflow_active = phi.math.cast(phi.math.tensor(timestep < 0.2), phi.math.DType(int, 32))
            
            inflow = inflow_geometry @ CenteredGrid(0, smoke.extrapolation, x=smoke_res ** 2,
                                                              y=smoke_res ** 2, bounds=BOUNDS)
            
            velocity = velocity @ StaggeredGrid(0, velocity.extrapolation, x=v_res ** 2, y=v_res ** 2,
                                                     bounds=BOUNDS)
            
            
            smoke = (1 - inflow_active) * smoke + inflow_active * smoke @ inflow
            
            smoke = advect.mac_cormack(smoke, velocity, time_step) + time_step * inflow_active * inflow

            buoyancy_force = smoke * (0.1, 0) @ velocity  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, time_step) + time_step * buoyancy_force
            try:
                with math.SolveTape() as solves:
                    velocity, pressure = fluid.make_incompressible(velocity, obstacle_list,
                                                                   Solve(pressure_solver, 1e-5, 0))
               
                # print(f"Pressure solve {v_res ** 2}x{v_res ** 2} with {solves[0].method}: {solves[0].solve_time * 1000:.0f} ms ({solves[0].iterations} iterations)")

            except ConvergenceException as err:
            
                print(f"Pressure solve {v_res ** 2}x{v_res ** 2} with {err.result.method}: {err}\nMax residual: {math.max(abs(err.result.residual.values))}")


                save_sample = False

                velocity -= field.spatial_gradient(err.result.x, velocity.extrapolation, type=type(velocity))

            # velocity_x_state = jnp.asarray(velocity.vector['x'].values.native(order='x,y'), jnp.float64)
            # velocity_y_state = jnp.asarray(velocity.vector['y'].values.native(order='x,y'), jnp.float64)

            # velocity_x_state = jnp.asarray(velocity.vector[1].values.native(order='batch,x,y'), jnp.float64)
            # velocity_y_state = jnp.asarray(velocity.vector[0].values.native(order='batch,x,y'), jnp.float64)
            
            # smoke_state = jnp.asarray(smoke.values.native(order='batch,x,y'), jnp.float64)

            velocity_x_state = phi.math.backend.default_backend().as_tensor(
                velocity.vector[1].values.native(order='batch,x,y'))
            velocity_y_state = phi.math.backend.default_backend().as_tensor(
                velocity.vector[0].values.native(order='batch,x,y'))
            smoke_state = phi.math.backend.default_backend().as_tensor(smoke.values.native(order='batch,x,y'))

            return smoke_state, velocity_x_state, velocity_y_state
    
    print('jit compile physics')
    # return physics_forward 
    # return jax.jit(physics_forward)
    return math.jit_compile(physics_forward)