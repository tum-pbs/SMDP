""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
The grid resolution of the smoke density and velocity field can be changed during the simulation.
"""

from phi.torch.flow import *
import torch


import random
from random import randint, seed, choice

import os
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import *
from tqdm import tqdm
import h5py
import pickle

STATS_FILE = 'smoke_plumes_stats.p'
H5PY_PATH = "smoke_plumes_r%1d.h5"

def create_obstacle_list(bounds : Box):
    rng = random.Random()
    width = bounds.size[1]
    height = bounds.size[0]
    size = int((width + height) * 0.06)

    obstacle_list = []

    for _ in range(randint(1, 2)):

        # Do not place obstacles below 30

        x1 = rng.randint(1, width)
        y1 = rng.randint(50, height)
        
        orientation = rng.choice([0, 1, 2, 3])
        length = rng.randint(10, height)
        thickness = rng.randint(5, size)
        if orientation == 0:
            obstacle = Box[x1:min(x1+length, height), y1:min(y1+thickness, width)] # right
        elif orientation == 1:
            obstacle = Box[x1:min(x1+thickness, height), y1:min(y1+length, width)] # top
        elif orientation == 2:
            obstacle = Box[max(x1-length, 0):x1, max(y1-thickness, 20):y1]
        else:
            obstacle = Box[max(x1-thickness, 0):x1, max(y1-length, 20):y1]

        print(obstacle)
        obstacle_list.append(Obstacle(obstacle))

        # Draw random circles
    for _ in range(randint(1, 2)):
        x1, y1 = rng.randint(1, width), rng.randint(50, height)
        radius = rng.randint(5, size)
        obstacle = Sphere(center=(x1, y1), radius=radius)
        print(obstacle)
        obstacle_list.append(Obstacle(obstacle))

    return obstacle_list

def save_snapshots(rank, num_samples, timesteps):
    # VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    # device = torch.device(f'cuda:{VISIBLE_DEVICES.split(",")[rank]}')
    # torch.cuda.set_device(int(VISIBLE_DEVICES.split(",")[rank]))

    TORCH.set_default_device(TORCH.list_devices('GPU')[rank])
    print(f'rank {rank} running on ', (TORCH.list_devices('GPU')[rank]))

    with h5py.File(H5PY_PATH % rank, 'a') as hf:
        Scene
        for n in tqdm(range(num_samples)):

            while True:
            
                data_masks = []
                data_smoke = []
                data_velocity_x = []
                data_velocity_y = []

                smoke_res = vis.control(8, (3, 20))
                v_res = vis.control(8, (3, 20))
                # pressure_solver = vis.control('auto', ('auto', 'CG', 'CG-adaptive', 'CG-native', 'direct', 'GMres', 'lGMres', 'biCG', 'CGS', 'QMR', 'GCrotMK'))
                pressure_solver = vis.control('auto', ('auto', 'CG'))

                BOUNDS = Box[0:100, 0:100]
                INFLOW = Sphere(center=(50, 10), radius=5)
                velocity = StaggeredGrid((0, 0), extrapolation.ZERO, x=v_res ** 2, y=v_res ** 2, bounds=BOUNDS)
                smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=smoke_res ** 2, y=smoke_res ** 2, bounds=BOUNDS)

                obstacle_list = create_obstacle_list(BOUNDS)
                # obstacle = Obstacle(Box[47:53, 20:70]) # angular_velocity=0.05
                obstacle_mask = CenteredGrid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacle_list])),
                                             extrapolation.ZERO, x=smoke_res ** 2, y=smoke_res ** 2,
                                             bounds=BOUNDS)  # to show in user interface

                save_sample = True
                
                # viewer = view(smoke, obstacle_mask, velocity, display=('smoke', 'obstacle_mask'), play=False)
                for timestep in tqdm(range(max(timesteps))):
                    # Resize grids if needed
                    
                    if timestep < 20:
                    
                        inflow = SoftGeometryMask(INFLOW) @ CenteredGrid(0, smoke.extrapolation, x=smoke_res ** 2,
                                                                          y=smoke_res ** 2, bounds=BOUNDS)
                        smoke = smoke @ inflow
                        velocity = velocity @ StaggeredGrid(0, velocity.extrapolation, x=v_res ** 2, y=v_res ** 2,
                                                             bounds=BOUNDS)
                        # Physics step
                        smoke = advect.mac_cormack(smoke, velocity, 1) + inflow
                        
                    else:
                        
                        velocity = velocity @ StaggeredGrid(0, velocity.extrapolation, x=v_res ** 2, y=v_res ** 2,
                                                             bounds=BOUNDS)
                        # Physics step
                        smoke = advect.mac_cormack(smoke, velocity, 1) 
                        
                        
                    
                    buoyancy_force = smoke * (0, 0.1) @ velocity  # resamples smoke to velocity sample points
                    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
                    try:
                        with math.SolveTape() as solves:
                            velocity, pressure = fluid.make_incompressible(velocity, obstacle_list,
                                                                           Solve(pressure_solver, 1e-5, 0))
                        # viewer.log_scalars(solve_time=solves[0].solve_time)
                        # viewer.info(f"Presure solve {v_res ** 2}x{v_res ** 2} with {solves[0].method}: {solves[0].solve_time * 1000:.0f} ms ({solves[0].iterations} iterations)")

                        print(f"Pressure solve {v_res ** 2}x{v_res ** 2} with {solves[0].method}: {solves[0].solve_time * 1000:.0f} ms ({solves[0].iterations} iterations)")

                    except ConvergenceException as err:
                        # viewer.info(f"Presure solve {v_res ** 2}x{v_res ** 2} with {err.result.method}: {err}\nMax residual: {math.max(abs(err.result.residual.values))}")

                        print(f"Pressure solve {v_res ** 2}x{v_res ** 2} with {err.result.method}: {err}\nMax residual: {math.max(abs(err.result.residual.values))}")
                        
                        print(f"Restarting sample {n} on rank {rank}")
                        
                        save_sample = False
                        
                        break
                        
                        velocity -= field.spatial_gradient(err.result.x, velocity.extrapolation, type=type(velocity))

                    if timestep in timesteps:
                        data_masks.append(math.numpy(obstacle_mask.values.numpy(order=('y', 'x'))))
                        data_velocity_x.append(math.numpy(velocity.vector[0].values.numpy(order=('y', 'x'))))
                        data_velocity_y.append(math.numpy(velocity.vector[1].values.numpy(order=('y', 'x'))))
                        data_smoke.append(math.numpy(smoke.values.numpy(order=('y', 'x'))))
                        
                if save_sample:

                    data_masks = np.stack(data_masks)
                    data_velocity_x = np.stack(data_velocity_x)
                    data_velocity_y = np.stack(data_velocity_y)
                    data_smoke = np.stack(data_smoke)

                    grp = hf.create_group(str(n))

                    grp.create_dataset(
                        name='smoke_res', data=smoke_res)
                    
                    grp.create_dataset(
                        name='v_res', data=v_res)
                    
                    bounds_dict = BOUNDS.__dict__
                    bounds_data = [bounds_dict['_lower'][0], bounds_dict['_lower'][1], bounds_dict['_upper'][0], bounds_dict['_upper'][1]]
                   
                    grp.create_dataset(name='BOUNDS', data=np.asarray(bounds_data, dtype=np.int32))
                    
                    inflow_dict = INFLOW.__dict__
                    inflow_data = [inflow_dict['_center'][0], inflow_dict['_center'][1], inflow_dict['_radius']]
                   
                    grp.create_dataset(name='INFLOW', data=np.asarray(inflow_data, dtype=np.float32))
        
                    def obstacle_to_data(o):
                        o_dict = o.__dict__['_geometry'].__dict__
                        # Check if sphere
                        if '_radius' in o_dict:
                            o_data = ['SPHERE', o_dict['_center'][0], o_dict['_center'][1], o_dict['_radius']]
                            
                        # Must be box
                        else:
                            o_data = ['BOX', o_dict['_lower'][0], o_dict['_lower'][1], o_dict['_upper'][0], o_dict['_upper'][1]]
                            
                        return o_data
                    
                    obstacle_list_data_ = [str(obstacle_to_data(o)) for o in obstacle_list]
                    # obstacle_list_data = []
                    # for x in obstacle_list_data_:
                    #     obstacle_list_data.extend(x)
                        
                    dt = h5py.special_dtype(vlen=str)
                    dset_obstacle_list = grp.create_dataset('obstacle_list', (len(obstacle_list),), dtype=dt)
                    for i in range(len(obstacle_list)):
                        
                        dset_obstacle_list[i] = obstacle_list_data_[i]
        
                    grp.create_dataset(
                        name='mask', data=data_masks.astype('float32'),
                        shape=data_masks.shape, maxshape=data_masks.shape, compression="gzip")

                    grp.create_dataset(
                        name='velocity_x', data=data_velocity_x.astype('float32'),
                        shape=data_velocity_x.shape, maxshape=data_velocity_x.shape, compression="gzip")
                    grp.create_dataset(
                        name='velocity_y', data=data_velocity_y.astype('float32'),
                        shape=data_velocity_y.shape, maxshape=data_velocity_y.shape, compression="gzip")

                    grp.create_dataset(
                        name='smoke', data=data_smoke.astype('float32'),
                        shape=data_smoke.shape, maxshape=data_smoke.shape, compression="gzip")

                    break

                else:

                    continue

if __name__ == '__main__':

    # VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

    VISIBLE_DEVICES = "0"
    NUM_GPUS = len(VISIBLE_DEVICES.split(","))
    NUM_PROCESSES = NUM_GPUS
    NUM_SAMPLES = 200

    with Pool(processes=NUM_PROCESSES) as pool:

        timesteps = [list(range(100))]
        
        _ = pool.starmap(save_snapshots, zip(range(NUM_PROCESSES), [NUM_SAMPLES]*NUM_PROCESSES, timesteps*NUM_PROCESSES))

        print('finished')