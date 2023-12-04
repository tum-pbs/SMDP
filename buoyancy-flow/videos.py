import jax.numpy as jnp

from matplotlib.gridspec import GridSpec
import moviepy.editor as mp

import wandb

import matplotlib.pyplot as plt
import numpy as np

import jax.random as jr

from eval import *

from phi.jax.flow import *

phi.math.backend.set_global_default_backend(phi.jax.JAX) 

def make_axes_invisible(ax):
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def save_simulation_video_score_decoupled(log_dict, params, inference_dict, rng, data, DT, NSTEPS, noise, savename):
    
    update = inference_dict['update']
    physics_backward_type = inference_dict['physics_backward_type']
    
    correction_probability_flow_fn_ = inference_dict['probability_flow']
    correction_reverse_sde_fn_ = inference_dict['reverse_sde']
    
    physics_forward_fn = inference_dict['physics_forward']
    physics_backward_fn = inference_dict['physics_backward']
    
    correction_probability_flow_fn = lambda state, t: correction_probability_flow_fn_(params, state, t)
    correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(params, state, t)
    
    smoke_state = data['smoke']
    vel_x_state = data['vel_x']
    vel_y_state = data['vel_y']
    mask_state = data['mask']
    
    ground_truth = list(zip(jnp.expand_dims(smoke_state, axis=1), jnp.expand_dims(vel_x_state, axis=1), jnp.expand_dims(vel_y_state, axis=1), jnp.expand_dims(mask_state, axis=1)))
    
    obstacles = [data['obstacle_list']]
    obstacles = batch_geometries_pre(obstacles)
    
    simulation_metadata = {}
    
    inflow = data['INFLOW']

    center = math.tensor([(inflow['_center'][1], inflow['_center'][0])], batch('batch'), channel(vector='x,y'))
    
    simulation_metadata['INFLOW'] = Sphere(center=center, radius=inflow['_radius'])
    simulation_metadata['smoke_res'] = data['smoke_res']
    simulation_metadata['v_res'] = data['v_res']
    
    bounds = data['BOUNDS']
    
    simulation_metadata['BOUNDS'] = Box(x=(bounds['_lower'][0],bounds['_upper'][0]), y=(bounds['_lower'][1],bounds['_upper'][1]))
    
    simulation_metadata['DT'] = 0.01
    simulation_metadata['NSTEPS'] = NSTEPS
    
    vmin_smoke = np.min(smoke_state)
    vmax_smoke = np.max(smoke_state)
    vmin_vel_x = np.min(vel_x_state)
    vmax_vel_x = np.max(vel_x_state)
    vmin_vel_y = np.min(vel_y_state)
    vmax_vel_y = np.max(vel_y_state)
    
    vmin_dict = {0 : vmin_smoke, 1 : np.minimum(vmin_vel_x, vmin_vel_y), 2 : np.minimum(vmin_vel_x, vmin_vel_y)}
    vmax_dict = {0 : vmax_smoke, 1 : np.maximum(vmax_vel_x, vmax_vel_y), 2 : np.maximum(vmax_vel_x, vmax_vel_y)}
        
    data_dict = {}
    
    for i in range(3):
    
        smoke_state_init = smoke_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=smoke_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        vel_x_state_init = vel_x_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_x_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        vel_y_state_init = vel_y_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_y_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        mask_state_init = mask_state[-1][None]
        
        states_backward_rand = eval_backward_score_decoupled([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, noise, type=physics_backward_type, update=update,  physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
        data_dict[i] = states_backward_rand
        
    smoke_state_init = smoke_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=smoke_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    vel_x_state_init = vel_x_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_x_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    vel_y_state_init = vel_y_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_y_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    mask_state_init = mask_state[-1][None]
     
    states_backward_0_coeff = eval_backward_score_decoupled([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, 0, correction_coefficient=0.0, type=physics_backward_type, update=update, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
    states_backward_1_coeff = eval_backward_score_decoupled([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, 0, type=physics_backward_type, update=update, correction_coefficient=1.0, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
    def get_image(idx, title, data_dict, no_correction, probability_flow, ground_truth, batch_n=0):
    
        height = 6
        width = 9
        dpi = 100
    
        # plot three images, smoke, vel_x and vel_y
    
        images = {}
    
        name_dict = {0 : 'Smoke ', 1 : 'Velocity x ', 2: 'Velocity y '}
       
        for n in range(3):
    
            elem_shape = data_dict[0][idx][n][batch_n].shape
    
            fig = plt.figure(figsize=(width, height))
            fig.set_dpi(dpi)

            gs = GridSpec(6, 9, figure=fig)

            rand_ax_dict = {}

            rand_ax_dict[0] = fig.add_subplot(gs[0:3, 0:3])
            rand_ax_dict[1] = fig.add_subplot(gs[0:3, 3:6])

            rand_ax_dict[2] = fig.add_subplot(gs[3:6, 0:3])

            for i in range(3):
                disp_ = data_dict[i][idx][n][batch_n] + (1-data_dict[i][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
                rand_ax_dict[i].imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
                make_axes_invisible(rand_ax_dict[i])

            ax_probability_flow = fig.add_subplot(gs[3:6, 3:6])
            make_axes_invisible(ax_probability_flow)
            
            ax_no_correction = fig.add_subplot(gs[0:3, 6:9])
            make_axes_invisible(ax_no_correction)

            disp_ = probability_flow[idx][n][batch_n] + (1-probability_flow[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_probability_flow.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_probability_flow.text(0.1, 0.9, 'probability flow', color='white', ha='left', va='center', transform=ax_probability_flow.transAxes, fontsize='large')

            disp_ = no_correction[idx][n][batch_n] + (1-no_correction[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_no_correction.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_no_correction.text(0.1, 0.9, 'no correction', color='white', ha='left', va='center', transform=ax_no_correction.transAxes, fontsize='large')

            ax_gt = fig.add_subplot(gs[3:6, 6:9])
            make_axes_invisible(ax_gt)
            
            disp_ = ground_truth[idx][n][batch_n] + (1-ground_truth[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_gt.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_gt.text(0.1, 0.9, 'ground truth', color='white', ha='left', va='center', transform=ax_gt.transAxes, fontsize='large')

            plt.suptitle(name_dict[n] + title)

            plt.tight_layout()

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images[n] = image.reshape(dpi * height, dpi * width, 3)
            
            plt.close(fig)
        
        return images
    
    images = []
    for i in range(NSTEPS):
        images.append(get_image(i, f'backward - {i * DT: .3f}', data_dict, states_backward_0_coeff, states_backward_1_coeff, ground_truth[::-1]))
        
    for i in range(10):
    
        images.append(images[-1])
        
    # Forward Simulation 
        
    data_dict_forward = {}
        
    for i in range(3):
    
        data_dict_forward[i] = eval_forward(data_dict[i][-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
        
    states_backward_0_coeff_forward = eval_forward(states_backward_0_coeff[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    states_backward_1_coeff_forward = eval_forward(states_backward_1_coeff[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    ground_truth_forward = eval_forward(ground_truth[0], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    for i in range(NSTEPS):
        images.append(get_image(i, f'forward - {(i+1) * DT: .3f}', data_dict_forward, 
                                states_backward_0_coeff_forward, states_backward_1_coeff_forward, ground_truth_forward))
    
    for i in range(10):
    
        images.append(images[-1])
    
    smoke_images = [im[0] for im in images]
    vel_x_images = [im[1] for im in images]
    vel_y_images = [im[2] for im in images]
    
    clip = mp.ImageSequenceClip(smoke_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_smoke.mp4', fps=8)
    log_dict[f'{savename}_smoke'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_smoke.mp4')
    
    clip = mp.ImageSequenceClip(vel_x_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_x.mp4', fps=8)
    log_dict[f'{savename}_vel_x'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_x.mp4')
    
    clip = mp.ImageSequenceClip(vel_y_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_y.mp4', fps=8)
    log_dict[f'{savename}_vel_y'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_y.mp4')
    
    return log_dict
    
def save_simulation_video_score(log_dict, params, inference_dict, rng, data, DT, NSTEPS, noise, savename):
    
    update = inference_dict['update']
    physics_backward_type = inference_dict['physics_backward_type']
    
    correction_probability_flow_fn_ = inference_dict['probability_flow']
    correction_reverse_sde_fn_ = inference_dict['reverse_sde']
    
    physics_forward_fn = inference_dict['physics_forward']
    physics_backward_fn = inference_dict['physics_backward']
    
    correction_probability_flow_fn = lambda state, t: correction_probability_flow_fn_(params, state, t)
    correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(params, state, t)
    
    smoke_state = data['smoke']
    vel_x_state = data['vel_x']
    vel_y_state = data['vel_y']
    mask_state = data['mask']
    
    ground_truth = list(zip(jnp.expand_dims(smoke_state, axis=1), jnp.expand_dims(vel_x_state, axis=1), jnp.expand_dims(vel_y_state, axis=1), jnp.expand_dims(mask_state, axis=1)))
    
    obstacles = [data['obstacle_list']]
    obstacles = batch_geometries_pre(obstacles)
    
    simulation_metadata = {}
    
    inflow = data['INFLOW']

    center = math.tensor([(inflow['_center'][1], inflow['_center'][0])], batch('batch'), channel(vector='x,y'))
    
    simulation_metadata['INFLOW'] = Sphere(center=center, radius=inflow['_radius'])
    simulation_metadata['smoke_res'] = data['smoke_res']
    simulation_metadata['v_res'] = data['v_res']
    
    bounds = data['BOUNDS']
    
    simulation_metadata['BOUNDS'] = Box(x=(bounds['_lower'][0],bounds['_upper'][0]), y=(bounds['_lower'][1],bounds['_upper'][1]))
    
    simulation_metadata['DT'] = 0.01
    simulation_metadata['NSTEPS'] = NSTEPS
    
    vmin_smoke = np.min(smoke_state)
    vmax_smoke = np.max(smoke_state)
    vmin_vel_x = np.min(vel_x_state)
    vmax_vel_x = np.max(vel_x_state)
    vmin_vel_y = np.min(vel_y_state)
    vmax_vel_y = np.max(vel_y_state)
    
    vmin_dict = {0 : vmin_smoke, 1 : np.minimum(vmin_vel_x, vmin_vel_y), 2 : np.minimum(vmin_vel_x, vmin_vel_y)}
    vmax_dict = {0 : vmax_smoke, 1 : np.maximum(vmax_vel_x, vmax_vel_y), 2 : np.maximum(vmax_vel_x, vmax_vel_y)}
        
    data_dict = {}
    
    for i in range(3):
    
        smoke_state_init = smoke_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=smoke_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        vel_x_state_init = vel_x_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_x_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        vel_y_state_init = vel_y_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_y_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        mask_state_init = mask_state[-1][None]
        
        states_backward_rand = eval_backward_score([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_reverse_sde_fn, rng, noise, type=physics_backward_type, update=update,  physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
        data_dict[i] = states_backward_rand
        
    smoke_state_init = smoke_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=smoke_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    vel_x_state_init = vel_x_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_x_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    vel_y_state_init = vel_y_state[-1][None] + jnp.sqrt(DT) * noise * jr.normal(rng, shape=vel_y_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    mask_state_init = mask_state[-1][None]
     
    states_backward_0_coeff = eval_backward_score([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, 0, correction_coefficient=0.0, type=physics_backward_type, update=update, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
    states_backward_1_coeff = eval_backward_score([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, rng, 0, type=physics_backward_type, update=update, correction_coefficient=1.0, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
    def get_image(idx, title, data_dict, no_correction, probability_flow, ground_truth, batch_n=0):
    
        height = 6
        width = 9
        dpi = 100
    
        # plot three images, smoke, vel_x and vel_y
    
        images = {}
    
        name_dict = {0 : 'Smoke ', 1 : 'Velocity x ', 2: 'Velocity y '}
       
        for n in range(3):
    
            elem_shape = data_dict[0][idx][n][batch_n].shape
    
            fig = plt.figure(figsize=(width, height))
            fig.set_dpi(dpi)

            gs = GridSpec(6, 9, figure=fig)

            rand_ax_dict = {}

            rand_ax_dict[0] = fig.add_subplot(gs[0:3, 0:3])
            rand_ax_dict[1] = fig.add_subplot(gs[0:3, 3:6])

            rand_ax_dict[2] = fig.add_subplot(gs[3:6, 0:3])

            for i in range(3):
                disp_ = data_dict[i][idx][n][batch_n] + (1-data_dict[i][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
                rand_ax_dict[i].imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
                make_axes_invisible(rand_ax_dict[i])

            ax_probability_flow = fig.add_subplot(gs[3:6, 3:6])
            make_axes_invisible(ax_probability_flow)
            
            ax_no_correction = fig.add_subplot(gs[0:3, 6:9])
            make_axes_invisible(ax_no_correction)

            disp_ = probability_flow[idx][n][batch_n] + (1-probability_flow[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_probability_flow.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_probability_flow.text(0.1, 0.9, 'probability flow', color='white', ha='left', va='center', transform=ax_probability_flow.transAxes, fontsize='large')

            disp_ = no_correction[idx][n][batch_n] + (1-no_correction[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_no_correction.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_no_correction.text(0.1, 0.9, 'no correction', color='white', ha='left', va='center', transform=ax_no_correction.transAxes, fontsize='large')

            ax_gt = fig.add_subplot(gs[3:6, 6:9])
            make_axes_invisible(ax_gt)
            
            disp_ = ground_truth[idx][n][batch_n] + (1-ground_truth[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_gt.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_gt.text(0.1, 0.9, 'ground truth', color='white', ha='left', va='center', transform=ax_gt.transAxes, fontsize='large')

            plt.suptitle(name_dict[n] + title)

            plt.tight_layout()

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images[n] = image.reshape(dpi * height, dpi * width, 3)
            
            plt.close(fig)
        
        return images
    
    images = []
    for i in range(NSTEPS):
        images.append(get_image(i, f'backward - {i * DT: .3f}', data_dict, states_backward_0_coeff, states_backward_1_coeff, ground_truth[::-1]))
        
    for i in range(10):
    
        images.append(images[-1])
        
    # Forward Simulation 
        
    data_dict_forward = {}
        
    for i in range(3):
    
        data_dict_forward[i] = eval_forward(data_dict[i][-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
        
    states_backward_0_coeff_forward = eval_forward(states_backward_0_coeff[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    states_backward_1_coeff_forward = eval_forward(states_backward_1_coeff[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    ground_truth_forward = eval_forward(ground_truth[0], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    for i in range(NSTEPS):
        images.append(get_image(i, f'forward - {(i+1) * DT: .3f}', data_dict_forward, 
                                states_backward_0_coeff_forward, states_backward_1_coeff_forward, ground_truth_forward))
    
    for i in range(10):
    
        images.append(images[-1])
    
    smoke_images = [im[0] for im in images]
    vel_x_images = [im[1] for im in images]
    vel_y_images = [im[2] for im in images]
    
    clip = mp.ImageSequenceClip(smoke_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_smoke.mp4', fps=8)
    log_dict[f'{savename}_smoke'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_smoke.mp4')
    
    clip = mp.ImageSequenceClip(vel_x_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_x.mp4', fps=8)
    log_dict[f'{savename}_vel_x'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_x.mp4')
    
    clip = mp.ImageSequenceClip(vel_y_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_y.mp4', fps=8)
    log_dict[f'{savename}_vel_y'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_y.mp4')
    
    return log_dict

    
def save_simulation(data, DT, NSTEPS, savename):
    
    origDT = 0.01
    DTscale = origDT / DT
    
    vmin_smoke = np.min(data['ground_truth'][0][0])
    vmax_smoke = np.max(data['ground_truth'][0][0])
    vmin_vel_x = np.min(data['ground_truth'][0][1])
    vmax_vel_x = np.max(data['ground_truth'][0][1])
    vmin_vel_y = np.min(data['ground_truth'][0][2])
    vmax_vel_y = np.max(data['ground_truth'][0][2])
    
    vmin_dict = {0 : vmin_smoke, 1 : np.minimum(vmin_vel_x, vmin_vel_y), 2 : np.minimum(vmin_vel_x, vmin_vel_y)}
    vmax_dict = {0 : vmax_smoke, 1 : np.maximum(vmax_vel_x, vmax_vel_y), 2 : np.maximum(vmax_vel_x, vmax_vel_y)}
    
    
    def get_image(idx, title, data, batch_n = 0):
    
        height = 6
        width = 9
        dpi = 100
    
        # plot three images, smoke, vel_x and vel_y
    
        images = {}
    
        name_dict = {0 : 'Smoke ', 1 : 'Velocity x ', 2: 'Velocity y '}
       
        for n in range(3):
    
            elem_shape = data['ground_truth'][int(idx/DTscale)][n][batch_n].shape
    
            fig = plt.figure(figsize=(width, height))
            fig.set_dpi(dpi)

            gs = GridSpec(6, 9, figure=fig)
          
            ax_reverse_sde_pc = fig.add_subplot(gs[0:3, 0:3])
            make_axes_invisible(ax_reverse_sde_pc)
            disp_ = data['reverse_sde_0_pc'][idx][n][batch_n] + (1-data['reverse_sde_0_pc'][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_reverse_sde_pc.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_reverse_sde_pc.text(0.1, 0.9, 'reverse sde pc', color='white', ha='left', va='center', transform=ax_reverse_sde_pc.transAxes, fontsize='large')

            ax_reverse_sde = fig.add_subplot(gs[0:3, 3:6])
            make_axes_invisible(ax_reverse_sde)
            disp_ = data['reverse_sde_0'][idx][n][batch_n] + (1-data['reverse_sde_0'][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_reverse_sde.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_reverse_sde.text(0.1, 0.9, 'reverse sde', color='white', ha='left', va='center', transform=ax_reverse_sde.transAxes, fontsize='large')
            
            ax_probability_flow_noise = fig.add_subplot(gs[3:6, 0:3])
            make_axes_invisible(ax_probability_flow_noise)
            disp_ = data['probability_flow_noise'][idx][n][batch_n] + (1-data['probability_flow_noise'][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_probability_flow_noise.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_probability_flow_noise.text(0.1, 0.9, 'probability flow noise', color='white', ha='left', va='center', transform=ax_probability_flow_noise.transAxes, fontsize='large')

            ax_probability_flow = fig.add_subplot(gs[3:6, 3:6])
            make_axes_invisible(ax_probability_flow)
            disp_ = data['probability_flow'][idx][n][batch_n] + (1-data['probability_flow'][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_probability_flow.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_probability_flow.text(0.1, 0.9, 'probability flow', color='white', ha='left', va='center', transform=ax_probability_flow.transAxes, fontsize='large')

            ax_no_correction = fig.add_subplot(gs[0:3, 6:9])
            make_axes_invisible(ax_no_correction)
            disp_ = data['no_correction'][idx][n][batch_n] + (1-data['no_correction'][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_no_correction.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_no_correction.text(0.1, 0.9, 'no correction', color='white', ha='left', va='center', transform=ax_no_correction.transAxes, fontsize='large')

            ax_gt = fig.add_subplot(gs[3:6, 6:9])
            make_axes_invisible(ax_gt)
            disp_ = data['ground_truth'][int(idx/DTscale)][n][batch_n] + (1-data['ground_truth'][int(idx/DTscale)][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_gt.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_gt.text(0.1, 0.9, 'ground truth', color='white', ha='left', va='center', transform=ax_gt.transAxes, fontsize='large')

            plt.suptitle(name_dict[n] + title)

            plt.tight_layout()

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images[n] = image.reshape(dpi * height, dpi * width, 3)
            
            plt.close(fig)
        
        return images
    
    images = []
    for i in range(NSTEPS):
        images.append(get_image(i, f'backward - {i * DT: .3f}', data))
        
    for i in range(10):
    
        images.append(images[-1])
        
    smoke_images = [im[0] for im in images]
    vel_x_images = [im[1] for im in images]
    vel_y_images = [im[2] for im in images]
    
    clip = mp.ImageSequenceClip(smoke_images, fps=8)
    clip.write_videofile(f'videos/{savename}_smoke.mp4', fps=8)
    
    clip = mp.ImageSequenceClip(vel_x_images, fps=8)
    clip.write_videofile(f'videos/{savename}_vel_x.mp4', fps=8)
    
    clip = mp.ImageSequenceClip(vel_y_images, fps=8)
    clip.write_videofile(f'videos/{savename}_vel_y.mp4', fps=8)
    
    return 
    
    
def save_simulation_video(log_dict, params, inference_dict, rng, data, DT, NSTEPS, score_drift, score_noise, noise, savename):
    
    correction_probability_flow_fn_ = inference_dict['probability_flow']
    correction_reverse_sde_fn_ = inference_dict['reverse_sde']
    
    score_fn = inference_dict['score']
    
    physics_forward_fn = inference_dict['physics_forward']
    physics_backward_fn = inference_dict['physics_backward']
    
    correction_probability_flow_fn = lambda state, t: correction_probability_flow_fn_(params, state, t)
    correction_reverse_sde_fn = lambda state, t: correction_reverse_sde_fn_(params, state, t)
    
    smoke_state = data['smoke']
    vel_x_state = data['vel_x']
    vel_y_state = data['vel_y']
    mask_state = data['mask']
    
    ground_truth = list(zip(jnp.expand_dims(smoke_state, axis=1), jnp.expand_dims(vel_x_state, axis=1), jnp.expand_dims(vel_y_state, axis=1), jnp.expand_dims(mask_state, axis=1)))
    
    obstacles = [data['obstacle_list']]
    obstacles = batch_geometries_pre(obstacles)
    
    simulation_metadata = {}
    
    inflow = data['INFLOW']

    center = math.tensor([(inflow['_center'][1], inflow['_center'][0])], batch('batch'), channel(vector='x,y'))
    
    simulation_metadata['INFLOW'] = Sphere(center=center, radius=inflow['_radius'])
    simulation_metadata['smoke_res'] = data['smoke_res']
    simulation_metadata['v_res'] = data['v_res']
    
    bounds = data['BOUNDS']
    
    simulation_metadata['BOUNDS'] = Box(x=(bounds['_lower'][0],bounds['_upper'][0]), y=(bounds['_lower'][1],bounds['_upper'][1]))
    
    simulation_metadata['DT'] = 0.01
    simulation_metadata['NSTEPS'] = NSTEPS
    
    vmin_smoke = np.min(smoke_state)
    vmax_smoke = np.max(smoke_state)
    vmin_vel_x = np.min(vel_x_state)
    vmax_vel_x = np.max(vel_x_state)
    vmin_vel_y = np.min(vel_y_state)
    vmax_vel_y = np.max(vel_y_state)
    
    vmin_dict = {0 : vmin_smoke, 1 : np.minimum(vmin_vel_x, vmin_vel_y), 2 : np.minimum(vmin_vel_x, vmin_vel_y)}
    vmax_dict = {0 : vmax_smoke, 1 : np.maximum(vmax_vel_x, vmax_vel_y), 2 : np.maximum(vmax_vel_x, vmax_vel_y)}
        
    data_dict = {}
    
    for i in range(3):
    
        smoke_state_init = smoke_state[-1][None] + DT * noise * jr.normal(rng, shape=smoke_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        vel_x_state_init = vel_x_state[-1][None] + DT * noise * jr.normal(rng, shape=vel_x_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        vel_y_state_init = vel_y_state[-1][None] + DT * noise * jr.normal(rng, shape=vel_y_state[-1][None].shape)
        rng, _ = jax.random.split(rng)
        mask_state_init = mask_state[-1][None]
        
        states_backward_rand = eval_backward_rand([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_reverse_sde_fn, score_fn, rng, score_drift, score_noise, noise, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
        data_dict[i] = states_backward_rand
        
    smoke_state_init = smoke_state[-1][None] + DT * noise * jr.normal(rng, shape=smoke_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    vel_x_state_init = vel_x_state[-1][None] + DT * noise * jr.normal(rng, shape=vel_x_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    vel_y_state_init = vel_y_state[-1][None] + DT * noise * jr.normal(rng, shape=vel_y_state[-1][None].shape)
    rng, _ = jax.random.split(rng)
    mask_state_init = mask_state[-1][None]
     
    states_backward_0_coeff = eval_backward_rand([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, score_fn, rng, score_drift, score_noise, 0, correction_coefficient=0.0, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    states_backward_1_coeff = eval_backward_rand([smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], obstacles, simulation_metadata, correction_probability_flow_fn, score_fn, rng, score_drift, score_noise, 0, correction_coefficient=1.0, physics_forward_fn=physics_forward_fn, physics_backward_fn=physics_backward_fn)
    
    def get_image(idx, title, data_dict, no_correction, probability_flow, ground_truth, batch_n=0):
    
        height = 6
        width = 9
        dpi = 100
    
        # plot three images, smoke, vel_x and vel_y
    
        images = {}
    
        name_dict = {0 : 'Smoke ', 1 : 'Velocity x ', 2: 'Velocity y '}
       
        for n in range(3):
    
            elem_shape = data_dict[0][idx][n][batch_n].shape
    
            fig = plt.figure(figsize=(width, height))
            fig.set_dpi(dpi)

            gs = GridSpec(6, 9, figure=fig)

            rand_ax_dict = {}

            rand_ax_dict[0] = fig.add_subplot(gs[0:3, 0:3])
            rand_ax_dict[1] = fig.add_subplot(gs[0:3, 3:6])

            rand_ax_dict[2] = fig.add_subplot(gs[3:6, 0:3])

            for i in range(3):
                disp_ = data_dict[i][idx][n][batch_n] + (1-data_dict[i][idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
                rand_ax_dict[i].imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
                make_axes_invisible(rand_ax_dict[i])

            ax_probability_flow = fig.add_subplot(gs[3:6, 3:6])
            make_axes_invisible(ax_probability_flow)
            
            ax_no_correction = fig.add_subplot(gs[0:3, 6:9])
            make_axes_invisible(ax_no_correction)

            disp_ = probability_flow[idx][n][batch_n] + (1-probability_flow[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_probability_flow.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_probability_flow.text(0.1, 0.9, 'probability flow', color='white', ha='left', va='center', transform=ax_probability_flow.transAxes, fontsize='large')

            disp_ = no_correction[idx][n][batch_n] + (1-no_correction[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_no_correction.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_no_correction.text(0.1, 0.9, 'no correction', color='white', ha='left', va='center', transform=ax_no_correction.transAxes, fontsize='large')

            ax_gt = fig.add_subplot(gs[3:6, 6:9])
            make_axes_invisible(ax_gt)
            
            disp_ = ground_truth[idx][n][batch_n] + (1-ground_truth[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_gt.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_gt.text(0.1, 0.9, 'ground truth', color='white', ha='left', va='center', transform=ax_gt.transAxes, fontsize='large')

            plt.suptitle(name_dict[n] + title)

            plt.tight_layout()

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images[n] = image.reshape(dpi * height, dpi * width, 3)
            
            plt.close(fig)
        
        return images
    
    images = []
    for i in range(NSTEPS):
        images.append(get_image(i, f'backward - {i * DT: .3f}', data_dict, states_backward_0_coeff, states_backward_1_coeff, ground_truth[::-1]))
        
    for i in range(10):
    
        images.append(images[-1])
        
    # Forward Simulation 
        
    data_dict_forward = {}
        
    for i in range(3):
    
        data_dict_forward[i] = eval_forward(data_dict[i][-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
        
    states_backward_0_coeff_forward = eval_forward(states_backward_0_coeff[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    states_backward_1_coeff_forward = eval_forward(states_backward_1_coeff[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    ground_truth_forward = eval_forward(ground_truth[0], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    for i in range(NSTEPS):
        images.append(get_image(i, f'forward - {(i+1) * DT: .3f}', data_dict_forward, 
                                states_backward_0_coeff_forward, states_backward_1_coeff_forward, ground_truth_forward))
    
    for i in range(10):
    
        images.append(images[-1])
    
    smoke_images = [im[0] for im in images]
    vel_x_images = [im[1] for im in images]
    vel_y_images = [im[2] for im in images]
    
    clip = mp.ImageSequenceClip(smoke_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_smoke.mp4', fps=8)
    log_dict[f'{savename}_smoke'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_smoke.mp4')
    
    clip = mp.ImageSequenceClip(vel_x_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_x.mp4', fps=8)
    log_dict[f'{savename}_vel_x'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_x.mp4')
    
    clip = mp.ImageSequenceClip(vel_y_images, fps=8)
    clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_y.mp4', fps=8)
    log_dict[f'{savename}_vel_y'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_y.mp4')
    
    return log_dict

def save_simulation_video_conv(log_dict, params, inference_dict, rng, data, DT, NSTEPS, savename):
    
    network_fn = inference_dict['network']
    physics_forward_fn = inference_dict['physics_forward']
    
    
    smoke_state = data['smoke']
    vel_x_state = data['vel_x']
    vel_y_state = data['vel_y']
    mask_state = data['mask']
    
    origDT = 0.01
    DTscale = int(origDT / DT)
    
    ground_truth = list(zip(jnp.expand_dims(smoke_state, axis=1), jnp.expand_dims(vel_x_state, axis=1), jnp.expand_dims(vel_y_state, axis=1), jnp.expand_dims(mask_state, axis=1)))
    
    ground_truth_ = []
    for elem in ground_truth:
        ground_truth_.extend([elem, elem])
    ground_truth = ground_truth_
    
    obstacles = [data['obstacle_list']]
    obstacles = batch_geometries_pre(obstacles)
    
    simulation_metadata = {}
    
    inflow = data['INFLOW']

    center = math.tensor([(inflow['_center'][1], inflow['_center'][0])], batch('batch'), channel(vector='x,y'))
    
    simulation_metadata['INFLOW'] = Sphere(center=center, radius=inflow['_radius'])
    simulation_metadata['smoke_res'] = data['smoke_res']
    simulation_metadata['v_res'] = data['v_res']
    
    bounds = data['BOUNDS']
    
    simulation_metadata['BOUNDS'] = Box(x=(bounds['_lower'][0],bounds['_upper'][0]), y=(bounds['_lower'][1],bounds['_upper'][1]))
    
    simulation_metadata['DT'] = DT
    simulation_metadata['NSTEPS'] = NSTEPS
    
    vmin_smoke = np.min(smoke_state)
    vmax_smoke = np.max(smoke_state)
    vmin_vel_x = np.min(vel_x_state)
    vmax_vel_x = np.max(vel_x_state)
    vmin_vel_y = np.min(vel_y_state)
    vmax_vel_y = np.max(vel_y_state)
    
    vmin_dict = {0 : vmin_smoke, 1 : np.minimum(vmin_vel_x, vmin_vel_y), 2 : np.minimum(vmin_vel_x, vmin_vel_y)}
    vmax_dict = {0 : vmax_smoke, 1 : np.maximum(vmax_vel_x, vmax_vel_y), 2 : np.maximum(vmax_vel_x, vmax_vel_y)}  
    
    smoke_state_init = smoke_state[-1][None]
    vel_x_state_init = vel_x_state[-1][None] 
    vel_y_state_init = vel_y_state[-1][None]
    mask_state_init = mask_state[-1][None]    
    states_backward = eval_backward_conv(params, [smoke_state_init, vel_x_state_init, vel_y_state_init, mask_state_init], simulation_metadata, network_fn)
    
    def get_image(idx, title, prediction, ground_truth, batch_n=0):
    
        height = 3
        width = 6
        dpi = 100
    
        # plot three images, smoke, vel_x and vel_y
    
        images = {}
    
        name_dict = {0 : 'Smoke ', 1 : 'Velocity x ', 2: 'Velocity y '}
       
        for n in range(3):
    
            elem_shape = prediction[idx][n][batch_n].shape
    
            fig = plt.figure(figsize=(width, height))
            fig.set_dpi(dpi)

            gs = GridSpec(3, 6, figure=fig)

            
            ax_prediction = fig.add_subplot(gs[0:3, 0:3])
            
            disp_ = prediction[idx][n][batch_n] + (1-prediction[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_prediction.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            make_axes_invisible(ax_prediction)
            ax_prediction.text(0.1, 0.9, 'prediction', color='white', ha='left', va='center', transform=ax_prediction.transAxes, fontsize='large')

            ax_gt = fig.add_subplot(gs[0:3, 3:6])
            make_axes_invisible(ax_gt)            
            disp_ = ground_truth[idx][n][batch_n] + (1-ground_truth[idx][3][batch_n][0:elem_shape[0], 0:elem_shape[1]]) * vmax_dict[n]
            ax_gt.imshow(jnp.flip(disp_, axis=0), cmap='jet', vmin=vmin_dict[n], vmax=vmax_dict[n])
            ax_gt.text(0.1, 0.9, 'ground truth', color='white', ha='left', va='center', transform=ax_gt.transAxes, fontsize='large')

            plt.suptitle(name_dict[n] + title)

            plt.tight_layout()

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images[n] = image.reshape(dpi * height, dpi * width, 3)
            
            plt.close(fig)
        
        return images
    
    images = []
    for i in range(NSTEPS):
        images.append(get_image(i, f'backward - {i * DT: .3f}', states_backward, ground_truth[::-1]))
        
    for i in range(10):
    
        images.append(images[-1])
        
    # Forward Simulation 
   
    prediction_forward = eval_forward(states_backward[-1], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    ground_truth_forward = eval_forward(ground_truth[0], obstacles, simulation_metadata, physics_forward_fn = physics_forward_fn)
    
    for i in range(NSTEPS):
        images.append(get_image(i, f'forward - {(i+1) * DT: .3f}', prediction_forward, ground_truth_forward))
    
    for i in range(10):
    
        images.append(images[-1])
    
    smoke_images = [im[0] for im in images]
    vel_x_images = [im[1] for im in images]
    vel_y_images = [im[2] for im in images]
    
    clip = mp.ImageSequenceClip(smoke_images, fps=8)
    clip.write_videofile(f'videos/{savename}_smoke.mp4', fps=8)
    log_dict[f'{savename}_smoke'] = wandb.Video(f'videos/{savename}_smoke.mp4')
    # clip.write_videofile(f'videos/{wandb.run.id}_{savename}_smoke.mp4', fps=8)
    # log_dict[f'{savename}_smoke'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_smoke.mp4')
    
    clip = mp.ImageSequenceClip(vel_x_images, fps=8)
    clip.write_videofile(f'videos/{savename}_vel_x.mp4', fps=8)
    log_dict[f'{savename}_vel_x'] = wandb.Video(f'videos/{savename}_vel_x.mp4')
    # clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_x.mp4', fps=8)
    # log_dict[f'{savename}_vel_x'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_x.mp4')
    
    clip = mp.ImageSequenceClip(vel_y_images, fps=8)
    clip.write_videofile(f'videos/{savename}_vel_y.mp4', fps=8)
    log_dict[f'{savename}_vel_y'] = wandb.Video(f'videos/{savename}_vel_y.mp4')
    # clip.write_videofile(f'videos/{wandb.run.id}_{savename}_vel_y.mp4', fps=8)
    # log_dict[f'{savename}_vel_y'] = wandb.Video(f'videos/{wandb.run.id}_{savename}_vel_y.mp4')
    
    return log_dict