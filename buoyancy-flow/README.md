# Buoyancy-driven Flow with Obstacles

This directory contains code for reconstructions of buoyancy-driven flow with obstacles. 
Our simulation setup encompasses a fixed domain $\Omega \subset [0,1] \times [0,1]$ simulated from $t=0$ until $t=0.65$
with marker inflow at $(0.5, 0.1)$, which is active until $t=0.2$.
We use phiflow as a simulation backend, which is a differentiable PDE-solver.
For each simulation, we randomly place one to two boxes and spheres in the domain.

### Dataset 

Code to create the dataset for training and testing can be found in `data/smoke_plumes_save.py`.
We provide a Jupyter notebook to open and visualize the dataset in `data/data_visualization.ipynb`.

<p align="center">
  <img src="https://github.com/Akanota/smdp/assets/16702943/a11c166c-9f52-4e02-89d0-93216aa68d80" width="45%">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/Akanota/smdp/assets/16702943/4b41d2f6-68b5-485a-8612-0e506bb1f7e7" width="45%">
</p>

### Usage

```bash
> python train_embedded_physics.py --help

usage: train_separate_updates.py [-h] [--name NAME] --file FILE [--continue-id CONTINUE_ID]
                                 [--start-epoch START_EPOCH] [--gpu GPU] [--batch-size BATCH_SIZE]
                                 [--network-weights NETWORK_WEIGHTS] [--architecture ARCHITECTURE]
                                 [--training-noise TRAINING_NOISE] [--inference-noise INFERENCE_NOISE]
                                 [--inference-is-training] [--forward-sim] [--rollout-noise ROLLOUT_NOISE]
                                 [--test-only] [--test-file TEST_FILE] [--t1 T1] [--api-key API_KEY] [--update UPDATE]
                                 [--physics-backward PHYSICS_BACKWARD]

Parameter Parser

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of experiment (default: None)
  --file FILE           Training data (default: None)
  --continue-id CONTINUE_ID
                        ID of run to continue (default: None)
  --start-epoch START_EPOCH
                        Epoch to begin training (default: 0)
  --gpu GPU             Visible GPUs (default: None)
  --batch-size BATCH_SIZE
                        Batch size (default: 1)
  --network-weights NETWORK_WEIGHTS
                        File with weights used for initialization (default: None)
  --architecture ARCHITECTURE
                        Network architecture (default: dilated)
  --training-noise TRAINING_NOISE
                        Coefficient for noise during training (default: 1.0)
  --inference-noise INFERENCE_NOISE
                        Coefficient for noise during inference (default: 1.0)
  --inference-is-training
                        Flag if noise scales are the same for inference as for training (default: False)
  --forward-sim         Flag whether to learn probability flow ODE during forward simulation (default: False)
  --rollout-noise ROLLOUT_NOISE
                        Flag whether to include noise in rollouts (default: 0.0)
  --test-only           Flag whether to only run test (default: False)
  --test-file TEST_FILE
                        Test file (default: None)
  --t1 T1               End time of simulation (default: 0.65)
  --api-key API_KEY     Wanbb API key (default: None)
  --update UPDATE       1 for only states as network inputs (default); 2 for including physics in network inputs
                        (default: 1)
  --physics-backward PHYSICS_BACKWARD
                        What backward physics to use; either 1 for reusing negative forward physics or 2 for time
                        integration (default: 2)
```

To start the training similar to the experiments in the paper, run 

```bash
python train_embedded_physics.py --file data/smoke_plumes_r0.h5 --batch-size 4 --training-noise 0.1 --inference-noise 0.1 --test-file data/smoke_plumes_test_r0.h5
```

### Baselines

We provide implementations of three baseline methods: 
- Differentiable solvers and the limited-memory BFGS method, see `baselines/differentiable-physics`
- [Diffusion posterior sampling](https://openreview.net/forum?id=OnD9zGAGT0k), see `baselines/diffusion-posterior-sampling`
- Training an autoregressive model without physics, see `train_autoregressive.py`


