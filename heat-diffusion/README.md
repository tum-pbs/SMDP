# Heat Diffusion

The heat equation is a parabolic partial differential equation (PDE) that describes the distribution of heat (or variation in temperature) in a given region over time.
The PDE is given by $$\frac{\partial u}{\partial t} = \alpha \Delta u$$ with diffusivity constant $\alpha = 1.0$ in our case.

Below, we visualize initial conditions that are being evolved over time from $t=0$ until $t=0.2$ with the heat equation.

<p align="center">
  <img src="https://github.com/Akanota/smdp/assets/16702943/48b2eb3c-1be2-4d24-a9e1-2b71ab89a77a" width="30%" />
  <img src="https://github.com/Akanota/smdp/assets/16702943/378a40c4-03ad-4fb0-9e12-0cb6d203beb7" width="30%" />
</p>

### Dataset

As dataset, we consider Gaussian random fields with different resolutions and different power law exponents. 
There is a tutorial Jupyter notebook for generating the dataset files located at `data/gaussian_random_fields.ipynb`.
In the paper, we consider Gaussian random fields with power law exponent $n=4.0$ at resolution $32 \times 32$. 

<p align="center">
  <img src="https://github.com/Akanota/smdp/assets/16702943/e8d23782-0025-4eb9-b71d-100927ca9026" width="70%" />
</p>

### Stochastic heat equation and solving the PDE with Euler-Maruyama

To embed the heat diffusion directly into the framework of score-based generative modelling, we want to write the solution of the heat equation $\mathbf{x}_t$ over time $t$ as a stochastic differential equation (SDE), i.e.,
$$d\mathbf{x}=P(\mathbf{x})dt+g(t)dW.$$

To do that, we rely on a discretization of the spatial domains, so that we can write spatial derivatives, e.g., $\Delta u$, as finite differences. 
Then, we can solve the system with the Euler method and a solver that evolves the system to the next time step, i.e., for two adjacent time steps $t_{i}$ and $t_{i+1}$, we can write 
$$x_{t_{i+1}}=x_{t_{i}}+P_{\Delta t}(x_{t_{i}}) \Delta t.$$

Additionally, we add Gaussian noise to the system, i.e., $g(t) dW$ with $g \equiv 0.1$, which resembles the stochastic heat equation.

To solve the system, we use the Euler-Maruyama method, which is a stochastic version of the Euler method. 

We provide a self-contained notebook to solve the heat equation with the Euler-Maruyama method and the Python library `diffrax` in `notebooks/heat_diffusion_spectral_solver.ipynb`.

Below we visualize different solutions at $t=0.2$ obtained with the Euler-Maruyama method and noise scale $g\equiv 0.1$.

<p align="center">
  <img src="https://github.com/Akanota/smdp/assets/16702943/36798e7f-4014-4fbb-be17-c3989cdf972a" width="70%" />
</p>
 
## Training

To train models to learn corrections that approximate the score for a given initial distribution, we provide a training script `train.py` that can be used as follows:

```bash
> python train.py --help
usage: Parameter Parser [-h] [--name NAME] --train-file TRAIN_FILE
                        [--val-file VAL_FILE] [--test-file TEST_FILE]
                        [--continue-id CONTINUE_ID] [--gpu GPU]
                        [--batch-size BATCH_SIZE] [--start_epoch START_EPOCH]
                        [--step-size STEP_SIZE] [--architecture ARCHITECTURE]
                        [--resolution RESOLUTION] [--lr LR] [--t1 T1]
                        [--api-key API_KEY]
                        [--network-weights NETWORK_WEIGHTS] [--test-only]
                        {EmbeddedPhysics,Autoregressive,SSM,FNO,Bayesian} ...

positional arguments:
  {EmbeddedPhysics,Autoregressive,SSM,FNO,Bayesian}
                        Training method

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of experiment (default: None)
  --train-file TRAIN_FILE
                        Training data (default: None)
  --val-file VAL_FILE   Validation data (default: None)
  --test-file TEST_FILE
                        Testing data (default: None)
  --continue-id CONTINUE_ID
                        ID of run to continue (default: None)
  --gpu GPU             Visible GPUs (default: None)
  --batch-size BATCH_SIZE
                        Batch size (default: 16)
  --start_epoch START_EPOCH
                        Start epoch (default: 0)
  --step-size STEP_SIZE
                        Step size for simulation (default: 0.01)
  --architecture ARCHITECTURE
                        Network architecture (default: EncoderDecoder)
  --resolution RESOLUTION
                        Resolution of data (default: 32)
  --lr LR               Learning rate (default: 0.0001)
  --t1 T1               End time of simulation (default: 0.2)
  --api-key API_KEY     Wanbd API key (default: None)
  --network-weights NETWORK_WEIGHTS
                        File with weights used for initialization (default:
                        None)
  --test-only           Only do final tests (default: False)

   ```
Models are saved in the directory `weights` and training logs are stored in the directory `wandb`.   

For example, to start a training similar to the experiments in the paper, run 

```bash
python train.py --name HeatDiffusion --train-file data/files/32/2d_gaussian_random_field_4.h5 --test-file data/files/32/2d_gaussian_random_field_4_test.h5 --batch-size 4 EmbeddedPhysics
```

The method `EmbeddedPhysics` corresponds to our proposed method.

We also provide implementations of baselines methods: Fourier neural operators `FNO`, autoregressive models with ResNets `Autoregressive`, sliced score matching `SSM`, and Bayesian neural networks with spatial dropout `Bayesian`.

## Inference

We provide a self-contained notebook that performs inference using the Euler-Maruyama method (for the SDE version of our method) 
and trained network weights in `notebooks/inference.ipynb`.

Below, we visualize the results of inference with our proposed SDE and ODE methods on the heat diffusion example.

<p align="center">
  <img src="https://github.com/Akanota/smdp/assets/16702943/cc083920-a11d-4e76-a827-d3b9e4fe9228" width="100%" />
</p>
