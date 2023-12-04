# Solving Inverse Physics Problems with Score Matching

This repository contains the code for the paper "Solving Inverse Physics Problems with Score Matching" by [Benjamin Holzschuh](https://ge.in.tum.de/about/benjamin-holzschuh/), [Simona Vegetti](https://www.mpa-garching.mpg.de/person/44138/), and [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/). The paper can be found [here](https://arxiv.org/abs/2301.10250).


Our works proposes a novel approach to solve inverse problems involving the temporal evolution of physics systems by leveraging the idea of score matching. The system’s current state is moved backward in time step by step by combining an approximate inverse physics simulator and a learned correction function. A central
insight of our work is that training the learned correction with a single-step loss is equivalent to a score matching objective, while recursively predicting longer
parts of the trajectory during training relates to maximum likelihood training of a corresponding probability flow. In the paper, we highlight the advantages of our algorithm compared to standard denoising score matching and implicit score matching, as well as fully learned baselines for a wide range of inverse physics problems. 
The resulting inverse solver has excellent accuracy and temporal stability and, in contrast to other learned inverse solvers, allows for sampling the posterior of the solutions. 

Feel free to contact us if you have questions or suggestions regarding our work.

## Method Overview  
<p align="center">
  <img src="https://github.com/tum-pbs/SMDP/assets/16702943/cdf7b296-d1b1-4e55-be32-eb590003e7c0" width="90%" />
</p>

## Installation and Requirements

The code is written in Python 3.8 and tested with CUDA 11.4:

```bash conda create -n smdp python=3.8```

The majority of the code is based on JAX, which we install first with 

```bash 
conda activate smdp

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` 

For each experiment, additional packages can be installed with 

```bash
pip install -r requirements.txt
```

within the experiment's directory.

## Project Structure

Our code covers several experiments. Each experiment is located in a separate folder. 

### Toy Problems with Simple SDEs

We learn score fields for 1D-processes with simple SDEs. The experiments are located in the folder `toy-example`.
The simple setup allows us to compare the score learned by our method with the analytical score and 
evaluate how well the posterior distribution obtained from our method matches the true posterior distribution.

![toy_example_thumb](https://github.com/tum-pbs/SMDP/assets/16702943/af589b4d-513f-479b-979d-88cad1b34636)

### Burgers' Equation

In a slightly more involved example, we learn the score field for Burgers' equation. The experiments are located in the folder `burgers-equation`.
A difficulty here is that the physics is very sensitive to small perturbations. Therefore, the 1-step training of our method is not sufficient to produce
stable trajectories over longer time horizons. This is why our proposed multi-step training is crucial for this example. <b> Coming soon. </b>

### Heat Diffusion

In this example, we learn the score field for the stochastic heat diffusion equation. The experiments are located in the folder `heat-diffusion`.  
As the diffusive nature of the equation destroys information over time, small-scale structures need to be created during inference. 
This highlights the advanatages of the SDE version of our method, as noise added to the trajectories can be used to create missing details.

![heat_equation_example](https://github.com/tum-pbs/SMDP/assets/16702943/f4bb5200-058a-430e-9c95-ab82fa3016ac)


### Buoyancy-driven Flow with Obstacles

This example is located in the folder `buoyancy-flow`. We learn the score field for a buoyancy-driven flow with obstacles. 
What makes this experiment challenging is that it involves non-linear physics and randomly placed obstacles for each simulation. 
This means that the learned score field needs to be able to generalize very well to unseen scenarios.


![buoyancy_flow_overview](https://github.com/tum-pbs/SMDP/assets/16702943/b410e0e8-a1e0-47a5-ba07-2f59728aeeea)

### Isotropic Forced Turbulence

Finally, we learn the score field for isotropic forced turbulence. The experiments are located in the folder `navier-stokes`.
In this example, we do not have a numerical solver for the forward problem. Instead, we train a (time-independent) neural 
network for the physics and a (time-dependent) neural network for the score field. <b> Coming soon. </b>
## Citation

If you use our approach or code, please cite our paper:

```
@article{holzschuh2023smdp,
  title="{Solving Inverse Physics Problems with Score Matching}",
  author={Holzschuh, Benjamin and Vegetti, Simona and Thuerey, Nils},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
