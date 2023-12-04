# Toy Problems with Simple SDEs

In this experiment, we test our method on simple SDEs with known initial distributions at $t=0$. 
In our paper, we consider a quadratic SDE given by 
$dx = \mathcal{P}(x) dt + g dW$ with $\mathcal{P}(x) := - \text{sign}(x)x^2$ and diffusion coefficient $g \equiv 0.1$. 
$W$ denotes the standard Brownian motion. The initial value of the path at $t=0$ are drawn from a Bernoulli distribution with values $-1$ and $1$ and equal probability.

### Data set

Our data sets consist of 2500 (100%), 250 (10%) and 25 (1%) trajectories from t=0.0 to t=10.0 sampled 
with a time step of $\Delta t = 0.02$. You can see a plot of the trajectories below.

<p align="center" width="100%">
    <img width="40%" src="https://github.com/Akanota/smdp/assets/16702943/595edded-7143-4e5c-9227-f0499a305b97"> 
</p>

## Training and Inference

We provide a self-contained notebook for training and inference using our method in `notebooks/training_and_inference.ipynb`.

Alternatively, you can use the following command to train a model on the toy example data set:

```bash
usage: Parameter Parser [-h] [--name NAME] [--config CONFIG] [--seed SEED]
                        [--continue-training CONTINUE_TRAINING] [--log-wandb]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of experiment (default: None)
  --config CONFIG       training configuration file (default: )
  --seed SEED           random seed (default: 0)
  --continue-training CONTINUE_TRAINING
                        id of run to be continued (previous checkpoint) (default: None)
  --log-wandb           turn on logging with wandb (default: False)
```

There are a number of training configurations in the `configs` folder which can be used to recreate the results from our paper.
For example, to train with our method, 1-step loss, the multilayer perceptron architecture, and 100% of the data set size, you can use the following command:

```bash 
python train.py --name "1-step MLP 100%" --config smdp_reg_mlp_big
```

### Visualization of learned score 

The score is approximated with a neural network $s_\theta(x,t)$. Visualizing the network outputs on a rectangular grid in the 
domain $[0, 10] \times [-1,1]$ yields the following plot:

<p align="center" width="100%">
    <img width="50%" src="https://github.com/Akanota/smdp/assets/16702943/d4a63fed-cf09-44ed-baff-1c375f2faf04"> 
</p>

For inference, blue regions have a negative score, so $\nabla_x \log p_t(x) < 0$ and the data likelihood is increased, if the trajectory falls.
On the other side, red regions indicate $\nabla_x \log p_t(x) > 0$ and trajectories will increase. 

### Probability flow ODE

The probability flow ODE of the SDE is given by $dx = \left[ \mathcal{P(x)} - \frac{1}{2} g^2 s_\theta(x,t) \right] dt$,
where $s_\theta(x,t)$ is the score network. Note that we solve the probability flow ODE from t=10.0 until t=0.0, since we 
want to solve initial states given an end value of a trajectory. Notice that paths with values $>0$ at $t=10.0$ are colored blue and end in $1$, whereas for paths $<0$ go to $-1$.

<p align="center" width="100%">
    <img width="40%" src="https://github.com/Akanota/smdp/assets/16702943/c6609ab5-6073-4a3d-ae00-7dc6f52b70e0"> 
</p>

### Reverse-time SDE

The reverse-time SDE is given by $dx = \left[ \mathcal{P(x)} - g^2 s_\theta(x,t) \right] dt + g dW$.
This is similar to the probability flow ODE but has an additional diffusion term and a difference weighting of the score approximation $s_\theta(x,t)$. 
We solve the reverse-time SDE from $t=10$ until $t=0$. Paths are now non-smooth and can cross each other.

<p align="center" width="100%">
    <img width="40%" src="https://github.com/Akanota/smdp/assets/16702943/27d6c41d-cb0d-4e5d-a25b-2d3f4d6374d2"> 
</p>

## Comparison of learned score with analytic score

In the first experiments considered here and in the paper, we have considered a quadratic SDE with known initial distribution.
Because the drift is quadratic, it is not trivial to compute the ground truth score in a closed form expression and it is also 
not possible to adapt the standard training of diffusion models for this case.

However, we consider the SDE with affine drift $dx = - \lambda x dt + g dW$.
This allows us to compute the true score in a closed form expression and we can directly compare the learned score with the
analytic score. See below for a comparison as well as data trajectories, the probability flow ODE and the reverse-time SDE.
We mask the scores in the regions where the data is very sparse. Overall, our method is able to learn the score extremely well.
The code for this is contained in a Jupyter notebook in `notebooks/analytic_score_comparison.ipynb`.

<p align="center" width="100%">
    <img width="100%" src="https://github.com/Akanota/smdp/assets/16702943/d82d4b28-fa74-47d4-8c4f-c1cdf37499a4"> 
</p>
