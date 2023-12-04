import jax.numpy as jnp
from diffrax import VirtualBrownianTree, MultiTerm, ODETerm, WeaklyDiagonalControlTerm, \
    Euler, SaveAt, diffeqsolve
import jax.random as jr

import diffrax as dfx


def toy_sde(drift, x0: float, diffusion: float, key: jr.PRNGKey, t0=0.0, t1=10.0):
    """
    Toy SDE with drift and diffusion: dx = drift(x) dt + diffusion dW

    :param drift: drift function (independent of time); takes x as input and returns float
    :param x0: initial value at t = 0
    :param diffusion: diffusion coefficient of brownian motion
    :param key: jr.PRNGKey to reproduce results
    :param t0: start time, usually 0
    :param t1: end time, usually 10
    :return: diffrax solution object
    """
    initial_shape = (1,)
    y0 = jnp.ones(shape=initial_shape) * x0

    drift_lambda = lambda t, y, args: drift(y)

    diffusion_lambda = lambda t, y, args: diffusion * jnp.ones(initial_shape)

    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=initial_shape, key=key)
    terms = MultiTerm(ODETerm(drift_lambda), WeaklyDiagonalControlTerm(diffusion_lambda, brownian_motion))
    solver = Euler()

    saveat = SaveAt(dense=True)

    sol = diffeqsolve(terms, solver, t0, t1, dt0=0.01, y0=y0, saveat=saveat)

    return sol


def probability_flow_ode(initial_value, eval_fn, physics_operator, diffusion, t0=0.0, t1=10.0, dt=0.01):
    """
    Probability flow ODE for the toy example with drift and diffusion: dx = drift(x) dt + diffusion dW
    :param initial_value: initial value at t = t1
    :param eval_fn: eval function of score function
    :param physics_operator: drift of the toy example
    :param diffusion: diffusion coefficient of brownian motion
    :param t0: end time for the probability flow ODE inference
    :param t1: start time for the probability flow ODE inference
    :param dt: time step; if t0 < t1, dt should be negative, else positive
    :return: path represented by list of values
    """
    initial_shape = (1,)
    y1 = jnp.ones(shape=initial_shape) * initial_value

    t0 = jnp.array(t0)
    t1 = jnp.array(t1)
    dt = jnp.array(dt)

    def drift(t, y, args):
        return physics_operator(y) - 0.5 * (diffusion ** 2) * eval_fn(y, t)

    terms = ODETerm(drift)
    solver = dfx.Euler()

    args = None
    tprev = jnp.array(t1)
    tnext = jnp.array(t1 - dt)
    y = y1

    state = solver.init(terms, tprev, tnext, y1, args)

    y_list = []

    for i in range(((t1 - t0) / dt).astype(int)):
        y, _, _, state, _ = solver.step(terms, tprev, tnext, y, args, state, made_jump=False)

        tprev = tnext
        tnext = jnp.array(jnp.maximum(tprev - dt, t0))

        y_list.append(y)

    return y_list


def reverse_time_sde(initial_value, eval_fn, physics_operator, diffusion, key, t0=0.0, t1=10.0, dt=0.01):
    """
        Probability flow SDE for the toy example with drift and diffusion: dx = drift(x) dt + diffusion dW
        :param initial_value: initial value at t = t1
        :param eval_fn: eval function of score function
        :param physics_operator: drift of the toy example
        :param diffusion: diffusion coefficient of brownian motion
        :param key: jr.PRNGKey for brownian motion
        :param t0: end time for the probability flow SDE inference
        :param t1: start time for the probability flow SDE inference
        :param dt: time step; if t0 < t1, dt should be negative, else positive
        :return: path represented by list of values
        """
    initial_shape = (1,)
    y1 = jnp.ones(shape=initial_shape) * initial_value

    t0 = jnp.array(t0)
    t1 = jnp.array(t1)
    dt = jnp.array(dt)

    def drift(t, y, args):
        return physics_operator(y) - (diffusion ** 2) * eval_fn(y, t)

    diffusion_fn = lambda t, y, args: diffusion * jnp.ones(initial_shape)

    brownian_motion = VirtualBrownianTree(t1, t0, tol=1e-3, shape=initial_shape, key=key)
    terms = MultiTerm(ODETerm(drift), WeaklyDiagonalControlTerm(diffusion_fn, brownian_motion))

    solver = dfx.Euler()

    args = None
    tprev = jnp.array(t1)
    tnext = jnp.array(t1 - dt)
    y = y1

    state = solver.init(terms, tprev, tnext, y1, args)

    y_list = []

    for i in range(((t1 - t0) / dt).astype(int)):
        y, _, _, state, _ = solver.step(terms, tprev, tnext, y, args, state, made_jump=False)

        tprev = tnext

        tnext = jnp.array(jnp.maximum(tprev - dt, t0))

        y_list.append(y)

    return y_list


