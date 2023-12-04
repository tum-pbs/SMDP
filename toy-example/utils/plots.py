import jax.numpy as jnp
import jax.random as jr

from utils.utils import create_eval_fn
from dataset import iterate_batches

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax import vmap

from processes import probability_flow_ode, reverse_time_sde

dpi = 200
height = 6
width = 4


def _save_snapshot_score(eval_fn, meshgrid, savename=None, batch_size=1000, key=None, diffusion=0.03,
                         absorb_diffusion=True, c=1.0):
    """
    Save snapshot of score function
    Parameters
    ----------
    eval_fn: evaluation function of score model
    meshgrid: meshgrid of points, used to evaluate eval_fn
    savename: name of file to save to
    batch_size: batch size for evaluation
    key: random key for evaluation of eval_fn (only relevant if eval_fn is stochastic)
    diffusion: diffusion coefficient; used for normalization when plotting
    """

    if key is None:
        key = jr.PRNGKey(0)

    if absorb_diffusion:
        eval_fn__ = lambda x, t: eval_fn(x, t) / (diffusion ** 2)
    else:
        eval_fn__ = eval_fn

    # correction to the score function (relevant for backwards only training)
    eval_fn_ = lambda x, t: c * eval_fn__(x, t)

    x, y = meshgrid
    full_domain = jnp.hstack((x.flatten()[:, None], y.flatten()[:, None]))

    generator = iterate_batches(full_domain, batch_size, shuffle=False)
    results = []

    for batch in generator:
        inputs = batch[0]
        t_batch = inputs[:, 0:1]
        x_batch = inputs[:, 1:2]

        results.append(eval_fn_(x_batch, t_batch))

    results = jnp.concatenate(results, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(height, width))
    fig.set_dpi(dpi)

    u_pred = results.flatten()

    # replace third argument with 2 arrays of size n corresponding to finer resolution
    # u_pred_interpolated = griddata(full_domain, u_pred, full_domain, method='cubic')

    u_pred_interpolated = u_pred.reshape(x.shape)

    vmax = 75
    h = ax.imshow(jnp.flip(u_pred_interpolated, axis=0), cmap='seismic',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  aspect='auto', vmin=-vmax, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.set_xlabel(r"$t$", color="black", size=20)
    ax.set_ylabel(r"$x$", color="black", size=20)

    ax.tick_params(labelsize=15)

    ax.grid(False)

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')

    fig.canvas.draw()

    if savename:
        file_path = savename
        plt.savefig(file_path, transparent=True)

    plt.show()

    return fig


def save_snapshot_score(eval_fn, t0=0.0, t1=10.0, **kwargs):
    """
    Save snapshot of score function
    Parameters
    ----------
    eval_fn: evaluation function of score model
    t0: start time
    t1: end time
    kwargs: additional arguments for _save_snapshot_score
    """
    x = jnp.linspace(t0, t1, 400)
    y = jnp.linspace(-1.25, 1.25, 200)

    meshgrid = jnp.meshgrid(x, y)

    return _save_snapshot_score(eval_fn, meshgrid, **kwargs)


def plot_paths(config, key, n_samples, forward_fn, model_params):
    eval_fn = create_eval_fn(forward_fn, model_params)

    if config["absorb_diffusion"]:
        eval_fn__ = lambda x, t: eval_fn(x, t) / (config["dataset"]["diffusion"] ** 2)
    else:
        eval_fn__ = eval_fn

    # correction to the score function (relevant for backwards only training)
    eval_fn_ = lambda x, t: config["C"] * eval_fn__(x, t)

    sampling_interval = config["eval"]["sampling_interval"]

    sampling_points = jnp.linspace(sampling_interval[0], sampling_interval[1], n_samples)

    eval_prob_flow = lambda x: probability_flow_ode(x, eval_fn_,
                                                    config["dataset"]["drift"], config["dataset"]["diffusion"],
                                                    t0=config["dataset"]["t0"], t1=config["dataset"]["t1"],
                                                    dt=config["dataset"]["dt"])

    eval_reverse_time_sde = lambda x, key: reverse_time_sde(x, eval_fn_,
                                                            config["dataset"]["drift"], config["dataset"]["diffusion"],
                                                            key, t0=config["dataset"]["t0"], t1=config["dataset"]["t1"],
                                                            dt=config["dataset"]["dt"])

    key_list = jr.split(key, n_samples)

    values_reverse_time_sde = jnp.array(vmap(eval_reverse_time_sde, in_axes=(0, 0))(sampling_points, key_list))
    values_probability_flow_ode = jnp.array(vmap(eval_prob_flow)(sampling_points))

    def plot_paths_(values, label):

        values = values[:, :, 0].transpose()

        fig, ax = plt.subplots(figsize=(6, 4))

        for path in values:

            if path[-1] > 0:
                col = 'blue'
            else:
                col = 'red'

            ax.plot(jnp.linspace(config["dataset"]["t1"], config["dataset"]["t0"], len(path)), path, color=col)

        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.set_xlabel(r'$t$', size=20)
        ax.set_ylabel(r'$x$', size=20)
        ax.grid(True)
        ax.tick_params(labelsize=15)

        plt.xlim([config["dataset"]["t1"] + 0.25, config["dataset"]["t0"] - 0.25])
        plt.ylim([-1.25, 1.25])

        plt.title(label, size=20)

        plt.show()

        return fig

    fig_ode = plot_paths_(values_probability_flow_ode, "Probability flow ODE")
    fig_sde = plot_paths_(values_reverse_time_sde, "Reverse time SDE")

    return fig_ode, fig_sde
