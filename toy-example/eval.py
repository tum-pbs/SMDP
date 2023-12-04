from processes import probability_flow_ode, reverse_time_sde
from utils.utils import create_eval_fn
import jax.random as jr
import jax.numpy as jnp
from jax import vmap


def evaluate_network(config, model_params, forward_fn):
    eval_fn = create_eval_fn(forward_fn, model_params)
    eval_key = jr.PRNGKey(config["eval"]["eval_seed"])
    eval_dict = {}

    if config["absorb_diffusion"]:
        eval_fn__ = lambda x, t: eval_fn(x, t) / (config["dataset"]["diffusion"] ** 2)
    else:
        eval_fn__ = eval_fn

    # correction to the score function (relevant for backwards only training)
    eval_fn_ = lambda x, t: config["C"] * eval_fn__(x, t)

    # values_probability_flow = []
    # values_reverse_time_sde = []

    sampling_interval = config["eval"]["sampling_interval"]
    num_samples = config["eval"]["samples"]

    # for i in range(num_samples):
    #    initial_value = jr.uniform(eval_key, shape=(1,), minval=sampling_interval[0], maxval=sampling_interval[1])
    #    eval_key = jr.split(eval_key, 1)[0]

    #    values_probability_flow.append(probability_flow_ode(initial_value, eval_fn,
    #                                                        config["dataset"]["physics_operator"], diffusion,
    #                                                        t0=config["dataset"]["t0"], t1=config["dataset"]["t1"],
    #                                                        dt=config["dataset"]["dt"]))

    #    values_reverse_time_sde.append(reverse_time_sde(initial_value, eval_fn,
    #                                                    config["dataset"]["physics_operator"], diffusion, eval_key,
    #                                                    t0=config["dataset"]["t0"], t1=config["dataset"]["t1"],
    #                                                    dt=config["dataset"]["dt"]))
    #    eval_key = jr.split(eval_key, 1)[0]

    sampling_points = jnp.linspace(sampling_interval[0], sampling_interval[1], num_samples)

    eval_prob_flow = lambda x: probability_flow_ode(x, eval_fn_,
                                                    config["dataset"]["drift"], config["dataset"]["diffusion"],
                                                    t0=config["dataset"]["t0"], t1=config["dataset"]["t1"],
                                                    dt=config["dataset"]["dt"])

    eval_reverse_time_sde = lambda x, key: reverse_time_sde(x, eval_fn_,
                                                            config["dataset"]["drift"], config["dataset"]["diffusion"],
                                                            key, t0=config["dataset"]["t0"], t1=config["dataset"]["t1"],
                                                            dt=config["dataset"]["dt"])

    values_probability_flow = vmap(eval_prob_flow)(sampling_points)

    key_list = []
    for _ in range(num_samples):
        key_list.append(eval_key)
        eval_key = jr.split(eval_key, 1)[0]
    key_list = jnp.array(key_list)

    values_reverse_time_sde = vmap(eval_reverse_time_sde, in_axes=(0, 0))(sampling_points, key_list)

    y_1 = 1
    y_1_hits = []
    y_2 = -1
    y_2_hits = []

    end_value = values_probability_flow[-1]

    for v in end_value:

        if jnp.abs(v - y_1) < config["eval"]["accuracy"] * jnp.abs(y_1):
            y_1_hits.append(v)
        elif jnp.abs(v - y_2) < config["eval"]["accuracy"] * jnp.abs(y_2):
            y_2_hits.append(v)

    eval_dict["probability_flow_ode"] = {"y_1_hits": len(y_1_hits) / num_samples,
                                         "y_2_hits": len(y_2_hits) / num_samples,
                                         "no_hits": (num_samples - len(y_1_hits) - len(y_2_hits)) / num_samples}

    y_1_hits = []
    y_2_hits = []

    end_value = values_reverse_time_sde[-1]

    for v in end_value:
        v = v[-1]
        if jnp.abs(v - y_1) < config["eval"]["accuracy"] * jnp.abs(y_1):
            y_1_hits.append(v)
        elif jnp.abs(v - y_2) < config["eval"]["accuracy"] * jnp.abs(y_2):
            y_2_hits.append(v)

    eval_dict["reverse_time_sde"] = {"y_1_hits": len(y_1_hits) / num_samples,
                                     "y_2_hits": len(y_2_hits) / num_samples,
                                     "no_hits": (num_samples - len(y_1_hits) - len(y_2_hits)) / num_samples}

    return eval_dict
