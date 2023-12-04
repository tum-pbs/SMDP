import wandb

from eval import evaluate_network
from utils.plots import save_snapshot_score, plot_paths
from utils.utils import save_network, create_eval_fn
from matplotlib import pyplot as plt

import jax.random as jr


def get_callbacks(config, forward_fn, log_wandb):
    def callback(model_params_, logs_):

        if logs_['grad_updates'] - logs_['last_saved'] > config['save_every']:

            logs_['last_saved'] = logs_['grad_updates']

            save_network(model_params_, logs_, config['save_dir'])

            eval_fn = create_eval_fn(forward_fn, model_params_)

            # Plot score field for each window size
            with plt.style.context("seaborn-white"):
                score_field = save_snapshot_score(eval_fn,
                                                  savename=config['save_dir']
                                                           + f'score_field_{logs_["grad_updates"]}.svg',
                                                  diffusion=config['dataset']['diffusion'],
                                                  absorb_diffusion=config['absorb_diffusion'],
                                                  c=config['C'])

                key_ = jr.PRNGKey(config["eval"]["eval_seed"])

                paths_ode, paths_sde = plot_paths(config, key_, 6, forward_fn, model_params_)

            eval_dict = evaluate_network(config, model_params_, forward_fn)

            if log_wandb:
                logs_["score_field"] = score_field

                logs_['paths_ode'] = wandb.Image(paths_ode)
                logs_['paths_sde'] = wandb.Image(paths_sde)

                logs_[f'sde_hit_1'] = eval_dict['reverse_time_sde']['y_1_hits']
                logs_[f'sde_hit_2'] = eval_dict['reverse_time_sde']['y_2_hits']
                logs_[f'sde_no_hit'] = eval_dict['reverse_time_sde']['no_hits']

                logs_[f'ode_hit_1'] = eval_dict['probability_flow_ode']['y_1_hits']
                logs_[f'ode_hit_2'] = eval_dict['probability_flow_ode']['y_2_hits']
                logs_[f'ode_no_hit'] = eval_dict['probability_flow_ode']['no_hits']

                m_ode = 2 * min(eval_dict['probability_flow_ode']['y_1_hits'],
                                eval_dict['probability_flow_ode']['y_2_hits'])
                m_sde = 2 * min(eval_dict['reverse_time_sde']['y_1_hits'],
                                eval_dict['reverse_time_sde']['y_2_hits'])

                logs_['m_ode'] = m_ode
                logs_['m_sde'] = m_sde

                wandb.log(logs_)
                del logs_["score_field"]

                del logs_["paths_ode"]
                del logs_["paths_sde"]

                del logs_[f'sde_hit_1']
                del logs_[f'sde_hit_2']
                del logs_[f'sde_no_hit']
                del logs_[f'ode_hit_1']
                del logs_[f'ode_hit_2']
                del logs_[f'ode_no_hit']

                del logs_['m_ode']
                del logs_['m_sde']

        return logs_

    return callback
