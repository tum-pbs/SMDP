import argparse
import os

import optax
import wandb

from callbacks import get_callbacks
from dataset import get_dataset
from eval import evaluate_network
from models.models import get_model
from utils.stopping_criterion import StoppingCriterion
from utils.utils import get_random_id, load_network, create_default_update_fn, save_network
import jax.random as jr

from configs import get_config


def main(name, config, seed=0, continue_training=None, log_wandb=False):
    config = get_config(config)

    config['seed'] = seed
    config['name'] = name

    if log_wandb:

        import wandb

        wandb.init(project="diffusion-toy-problem", name=name, resume="allow", id=continue_training)
        wandb.config.update(config)

        run_id = wandb.run.id

    else:
        run_id = get_random_id()

    config['id'] = run_id

    config['save_dir'] = f'./results/{run_id}/'
    os.makedirs(config['save_dir'], exist_ok=True)

    train_network(config, log_wandb=log_wandb, continue_training=continue_training)


def train_network(config, log_wandb=False, continue_training=False):
    """
    Train network from config with smdp
    :param config: config dictionary for training
    :param log_wandb: whether to log to wandb
    :param continue_training: try continuing training from previous checkpoint
    :return:
    """

    if config['type'] == 'smdp':
        from methods.differentiable_physics import update_network, gradient_fn
    elif config['type'] == 'smdp-reg':
        from methods.differentiable_physics_regularization import update_network, gradient_fn
    elif config['type'] == 'ism':
        from methods.implicit_score_matching import update_network, gradient_fn
    elif config['type'] == 'ssm-vr':
        from methods.sliced_score_matching import update_network, gradient_fn
    else:
        raise NotImplementedError

    key = jr.PRNGKey(config['seed'])

    # Load data
    dataset = get_dataset(**config["dataset"], rebuild=False)

    # Initialize network
    forward_fn, model_params = get_model(**config["model"])
    key = jr.split(key)[0]

    logs = {"step": 0, "grad_updates": 0, "epoch_acc": 0, "last_saved": 0}
    opt_state = None

    # load network parameters
    if continue_training:

        try:
            model_params, logs, opt_state = load_network(config["save_dir"])
        except:
            print('Could not load network parameters. Starting from scratch.')

    stopping_criterion = StoppingCriterion(config['stopping_criterion']['type'], config['stopping_criterion']['tol'])

    if stopping_criterion(logs['epoch_acc'], logs['grad_updates']):
        print('Stopping criterion reached. Stopping training.')
        return model_params, logs

    model_loss_fn = gradient_fn(forward_fn, config["dataset"]['drift'])

    callbacks = get_callbacks(config, forward_fn, log_wandb)

    # Train network

    for train_config in config['training']:

        scheduler = optax.piecewise_constant_schedule(
            init_value=train_config['learning_rate'])

        # Optax optimizer using Adam
        opt = optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.99),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0))

        if opt_state is None:
            opt_state = opt.init(model_params)

        grad_update = create_default_update_fn(opt, model_loss_fn)

        model_dict = {'forward_fn': forward_fn, 'model_params': model_params, 'grad_update': grad_update}

        model_params, logs = update_network(model_dict=model_dict, dataset=dataset, opt_state=opt_state,
                                            logs=logs,
                                            training_config=train_config, training_callback=callbacks,
                                            key=key, stopping_criterion=stopping_criterion, log_wandb=log_wandb)

        logs["epoch"] = 0
        logs["rollout"] = 0

        save_network(model_params, logs, config['save_dir'])

        opt_state = None

    eval_dict = evaluate_network(config, model_params, forward_fn)

    for key_dict in eval_dict['reverse_time_sde']:
        if log_wandb:
            wandb.run.summary[f'reverse_time_sde_{key_dict}'] = eval_dict['reverse_time_sde'][key_dict]

    for key_dict in eval_dict['probability_flow_ode']:
        if log_wandb:
            wandb.run.summary[f'probability_flow_ode_{key_dict}'] = eval_dict['probability_flow_ode'][key_dict]

    if log_wandb:
        m_ode = 2 * min(eval_dict['probability_flow_ode']['y_1_hits'], eval_dict['probability_flow_ode']['y_2_hits'])
        m_sde = 2 * min(eval_dict['reverse_time_sde']['y_1_hits'], eval_dict['reverse_time_sde']['y_2_hits'])

        wandb.run.summary['m_ode'] = m_ode
        wandb.run.summary['m_sde'] = m_sde

    return model_params, logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parameter Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', required=True, help='name of experiment')
    parser.add_argument('--config', required=True, help='training configuration file')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--continue-training', default=None, help='id of run to be continued (previous checkpoint)')
    parser.add_argument('--log-wandb', help='turn on logging with wandb', action='store_true')

    args, unknown = parser.parse_known_args()

    main(**vars(args))
