import argparse
import wandb

from dataloader import *
from utils.utils import *
from models.ModelLoader import *

hyperparameter_defaults = {
    'name': None,
    'train_file': None,
    'val_file': None,
    'test_file': None,
    'continue_id': None,
    'gpu': None,
    'batch_size': 16,
    'method': None,
    'step_size': 0.01,
    'resolution': 32,
    'lr': 1e-4,
    'seed': 0,
    't1': 0.2,
    't0': 0.0,
    'api_key': None,
    'simulations_per_epoch': 1000,
    'epochs': 80,
    'start_epoch': 0,
    'finetune_lr_steps': 20,
    'finetune_lr_gamma': 0.5,
    'network_weights': None,
    'test_only': False,
    'architecture': 'EncoderDecoder',
    'embedded_physics': {
        'training_noise': 1.0,
        'time_steps_per_batch': 10,
        'num_steps': 20,
        'ROLLOUT_epochs': 2,
        'ROLLOUT_begin': 6,
        'ROLLOUT_add': 2,
        'ROLLOUT_increase': 7,
        'noise_during_training': False,
        'forward_sim': True,
        'backward_sim': True,
        'CUT_GRADIENT': 32,
        'time_format': 'linear',
        'training': {
            'noise': 1.0
        },
        'inference': {
            'noise': 1.0
        },
        'zero_forward': False,
        'inference_is_training': False,
        'update_step': 1
    },
    'ssm': {
        'time_format': 'linear',
        'M': 1,
        'inference_is_training': False,
        'training': {
            'noise': 1.0
        },
        'inference': {
            'noise': 1.0
        },
    },
    'autoregressive': {
        'step_size': 0.2 / 32,
        'time_steps_per_batch': 10,
        'ROLLOUT_epochs': 2,
        'ROLLOUT_begin': 6,
        'ROLLOUT_add': 2,
        'ROLLOUT_increase': 7,
        'noise_during_training': True,
        'training': {
            'noise': 0.1
        },
        'inference': {
            'noise': 0.1
        },
        'inference_is_training': False
    },
    'fno': {
        'modes': 12,
        'width': 20,
        'noise': 0.01,
        'loss': 'supervised',
        'scheduler_step': 20,
        'scheduler_gamma': 0.5,
    },

    'bayesian': {
        'noise': 0.01,
        'dropout_rate': 0.0,
        'loss': 'supervised',
        'scheduler_step': 20,
        'scheduler_gamma': 0.5,
    },
}


def get_embedded_physics_opt(*args):
    from optimizers.EmbeddedPhysics import EmbeddedPhysics
    return EmbeddedPhysics(*args)


def get_autoregressive_opt(*args):
    from optimizers.Autoregressive import Autoregressive
    return Autoregressive(*args)


def get_ssm_opt(*args):
    from optimizers.SlicedScoreMatching import SlicedScoreMatching
    return SlicedScoreMatching(*args)


def get_bayesian_opt(*args):
    from optimizers.Bayesian import Bayesian
    return Bayesian(*args)


def get_fno_opt(*args):
    from optimizers.FNO import FNO
    return FNO(*args)


def main(params):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".7"

    if (not 'api_key' in params) or (params['api_key'] is None):
        os.environ["WANDB_MODE"] = 'dryrun'
    else:
        os.environ["WANDB_API_KEY"] = params['api_key']

    if (not 'name' in params) or (params['name'] is None):
        name = 'SMDP-Heat-Equation'
    else:
        name = params['name']

    if 'gpu' in params and not (params['gpu'] is None):
        os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']

    if params['continue_id']:
        wandb.init(id=params['continue_id'], resume='allow', config=params,
                   project='heat-diffusion', name=name)
    else:
        wandb.init(config=params, project='heat-diffusion', name=name)

    key = jr.PRNGKey(params['seed'])

    with h5py.File(params['train_file'], 'r') as f:
        train_keys = list(f.keys())
        train_keys = list(zip([params['train_file']] * len(train_keys), train_keys))

    generator = DataLoader([params['train_file']], train_keys, int(params['batch_size']), name='train',
                           resolution=int(params['resolution']), t1=params['t1'])

    data_mean, data_std, data_min, data_max = generator.get_norm()

    generator.set_mean_and_std(data_mean, data_std)

    if params['val_file']:
        with h5py.File(params['val_file'], 'r') as f:
            val_keys = list(f.keys())
            val_keys = list(zip([params['val_file']] * len(val_keys), val_keys))

        val_generator = DataLoader([params['val_file']], train_keys, int(params['batch_size']), name='val',
                                   resolution=int(params['resolution']), t1=params['t1'], shuffle=False)
        val_generator.set_mean_and_std(data_mean, data_std)

    else:
        print('No validation set specified..')
        val_generator = None

    if params['test_file']:

        with h5py.File(params['test_file'], 'r') as f:
            test_keys = list(f.keys())
            test_keys = list(zip([params['test_file']] * len(test_keys), test_keys))

        test_generator = DataLoader([params['test_file']], test_keys, int(params['batch_size']), name='test',
                                    resolution=int(params['resolution']), t1=params['t1'], shuffle=False)

        test_generator.set_mean_and_std(data_mean, data_std)

    else:

        print('No test set specified.. using training set as testing set, watch out!')
        test_generator = generator

    data_shape = (1, int(params['resolution']), int(params['resolution']))

    params['data_mean'] = data_mean
    params['data_std'] = data_std
    params['data_min'] = data_min
    params['data_max'] = data_max
    params['data_shape'] = data_shape

    model_key, train_key, loader_key, sample_key = jr.split(key, 4)

    params['model_key'] = model_key
    params['train_key'] = model_key
    params['loader_key'] = model_key
    params['sample_key'] = model_key

    methods = {'EmbeddedPhysics': get_embedded_physics_opt, 'Autoregressive': get_autoregressive_opt,
               'Bayesian': get_bayesian_opt, 'FNO': get_fno_opt, 'SSM': get_ssm_opt}

    generators = (generator, val_generator, test_generator)

    opt_constructor = methods[params['method']]
    opt = opt_constructor(generators, params)

    model_weights = opt.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parameter Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default=None, help='Name of experiment')
    parser.add_argument('--train-file', required=True, help='Training data')
    parser.add_argument('--val-file', help='Validation data')
    parser.add_argument('--test-file', help='Testing data')
    parser.add_argument('--continue-id', default=None, help='ID of run to continue')
    parser.add_argument('--gpu', default=None, help='Visible GPUs')
    parser.add_argument('--batch-size', default=hyperparameter_defaults['batch_size'], type=int, help='Batch size')

    parser.add_argument('--start_epoch', default=hyperparameter_defaults['start_epoch'], type=int, help='Start epoch')

    parser.add_argument('--step-size', default=hyperparameter_defaults['step_size'], help='Step size for simulation')
    parser.add_argument('--architecture', default=hyperparameter_defaults['architecture'], help='Network architecture')
    parser.add_argument('--resolution', default=hyperparameter_defaults['resolution'], type=int,
                        help='Resolution of data')
    parser.add_argument('--lr', default=hyperparameter_defaults['lr'], help='Learning rate')
    parser.add_argument('--t1', default=hyperparameter_defaults['t1'], help='End time of simulation')
    parser.add_argument('--api-key', default=hyperparameter_defaults['api_key'], help='Wanbd API key')
    parser.add_argument('--network-weights', default=hyperparameter_defaults['network_weights'],
                        help='File with weights used for initialization')
    parser.add_argument('--test-only', help='Only do final tests', action='store_true')

    subparsers = parser.add_subparsers(help='Training method', dest='method')

    embedded_physics_parser = subparsers.add_parser('EmbeddedPhysics')
    embedded_physics_parser.add_argument('--training-noise',
                                         default=hyperparameter_defaults['embedded_physics']['training']['noise'],
                                         type=float, help='Coefficient for noise during training')
    embedded_physics_parser.add_argument('--inference-noise',
                                         default=hyperparameter_defaults['embedded_physics']['inference']['noise'],
                                         type=float, help='Coefficient for noise during training')
    embedded_physics_parser.add_argument('--inference-is-training',
                                         help='Flag if score parameters are the same for inference as for training',
                                         action='store_true')
    embedded_physics_parser.add_argument('--noise-during-training', help='Add noise during training steps',
                                         action='store_true')
    embedded_physics_parser.add_argument('--time-steps-per-batch',
                                         default=hyperparameter_defaults['embedded_physics']['time_steps_per_batch'],
                                         type=int, help='Time steps (picked randomly from 0 to t1/DT)')
    embedded_physics_parser.add_argument('--ROLLOUT-epochs',
                                         default=hyperparameter_defaults['embedded_physics']['ROLLOUT_epochs'],
                                         type=int, help='epochs for each rollout length')
    embedded_physics_parser.add_argument('--ROLLOUT-begin',
                                         default=hyperparameter_defaults['embedded_physics']['ROLLOUT_begin'],
                                         type=int, help='Initial rollout length')
    embedded_physics_parser.add_argument('--ROLLOUT-add',
                                         default=hyperparameter_defaults['embedded_physics']['ROLLOUT_add'],
                                         type=int,
                                         help='Value by which current rollout will be increased after ROLLOUT_epochs')
    embedded_physics_parser.add_argument('--ROLLOUT-increase',
                                         default=hyperparameter_defaults['embedded_physics']['ROLLOUT_increase'],
                                         type=int, help='Number of times the rollout is increased')
    embedded_physics_parser.add_argument('--forward-sim', help='Also train forward simulation', action='store_true')
    embedded_physics_parser.add_argument('--zero-forward', help='Train score to be zero during forward pass',
                                         action='store_true')
    embedded_physics_parser.add_argument('--num-steps',
                                         default=hyperparameter_defaults['embedded_physics']['num_steps'],
                                         type=int, help='Number of steps in time discretization')
    embedded_physics_parser.add_argument('--time-format',
                                         default=hyperparameter_defaults['embedded_physics']['time_format'],
                                         help='Linear or Log time discretization')
    embedded_physics_parser.add_argument('--CUT-GRADIENT',
                                         default=hyperparameter_defaults['embedded_physics']['CUT_GRADIENT'],
                                         type=int, help='how many steps the gradient should be backpropagated')
    embedded_physics_parser.add_argument('--update-step',
                                         default=hyperparameter_defaults['embedded_physics']['update_step'],
                                         type=int, help='1 or 2 update steps during solver step')

    autoregressive_parser = subparsers.add_parser('Autoregressive')

    ssm_parser = subparsers.add_parser('SSM')
    ssm_parser.add_argument('--M', default=hyperparameter_defaults['ssm']['M'], help='Number of accumulation steps')

    fno_parser = subparsers.add_parser('FNO')
    fno_parser.add_argument('--loss', default=hyperparameter_defaults['fno']['loss'], help='Loss')
    fno_parser.add_argument('--noise', default=hyperparameter_defaults['fno']['noise'], help='Noise end state')

    bayesian_parser = subparsers.add_parser('Bayesian')
    bayesian_parser.add_argument('--dropout-rate', default=hyperparameter_defaults['bayesian']['dropout_rate'],
                                 help='Dropout rate', type=float)
    bayesian_parser.add_argument('--loss', default=hyperparameter_defaults['bayesian']['loss'], help='Loss')
    bayesian_parser.add_argument('--noise', default=hyperparameter_defaults['bayesian']['noise'],
                                 help='Noise end state')

    args, _ = parser.parse_known_args()

    print('*********************')
    print(args)
    print('*********************')
    args_embedded_physics, _ = embedded_physics_parser.parse_known_args()
    print(args)
    print('*********************')

    params = parse_arguments(vars(args), hyperparameter_defaults)
    params['embedded_physics'] = parse_arguments(vars(args_embedded_physics),
                                                 hyperparameter_defaults['embedded_physics'])

    main(params)
