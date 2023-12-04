import jax.nn
import jax.numpy as jnp

import sys

sys.path.insert(0, '..')

from models.models import modelType
from utils.utils import stoppingCriterionType

def get_config():

    g = 0.03
    lambda_ = 7
    physics_operator = lambda x: - jnp.sign(x) * x * x * lambda_

    samples = 2500

    name = "smdp_reg_mlp"

    config = {
        "type": "smdp-reg",
        "absorb_diffusion": True,
        "C": 1.0,
        "name": name,
        "wandb": False,
        "save_every": 100000,
        'dataset': {
            'samples': samples,
            'name': f'toy-example-{samples}',
            'drift': physics_operator,
            'diffusion': g,
            'dt': 0.02,
            'seed': 0,
            't0': 0.0,
            't1': 10.0,
        },
        "model": {
            "model_type": modelType.mlp,
            'output_sizes': [30, 30, 25, 20, 10, 1],
            'log_t': True,
            'activation': jax.nn.elu,
            'key': None
        },
        "training": [{
                "epochs": 250,
                "subsample": 5,
                "batch_size": 256,
                "learning_rate": 0.001,
                "shuffle": True},
            {
                "epochs": 250,
                "subsample": 1,
                "batch_size": 256,
                "learning_rate": 0.0001,
                "shuffle": True},
            {
                "epochs": 750,
                "subsample": 1,
                "batch_size": 256,
                "learning_rate": 0.00001,
                "shuffle": True},
        ],
        "stopping_criterion": {
            "type": stoppingCriterionType.MAX_TIME,
            "tol": 60 * 23,  # 23 hours
        },
        "eval": {
            "samples": 1000,
            "accuracy": 0.1,  # 10%
            "sampling_interval": [-0.1, 0.1],
            "eval_seed": 0,
        }
    }

    return config


