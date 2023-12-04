import jax.numpy as jnp

import sys
import jax
sys.path.insert(0, '..')

from models.models import modelType
from utils.utils import stoppingCriterionType


def get_config():
    g = 0.03
    lambda_ = 7
    physics_operator = lambda x: - jnp.sign(x) * x * x * lambda_

    samples = 2500

    name = "smdp-paper"

    config = {
        "type": "smdp",
        "absorb_diffusion": True,
        "C": 0.5,
        "name": name,
        "wandb": False,
        "save_every": 50000,
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
        "training": [
            {
                "epochs": 1000,
                "sw_steps": 0,
                "sw_start": 2,
                "sw_increase": 0,
                "subsample": 5,
                "bidirectional": False,
                "batch_size": 512,
                "learning_rate": 0.001,
                "shuffle": False
            },
            {
                "epochs": 1000,
                "sw_steps": 8,
                "sw_start": 2,
                "sw_increase": 1,
                "subsample": 1,
                "bidirectional": False,
                "batch_size": 512,
                "learning_rate": 0.0001,
                "shuffle": False
            }],
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
