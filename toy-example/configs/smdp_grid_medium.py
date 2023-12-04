import jax.numpy as jnp

import sys

sys.path.insert(0, '..')

from models.models import modelType
from utils.utils import stoppingCriterionType


def get_config():
    g = 0.03
    lambda_ = 7
    physics_operator = lambda x: - jnp.sign(x) * x * x * lambda_

    samples = 250

    name = "smdp-paper"

    config = {
        "type": "smdp",
        "absorb_diffusion": True,
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
            "model_type": modelType.grid,
            "order": 1,
            "t0": 0.0,
            "t1": 10.0,
            "y0": -1.25,
            "y1": 1.25,
            "dt": 0.02,
            "dy": 0.01,
            "key": None
        },
        "training": [{
            "epochs": 1000,
            "sw_steps": 0,
            "sw_start": 2,
            "sw_increase": 0,
            "subsample": 1,
            "bidirectional": True,
            "batch_size": 256,
            "learning_rate": 0.01,
            "shuffle": False
        },
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
