import importlib
import os

CONFIG_MAPPING = {}

import sys
sys.path.append('..')

def get_config(name):
    global CONFIG_MAPPING

    name = name.replace('-', '_')

    if name not in CONFIG_MAPPING:
        raise ValueError(f"Config {name} not found.")
    else:
        return CONFIG_MAPPING[name]()


for module in os.listdir(os.path.dirname(__file__)):
    try:
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        # mod = importlib.import_module(module[:-3], package='configs')

        # print('configs.' + module[:-3])

        mod = importlib.import_module('configs.' + module[:-3], package='.')

        CONFIG_MAPPING[module[:-3]] = mod.get_config
        del module
    except Exception as e:
        print(f"Could not import {module} due to {e}")
