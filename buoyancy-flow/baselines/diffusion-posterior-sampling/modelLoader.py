def get_model(model: str, **kwargs):
    if model == 'UNET':
        from .unet import Unet
        return Unet(**kwargs)
    else:
        raise ValueError(f'Unknown model: {model}')


def get_default_config(model: str, **kwargs):
    config = {'model': model}
    if model == 'UNET':
        config['dim'] = 64
        config['dim_mults'] = (1, 2, 2, 4,)
    else:
        raise ValueError(f'Unknown model: {model}')

    config.update(kwargs)
    return config