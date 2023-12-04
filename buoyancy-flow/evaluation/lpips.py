import matplotlib.pyplot
import numpy as np
import lpips
import torch

loss_fn_alex = lpips.LPIPS(net='alex')


def lpips_dist(x0, x1):
    cmap = matplotlib.pyplot.get_cmap(name='viridis')
    min_x0 = np.min(x0)
    max_x0 = np.max(x0)
    min_x1 = np.min(x1)
    max_x1 = np.max(x1)
    x0 = 2 * (cmap((x0 - min_x0) / (max_x0 - min_x0)) - 0.5)
    x1 = 2 * (cmap((x1 - min_x1) / (max_x0 - min_x1)) - 0.5)
    x0 = x0[..., :3]
    x1 = x1[..., :3]

    x0 = np.transpose(x0, (0, 3, 1, 2))
    x1 = np.transpose(x1, (0, 3, 1, 2))

    x0 = torch.Tensor(x0)
    x1 = torch.Tensor(x1)

    return loss_fn_alex(x0, x1).detach().numpy()[0, 0, 0, 0]