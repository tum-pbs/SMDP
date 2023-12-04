import torch
from tqdm import tqdm

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def sample_conditional(model, observation, diffusion):
    image_channels = 4
    image_size = 64
    n_steps = 1000

    """Sample from the model"""
    with torch.no_grad():
        x = torch.randn([1, image_channels, image_size, image_size],
                        device=observation.device)

        x[:, 0] = observation

        for t_ in tqdm(range(n_steps)):
            t = n_steps - t_ - 1

            x = diffusion.p_sample_model(x, x.new_full((1,), t, dtype=torch.long),
                                         model)

            x[:, 0] = observation

        return x


def conditional_sampling(model, observation, device):

    image_channels = 4
    image_size = 64
    n_steps = 1000
    beta = torch.linspace(0.0001, 0.02, 1000).to(device)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    sigma2 = beta

    """Sample from the model"""
    with torch.no_grad():
        x = torch.randn([1, image_channels, image_size, image_size],
                        device=observation.device)

        x[:, 0] = observation

        for t_ in tqdm(range(n_steps)):
            t = n_steps - t_ - 1

            t_in = x.new_full((1,), t, dtype=torch.long)

            eps_theta = model(x, t_in)
            # [gather](utils.html) $\bar\alpha_t$
            alpha_bar_t = gather(alpha_bar, t_in)
            # $\alpha_t$
            alpha_t = gather(alpha, t_in)
            # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
            eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
            # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
            mean = 1 / (alpha_t ** 0.5) * (x - eps_coef * eps_theta)
            # $\sigma^2$
            var_t = gather(sigma2, t_in)

            # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
            eps = torch.randn(x.shape, device=x.device)
            # Sample
            x = mean + (var_t ** .5) * eps

            x[:, 0] = observation

        return x

