import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.unet import *


class DDPMnet(nn.Module):
    """
    dim:        dimension of images
    n_steps:    number of diffusion steps
    """

    def __init__(self,
                 unet_config,
                 n_steps=1000,
                 noise_schedule_start=1e-4,
                 noise_schedule_end=2e-2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.n_steps = n_steps

        self.noise_schedule = Parameter(torch.linspace(noise_schedule_start, noise_schedule_end, n_steps), requires_grad=False)

        self.alphas = 1 - self.noise_schedule

        self.prod_alphas = Parameter(torch.cumprod(self.alphas, dim=0), requires_grad=False)

        self.noise_net = UNet(**unet_config)

    def forward(self, x0, noise, t):
        xt = torch.sqrt(self.prod_alphas[t]) * x0 + torch.sqrt(1 - self.prod_alphas[t]) * noise
        return self.noise_net(xt, t)
