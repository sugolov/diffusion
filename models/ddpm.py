import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusers
from diffusers import UNet2DModel

from test.test_ddpm import *
class DDPM(nn.Module):
    """
    dim:        dimension of images
    n_steps:    number of diffusion steps
    """
    def __init__(self, dim=(32,32), n_steps=1000, noise_schedule=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.n_steps = n_steps

        if noise_schedule is not None:
            self.noise_schedule = self.noise_schedule
        else:
            self.noise_schedule = torch.linspace(1e-4, 2e-2, self.n_steps)

        self.alphas = 1 - self.noise_schedule

        self.prod_alphas = torch.cumprod(self.alphas, dim=0)

        ## TODO: implement epsilon_theta
        self.noisenet = None

    ## TODO: think about how to intelligently implement sampling and training
    def forward(self):
        pass

    def sample(self):
        pass

# TODO: implement a U-Net here, preferrably without diffusers.UNet
class NoiseNet(nn.Module):
    pass

if __name__ == "__main__":
    test_cifar10_load()