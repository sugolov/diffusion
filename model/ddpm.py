import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from unet import *

class DDPMnet(nn.Module):
    """
    dim:        dimension of images
    n_steps:    number of diffusion steps
    """

    def __init__(self,
                 dim=(3, 32, 32),
                 n_steps=1000,
                 noise_schedule=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.n_steps = n_steps

        if noise_schedule is not None:
            self.noise_schedule = self.noise_schedule
        else:
            self.noise_schedule = Parameter(torch.linspace(1e-4, 2e-2, self.n_steps), requires_grad=False)

        self.alphas = 1 - self.noise_schedule
        self.prod_alphas = Parameter(torch.cumprod(self.alphas, dim=0), requires_grad=False)
        self.noisenet = ResUNetCIFAR10()

    def forward(self, x0, noise, t):

        xt = torch.sqrt(self.prod_alphas[t]) * x0 + torch.sqrt(1 - self.prod_alphas[t]) * noise
        return self.noisenet(xt, t)


if __name__ == "__main__":
    x = torch.randn((128, 3, 32, 32))
    t = 4
    unet = ResUNetCIFAR10()
    s = unet(x, t)
    print(s.shape)


