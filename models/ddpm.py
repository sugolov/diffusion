import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, ReLU

import diffusers
from diffusers import UNet2DModel

from test.test_ddpm import *


class DDPM(nn.Module):
    """
    dim:        dimension of images
    n_steps:    number of diffusion steps
    """

    def __init__(self, dim=(32, 32), n_steps=1000, noise_schedule=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.n_steps = n_steps

        if noise_schedule is not None:
            self.noise_schedule = self.noise_schedule
        else:
            self.noise_schedule = torch.linspace(1e-4, 2e-2, self.n_steps)

        self.alphas = 1 - self.noise_schedule

        self.prod_alphas = torch.cumprod(self.alphas, dim=0)

        self.noisenet = None

    ## TODO: think about how to intelligently implement sampling and training
    def forward(self):
        pass

    def sample(self):
        pass


# TODO: implement a U-Net here, preferrably without diffusers.UNet

class UNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = ReLU(self.conv1(x))
        x2 = ReLU(self.conv2(x1))
        x = x + x2
        return x

class UNetCIFAR10(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)


        # TODO: completely fix these dimensions and add pads
        self.enc1 = UNetLayer(3, 8)     # 12x12x8
        self.enc2 = UNetLayer(8, 16)    # 8x8x16
        self.enc3 = UNetLayer(16, 32)   # 4x4x32
        self.enc4 = UNetLayer(32, 64)

        self.enc_mid = UNetLayer(32, 64)

        self.upconv1 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec1 = UNetLayer(32, 16)

        self.upconv2 = ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.dec2 = UNetLayer(32, 16)
        self.dec2 = UNetLayer(16, 8)
        self.dec4 = UNetLayer(8, 64)



if __name__ == "__main__":
    test_cifar10_load()
