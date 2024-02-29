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

class UNetConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, padding="same", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x1 = ReLU()(self.conv1(x))
        x2 = ReLU()(self.conv2(x1))
        return x2

class UNetEncLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, padding="same", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = UNetConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        return self.pool(self.conv(x))


class UNetCIFAR10(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = MaxPool2d(kernel_size=2)

        self.enc1 = UNetConvLayer(3, 16)     # 16x16x16
        self.enc2 = UNetConvLayer(16, 32)    # 32x8x8
        self.enc3 = UNetConvLayer(32, 64)   # 64x4x4

        self.upconv3 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec3 = UNetConvLayer(64, 32)

        self.upconv2 = ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec2 = UNetConvLayer(32, 16)

        self.upconv1 = ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.dec1 = UNetConvLayer(16, 8)

        self.outconv = Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def forward(self, x):

        # 16x16x16
        x1 = self.maxpool(self.enc1(x))
        print(x1.shape)
        # 32x8x8
        x2 = self.maxpool(self.enc2(x1))
        print(x2.shape)
        # 64x4x4
        x3 = self.maxpool(self.enc3(x2))
        print(x3.shape)

        # stack x2 on up-convolution of x3
        x2d = self.upconv3(x3)

        x1d = self.dec3(torch.cat((x2, x2d), dim=0))
        #x1d = self.dec2(torch.stack(x1, self.upconv2(x2d)))
        #out = self.dec1(torch.stack(x1, self.upconv2(x2d)))






if __name__ == "__main__":
    # test_cifar10_load()
    # test_conv_shapes()
    tensor = torch.randn((3,32,32))
    unet = UNetCIFAR10()
    unet(tensor)
