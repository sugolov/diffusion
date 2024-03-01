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

        self.noisenet = ResUNetCIFAR10()

    def forward(self):
        pass

    def sample(self):
        pass



class UNetConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x):
        x = ReLU()(self.conv1(x))
        x = ReLU()(self.conv2(x))
        return x

class UNetDecLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.conv = UNetConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.upconv = ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels, kernel_size=2, stride=2
        )

        self.res_conv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, x_res):
        x = self.upconv(x)
        x = x + self.res_conv(x_res)
        x = self.conv(x)
        return x


class ResUNetCIFAR10(nn.Module):
    """
    Hardcoded with CIFAR10 dimensions and fixed padding

    TODO: add residual connections
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxpool = MaxPool2d(kernel_size=2)

        # encoding convolutions
        self.enc1 = UNetConvLayer(3, 16)     # 16x16x16
        self.enc2 = UNetConvLayer(16, 32)    # 32x8x8
        self.enc3 = UNetConvLayer(32, 64)   # 64x4x4

        # middle convolution
        self.midconv = UNetConvLayer(64, 128)

        # decoder up convolutions
        # residual has `out_channels` channels
        self.dec3 = UNetDecLayer(128, 64)
        self.dec2 = UNetDecLayer(64, 32)
        self.dec1 = UNetDecLayer(32, 16)

        # out layers
        self.outconv = UNetConvLayer(16, 8)
        self.outlayer = Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def forward(self, x):

        # encoding steps
        x1 = self.enc1(x)       # 16x32x32
        x2 = self.enc2(self.maxpool(x1))    # 32x16x16
        x3 = self.enc3(self.maxpool(x2))    # 64x8x8

        # middle convolution
        x3d = self.midconv(self.maxpool(x3))

        # decoding steps

        x2d = self.dec3(x3d, x3)
        x1d = self.dec2(x2d, x2)
        xout = self.dec1(x1d, x1)

        return self.outlayer(self.outconv(xout))

if __name__ == "__main__":
    tensor = torch.randn((3, 32, 32))
    unet = ResUNetCIFAR10()
    s = unet(tensor)
    print(s.shape)
