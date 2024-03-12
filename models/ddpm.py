import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, ReLU, Linear, MultiheadAttention
from torch.nn.parameter import Parameter

import diffusers
from diffusers import UNet2DModel

from test.test_ddpm import *

class UNetConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 num_groups=4,
                 residual=True,
                 attention=False,
                 n_head=4,
                 time_channels=1024,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.has_residual = residual
        self.has_attention = attention
        self.n_head = n_head

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels

        # main convs
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # residual
        self.resconv = Conv2d(in_channels, out_channels, kernel_size=1, padding=padding)
        self.gn_res = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # attention
        self.attn = MultiheadAttention(embed_dim=out_channels**2, num_heads=self.n_head)

        # time embedding net
        self.time_net = nn.Linear(time_channels, out_channels)

    def _make_mlp(self, in_channels, out_channels):
        c = max([in_channels, out_channels])
        net = nn.Sequential(
            Linear(in_channels, 2 * c),
            ReLU(),
            Linear(2 * c, 4 * c),
            ReLU(),
            Linear(4 * c, 4 * c),
            ReLU(),
            Linear(4 * c, 2 * c),
            ReLU(),
            Linear(2 * c, out_channels)
        )
        return net


    def forward(self, x, t):
        # get time encoding
        t = self.positional_encoding(t, dim=self.time_channels)
        t = self.time_net(t)

        x_in = x

        x = ReLU()(self.conv1(x))
        x = self.gn1(x)

        x = x + t

        ## TODO: implement this better
        #if self.has_attention:
         #   x = self.attn(x)

        x = ReLU()(self.conv2(x))
        x = self.gn2(x)

        if self.has_residual:
            xres = ReLU()((self.resconv(x_in)))
            xres = self.gn_res(xres)
            return x + xres
        return x

    def positional_encoding(self, t, dim=1024, n=1e5):
        enc = torch.zeros(dim)
        # sine indices
        enc[2 * torch.arange(dim / 2, dtype=torch.int64)] = torch.sin(t / n ** (torch.arange(dim / 2) / dim))
        # cosine indices
        enc[1 + 2 * torch.arange(dim / 2, dtype=torch.int64)] = torch.cos(t / n ** (torch.arange(dim / 2) / dim))
        return enc


class UNetDecLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 num_groups=4,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.conv = UNetConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups
        )

        self.gn_up = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels
        )

        self.up_conv = ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels, kernel_size=2, stride=2
        )

        self.gn_res = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels
        )

        self.res_conv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, x_res):
        x = self.up_conv(x)
        x = self.gn_up(x)
        x = x + self.res_conv(x_res)
        x = self.conv(x)
        x = self.gn_res(x)
        return x


class ResUNetCIFAR10(nn.Module):
    """
    Hardcoded with CIFAR10 dimensions and fixed padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxpool = MaxPool2d(kernel_size=2)

        # encoding convolutions
        self.enc1 = UNetConvLayer(3, 16, num_groups=2)  # 16x16x16
        self.enc2 = UNetConvLayer(16, 32, num_groups=4)  # 32x8x8
        self.enc3 = UNetConvLayer(32, 64, num_groups=8)  # 64x4x4

        # middle convolution
        self.midconv = UNetConvLayer(64, 128)

        # decoder up convolutions
        # residual has `out_channels` channels
        self.dec3 = UNetDecLayer(128, 64, num_groups=8)
        self.dec2 = UNetDecLayer(64, 32, num_groups=4)
        self.dec1 = UNetDecLayer(32, 16, num_groups=2)

        # out layers
        self.outconv = UNetConvLayer(16, 8)
        self.gn_out = nn.GroupNorm(
            num_groups=1,
            num_channels=8
        )
        self.outlayer = Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def forward(self, x, t):
        # encoding steps
        x1 = self.enc1(x)  # 16x32x32
        x2 = self.enc2(self.maxpool(x1))  # 32x16x16
        x3 = self.enc3(self.maxpool(x2))  # 64x8x8

        # middle convolution
        x3d = self.midconv(self.maxpool(x3))

        # decoding steps
        x2d = self.dec3(x3d, x3)
        x1d = self.dec2(x2d, x2)
        xout = self.dec1(x1d, x1)

        return self.outlayer(self.gn_out(self.outconv(xout)))

    def positional_encoding(self, t, dim=1024, n=1e5):
        enc = torch.zeros(dim)
        # sine indices
        enc[2 * torch.arange(dim / 2, dtype=torch.int64)] = torch.sin(t / n ** (torch.arange(dim / 2) / dim))
        # cosine indices
        enc[1 + 2 * torch.arange(dim / 2, dtype=torch.int64)] = torch.cos(t / n ** (torch.arange(dim / 2) / dim))
        return enc


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
    x = torch.randn((2, 3, 32, 32))
    t = 4
    unet = ResUNetCIFAR10()
    s = unet(x, t)
    print(s.shape)
    print(unet.parameters())
