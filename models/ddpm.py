import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, ReLU, Linear, MultiheadAttention
from torch.nn.parameter import Parameter

import diffusers
from diffusers import UNet2DModel

#from test.test_ddpm import *

class MLP(nn.Module):
    def __init__(self, layer_dims, activations=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = []

        if activations is None:
            self.activations = [nn.ReLU() for _ in range(len(layer_dims)-1)]
        else:
            self.activations = activations

        for dim_in, dim_out in zip(layer_dims[:-1], layer_dims[1:],):
            self.layers.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for L, f in zip(self.layers[:-1], self.activations):
            x = f(L(x))

        x = self.layers[-1](x)

        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), kernel_size=2, padding="same", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))

class AttentionConv(nn.Module):
    def __init__(self, out_channels, kernel_size=2, padding="same", embed_dim=32, num_heads=4, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # main convs
        self.in_conv = ConvLayer(out_channels, 1, kernel_size=kernel_size, padding=padding)
        self.out_conv = ConvLayer(1, out_channels, kernel_size=kernel_size, padding=padding)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        x_in = x
        x = self.in_conv(x)

        # drop the channel dim during attention computation
        x = x[:, 0, :, :]
        x = self.attn(x, x, x)[0]
        x = x[:, None, :, :]

        x = self.out_conv(x)
        return x + x_in


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 num_groups=4,
                 mlp_layers=(1024,),
                 attention=False,
                 embed_dim=16,
                 num_heads=4,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # main convs
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # group norm
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.time_net = MLP(layer_dims=mlp_layers + (self.out_channels,))

        if attention:
            self.attn = AttentionConv(out_channels=out_channels, kernel_size=kernel_size, embed_dim=embed_dim, num_heads=num_heads)
        else:
            self.attn = nn.Identity()


    def forward(self, x, t=None):
        if t is not None:
            t = self.time_net(t)

        x = self.conv1(x)
        #x = self.gn1(x)

        if t is not None:
            x = x + t[:, None, None]

        x = self.attn(x)

        x = self.conv2(x)
        #x = self.gn2(x)

        return x

class UNetEncoderLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 num_groups=4,
                 residual=True,
                 mlp_layers=(1024,),
                 attention=False,
                 embed_dim=16,
                 num_heads=4,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.residual = residual

        # double conv
        self.conv = UNetConvBlock(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  num_groups=num_groups,
                                  mlp_layers=mlp_layers,
                                  attention=attention,
                                  embed_dim=embed_dim,
                                  num_heads=num_heads)
        # residual
        if self.residual:
            self.res_conv = Conv2d(in_channels, out_channels, kernel_size=1, padding=padding)
        else:
            self.res_conv = None

        self.gn_res = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x, t):
        x_in = x

        x = self.conv(x, t)

        if self.residual:
            x_res = nn.SiLU()(self.res_conv(x_in))
            #xres = self.gn_res(xres)
            return x + x_res
        return x

class UNetDecoderLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding="same",
                 num_groups=4,
                 mlp_layers=(1024,),
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.conv = UNetConvBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            mlp_layers=mlp_layers
        )

        #self.gn_up = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.up_conv = ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels, kernel_size=2, stride=2
        )

        #self.gn_res = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.res_conv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, x_res, t):
        x = self.up_conv(x)
        #x = self.gn_up(x)
        x = x + self.res_conv(x_res)

        x = self.conv(x, t)
        #x = self.gn_res(x)
        return x


class ResUNetCIFAR10(nn.Module):
    """
    Hardcoded with CIFAR10 dimensions and fixed padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxpool = MaxPool2d(kernel_size=2)

        # encoding convolutions
        self.enc1 = UNetEncoderLayer(3, 16, num_groups=2)  # 16x16x16
        self.enc2 = UNetEncoderLayer(16, 32, num_groups=4, attention=True, embed_dim=16)  # 32x8x8
        self.enc3 = UNetEncoderLayer(32, 64, num_groups=8)  # 64x4x4

        # middle convolution
        self.midconv = UNetConvBlock(64, 128)

        # decoder up convolutions
        # residual has `out_channels` channels
        self.dec3 = UNetDecoderLayer(128, 64, num_groups=8)
        self.dec2 = UNetDecoderLayer(64, 32, num_groups=4)
        self.dec1 = UNetDecoderLayer(32, 16, num_groups=2)

        # out layers
        self.outconv = UNetConvBlock(16, 8)
        self.gn_out = nn.GroupNorm(
            num_groups=1,
            num_channels=8
        )
        self.outlayer = Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def forward(self, x, t):
        t = self.positional_encoding(t)
        # encoding steps
        x1 = self.enc1(x, t)  # 16x32x32
        x2 = self.enc2(self.maxpool(x1), t)  # 32x16x16
        x3 = self.enc3(self.maxpool(x2), t)  # 64x8x8

        # middle convolution
        x3d = self.midconv(self.maxpool(x3))

        # decoding steps
        x2d = self.dec3(x3d, x3, t)
        x1d = self.dec2(x2d, x2, t)
        xout = self.dec1(x1d, x1, t)

        #return self.outlayer(self.gn_out(self.outconv(xout)))
        return self.outlayer(self.outconv(xout))

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
    x = torch.randn((128, 3, 32, 32))
    t = 4
    unet = ResUNetCIFAR10()
    s = unet(x, t)
    print(s.shape)
