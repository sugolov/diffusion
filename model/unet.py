import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d

from model.mlp import MLP


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), num_groups=1, kernel_size=2, padding="same",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.activation(x)
        return x


class AttentionConv(nn.Module):
    def __init__(self, out_channels, kernel_size=2, padding="same", embed_dim=64, num_heads=4, num_groups=1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # main convs
        self.in_conv = ConvLayer(out_channels, embed_dim, kernel_size=kernel_size, padding=padding,
                                 num_groups=num_groups)
        self.out_conv = ConvLayer(embed_dim, out_channels, kernel_size=kernel_size, padding=padding,
                                  num_groups=num_groups)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        x_in = x
        x = self.in_conv(x)

        b, c, d1, d2 = x.shape

        """
        idea: attend to channels c at each pixel of d1xd2
        """

        # make shape b, c, d1*d2
        x = torch.flatten(x, start_dim=-2)

        # make shape b, d1*d2, c
        # ie. attend to c channels at d1*d2 pixels
        x = torch.transpose(x, dim0=-1, dim1=-2)

        x = self.attn(x, x, x)[0]

        # make shape b, c, d1*d2
        # ie. make channels come first again
        x = torch.transpose(x, dim0=-1, dim1=-2)

        # make shape b, c, d1, d2
        x = x.reshape((b, c, d1, d2))

        x = self.out_conv(x)

        # residual connection
        x = x + x_in
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, padding="same", num_groups=1, mlp_layers=(1024,),
                 attention=False, embed_dim=16, num_heads=4, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # main convs
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                               num_groups=num_groups)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                               num_groups=num_groups)

        self.time_net = MLP(layer_dims=mlp_layers + (self.out_channels,))
        self.use_attn = attention

        if self.use_attn:
            self.attn = AttentionConv(out_channels=out_channels, kernel_size=kernel_size, embed_dim=embed_dim,
                                      num_heads=num_heads, num_groups=num_groups)
        # else:
        #    self.attn = nn.Identity()

    def forward(self, x, t=None):
        if t is not None:
            t = self.time_net(t)

        x = self.conv1(x)

        if t is not None:
            x = x + t[:, None, None]

        if self.use_attn:
            x = x + self.attn(x)

        x = self.conv2(x)

        return x


class UNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, padding="same", num_groups=1, residual=True,
                 mlp_layers=(1024,), attention=False, embed_dim=16, num_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.residual = residual

        # double conv
        self.conv_block = UNetConvBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups,
            mlp_layers=mlp_layers,
            attention=attention,
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        # residual
        if self.residual:
            self.res_conv = Conv2d(in_channels, out_channels, kernel_size=1, padding=padding)
        else:
            self.res_conv = None

        self.gn_res = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x, t):
        x_in = x

        x = self.conv_block(x, t)

        if self.residual:
            x_res = nn.SiLU()(self.res_conv(x_in))
            x_res = self.gn_res(x_res)
            x = x + x_res

        return x


class UNet(nn.Module):
    """
    UNet that can be set up with a single config
    """

    def __init__(self,
                 layer_channels,
                 layer_attention,
                 layer_groups,
                 maxpool_size=2,
                 time_emb_dim=1024,
                 time_n=1e5,
                 kernel_size=2,
                 upsample_size=2,
                 padding="same",
                 residual=True,
                 mlp_layers=None,
                 num_heads=4,
                 embed_dim=64,
                 *args, **kwargs):
        """
        Set up a UNet for DDPM with any layer config

        :param layer_channels: `in_channels' for subsequent encoder blocks
        :param layer_attention: whether encoder/decoder has attention, corresponding to `layer_channels'
        :param kwargs: params for sublayers
        """
        super().__init__(*args, **kwargs)

        self.time_emb_dim = time_emb_dim
        self.time_n = time_n

        self.time_net = MLP(layer_dims=(self.time_emb_dim, time_emb_dim,))

        self.maxpool = MaxPool2d(kernel_size=maxpool_size)

        self.encoders = nn.ModuleList([])
        self.upsamplers = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        # first dimension of MLP should just be time_emb_dim
        if mlp_layers is None:
            mlp_layers = (time_emb_dim,)

        for in_channels, out_channels, attn, num_groups in zip(
                layer_channels[:-1], layer_channels[1:], layer_attention, layer_groups
        ):
            self.encoders.append(
                UNetLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    attention=attn,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    mlp_layers=mlp_layers,
                    residual=residual,
                    num_groups=num_groups
                )
            )

        self.midconv = UNetConvBlock(
            in_channels=layer_channels[-1],
            out_channels=2*layer_channels[-1],
            attention=True
        )

        for in_channels, out_channels, attention, num_groups in zip(
                reversed(layer_channels[1:]), reversed(layer_channels[1:]), reversed(layer_attention),
                reversed(layer_groups)
        ):
            self.upsamplers.append(
                ConvTranspose2d(
                    in_channels=in_channels + out_channels,
                    out_channels=out_channels,
                    kernel_size=upsample_size,
                    stride=upsample_size)
            )
            self.decoders.append(
                UNetLayer(
                    in_channels=in_channels + out_channels,
                    out_channels=out_channels,
                    attention=attention,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    mlp_layers=mlp_layers,
                    num_groups=num_groups
                )
            )
        self.out_layer = nn.Conv2d(layer_channels[1], layer_channels[0], kernel_size=1)

    def forward(self, x, t):
        t = self.positional_encoding(t, self.time_emb_dim, self.time_n)
        t = self.time_net(t)

        x = self.encoders[0](x, t)
        x_enc = [x]

        for encoder in self.encoders[1:]:
            x = encoder(self.maxpool(x), t)
            x_enc.append(x)

        x_dec = self.midconv(self.maxpool(x))

        for decoder, upsampler, x_res in zip(self.decoders, self.upsamplers, reversed(x_enc)):
            x_dec = upsampler(x_dec)
            x_dec = decoder(torch.cat((x_res, x_dec), dim=-3), t)


        x_out = self.out_layer(x_dec)
        return x_out

    @staticmethod
    def positional_encoding(t, dim=128, n=1e5):
        enc = torch.zeros(dim, device="cuda")
        # sine indices
        enc[2 * torch.arange(dim / 2, dtype=torch.int64, device="cuda")] = torch.sin(
            t / n ** (torch.arange(dim / 2, device="cuda") / dim))
        # cosine indices
        enc[1 + 2 * torch.arange(dim / 2, dtype=torch.int64, device="cuda")] = torch.cos(
            t / n ** (torch.arange(dim / 2, device="cuda") / dim))

        return enc