
class UNetCIFAR10(nn.Module):
    """
    Hardcoded UNet with CIFAR10 dimensions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxpool = MaxPool2d(kernel_size=2)

        # encoding convolutions
        self.enc1 = UNetEncoderLayer(3, 16, num_groups=1)  # 16x16x16
        self.enc2 = UNetEncoderLayer(16, 32, num_groups=1, attention=True, embed_dim=16)  # 32x8x8
        self.enc3 = UNetEncoderLayer(32, 64, num_groups=1)  # 64x4x4

        # middle convolution
        self.midconv = UNetConvBlock(64, 128)

        # decoder up convolutions
        # residual has `out_channels` channels
        self.dec3 = UNetDecoderLayer(128, 64, num_groups=1)
        self.dec2 = UNetDecoderLayer(64, 32, num_groups=1, attention=True, embed_dim=16)
        self.dec1 = UNetDecoderLayer(32, 16, num_groups=1)

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

        # return self.outlayer(self.gn_out(self.outconv(xout)))
        return self.outlayer(self.outconv(xout))

    def positional_encoding(self, t, dim=1024, n=1e5):
        enc = torch.zeros(dim)
        # sine indices
        enc[2 * torch.arange(dim / 2, dtype=torch.int64)] = torch.sin(t / n ** (torch.arange(dim / 2) / dim))
        # cosine indices
        enc[1 + 2 * torch.arange(dim / 2, dtype=torch.int64)] = torch.cos(t / n ** (torch.arange(dim / 2) / dim))
        return enc.to("cuda")

    class UNetDecoderLayer(nn.Module):

        def __init__(self,
                     in_channels,
                     out_channels,
                     kernel_size=2,
                     upsample_size=2,
                     padding="same",
                     num_groups=1,
                     mlp_layers=(1024,),
                     attention=False,
                     num_heads=4,
                     embed_dim=16,
                     *args,
                     **kwargs
                     ):
            super().__init__(*args, **kwargs)

            self.conv = UNetConvBlock(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups,
                mlp_layers=mlp_layers,
                attention=attention,
                num_heads=num_heads,
                embed_dim=embed_dim
            )

            self.up_conv = ConvTranspose2d(in_channels=2 * in_channels, out_channels=out_channels,
                                           kernel_size=upsample_size, stride=upsample_size)
            self.res_conv = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding)

            self.use_attn = attention

            if self.use_attn:
                self.attn = AttentionConv(out_channels=out_channels, kernel_size=kernel_size, embed_dim=embed_dim,
                                          num_heads=num_heads, num_groups=num_groups)

            self.gn_up = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.gn_res = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        def forward(self, x, x_res, t):
            x = self.up_conv(x)
            x = self.gn_up(x)

            # TODO: should be stacked
            x = torch.cat((x, self.res_conv(x_res)), dim=-3)

            x = self.conv(x, t)
            x = self.gn_res(x)
            return x