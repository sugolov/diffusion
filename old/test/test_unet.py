import cifar10

from PIL import Image

import torch
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d
from model.unet import *


def test_cifar10_load():
    """
    Tries loading and saving image from CIFAR10

    :return: None
    """
    generator = cifar10.data_batch_generator()
    image, label = next(generator)

    print(image.shape)
    im = Image.fromarray(image)
    im.save("test/out/cifar10_test.jpg")


def test_conv_shapes():
    tensor = torch.randn((2, 32, 32))
    tensor = ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=2, stride=2)(tensor)
    # tensor = Conv2d(in_channels=3, out_channels=8, padding=1, kernel_size=2)(tensor)
    # tensor = MaxPool2d(kernel_size=2)(tensor)
    print(tensor.shape)


def test_unet():
    x = torch.randn((2, 3, 32, 32))
    t = 4
    unet = UNetCIFAR10()
    s = unet(x, t)
    print(s.shape)
    print(unet.parameters())


def test_mlp():
    tensor = torch.randn((3,))
    mlp = MLP(layer_dims=[3, 64, 64, 1])
    print(mlp(tensor))


def test_attn_conv():
    print("implementation-like transpositions")
    tensor = torch.tensor(range(96)).reshape((2, 3, 4, 4))
    print(tensor.shape)
    tensor = torch.flatten(tensor, start_dim=-2)
    print(tensor.shape)
    tensor = torch.transpose(tensor, dim0=-1, dim1=-2)
    print(tensor.shape)
    tensor = torch.transpose(tensor, dim0=-1, dim1=-2)
    print(tensor.shape)
    tensor = tensor.reshape((2, 3, 4, 4))
    print(tensor.shape)

    tensor = torch.randn((5, 3, 8, 8))
    conv = AttentionConv(3, embed_dim=8, num_groups=1)
    print(conv(tensor).shape)


def test_kwargs():
    config = {
        "out_channels": 3,
        "kernel_size": 2,
        "padding": "same",
        "embed_dim": 32,
        "num_heads": 4
    }

    conv = AttentionConv(**config)


def test_unet():
    net = UNet(
        layer_channels=[3, 8, 16, 32, 64],
        layer_attention=[False, True, False, False],
        layer_groups=(1,1,1,1)
    ).to("cuda")
    image = net(x=torch.randn((5, 3, 32, 32)).to("cuda"), t=4)
    print(image.shape)


if __name__ == "__main__":
    #test_attn_conv()
    # test_kwargs()
    test_unet()
