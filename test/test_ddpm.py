import cifar10

from PIL import Image

import torch
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d

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
    #tensor = Conv2d(in_channels=3, out_channels=8, padding=1, kernel_size=2)(tensor)
    #tensor = MaxPool2d(kernel_size=2)(tensor)
    print(tensor.shape)


