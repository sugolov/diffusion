import cifar10

from PIL import Image

def test_cifar10_load():
    """
    Tries loading and saving image from CIFAR10

    :return: None
    """
    generator = cifar10.data_batch_generator()
    image, label = next(generator)

    print(image.shape)
    im = Image.fromarray(image)
    im.save("../test/out/cifar10_test.jpg")



