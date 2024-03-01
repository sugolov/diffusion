
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.ddpm import DDPMnet
from utils.utils import timestamp

from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))


def train_cifar10(lr=2e-4, epochs=1, batch_size=64):

    dataset_train = datasets.CIFAR10(
        '../data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )


    training_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    ddpmnet = DDPMnet()

    # halving tensor size
    ddpmnet = ddpmnet.to(device)
    ddpmnet.half()

    #for layer in ddpmnet.modules():
    #    if isinstance(layer, torch.nn.GroupNorm):
    #        layer.half()


    optimizer = torch.optim.Adam(ddpmnet.parameters(), lr=lr)
    MSE = torch.nn.functional.mse_loss

    for i in range(epochs):

        timestamp(i+1)

        for batch, pred in training_data_loader:

            # sample
            t = torch.randint(ddpmnet.n_steps, (1,)).to(device)
            noise = torch.randn((batch.shape[0], 3, 32, 32)).to(device).half()

            batch = batch.to(device).half()

            # forward
            noise_pred = ddpmnet(x0=batch, noise=noise, t=t)
            loss = MSE(noise_pred, noise)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("end: " + str(current_time))


if __name__ == "__main__":
    train_cifar10(batch_size=128)




