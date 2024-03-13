from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch

from model.ddpm import DDPMnet
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#training_data_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

def train_cifar10(lr=2e-4, epochs=1, batch_size=64):

    # CIFAR10 training data
    dataset_train = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
    training_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # noise network
    ddpmnet = DDPMnet()

    # halving tensor size
    ddpmnet = ddpmnet.to(device)
    ddpmnet.half()

    # optimization
    optimizer = torch.optim.Adam(ddpmnet.parameters(), lr=lr)
    MSE = torch.nn.functional.mse_loss

    # training loop
    for i in range(epochs):

        # timestample
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

    timestamp("end")

# TODO: setup checkpointing, https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

if __name__ == "__main__":
    train_cifar10(batch_size=128)