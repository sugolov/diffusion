from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import wandb

from model.ddpm import DDPMnet
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_ddpm_cifar10(ddpm_net, data_location, lr=2e-4, epochs=1, batch_size=128, checkpoint_steps=10):

    # CIFAR10 training data
    dataset_train = datasets.CIFAR10(data_location, train=True, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
    training_data_loader = DataLoader(dataset_train, batch_size=int(batch_size), shuffle=True)

    run = wandb.init(name="ddpm_cifar10")

    # tensor device
    ddpm_net = ddpm_net.to(device)

    # optimization
    optimizer = torch.optim.Adam(ddpm_net.parameters(), lr=lr)
    MSE = torch.nn.functional.mse_loss

    # training loop
    for i in range(int(epochs)):
        # timestample
        timestamp("completed epoch " + str(i+1))

        for batch, pred in training_data_loader:

            # sample
            t = torch.randint(ddpm_net.n_steps, (1,)).to(device)
            noise = torch.randn((batch.shape[0], 3, 32, 32)).to(device)

            batch = batch.to(device)

            # forward
            noise_pred = ddpm_net(x0=batch, noise=noise, t=t)
            loss = MSE(noise_pred, noise)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        run.log(
            {
                "epoch": i,
                "loss": loss
            }
        )

        if i % checkpoint_steps == 0:
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": ddpm_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss
                },
                "_".join(["out/ddpm_cifar10", str(i)])
            )

    timestamp("end")


if __name__ == "__main__":
    unet_config = load_config(name="CIFAR10_unet_config", location="../model/config")
    ddpm_net = DDPMnet(unet_config)


