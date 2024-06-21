from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import wandb

from model.ddpm import DDPMnet
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_ddpm_cifar10(ddpm_net, data_location, lr=2e-4, n_epochs=1, batch_size=128, checkpoint_steps=10, save_epoch=None, optimizer_state_dict=None, model_state_dict=None):

    n_epochs = int(n_epochs)

    # CIFAR10 training data
    dataset_train = datasets.CIFAR10(data_location, train=True, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
    training_data_loader = DataLoader(dataset_train, batch_size=int(batch_size), shuffle=True)

    run = wandb.init(name="ddpm_cifar10")

    # tensor device
    ddpm_net = ddpm_net.to(device)

    if model_state_dict is not None:
        ddpm_net.load_state_dict(model_state_dict)

    # optimization
    optimizer = torch.optim.Adam(ddpm_net.parameters(), lr=lr)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    MSE = torch.nn.functional.mse_loss
    
    if save_epoch is None:
        epochs = range(n_epochs)
    else:
        save_epoch = int(save_epoch)
        epochs = range(save_epoch + 1, save_epoch + n_epochs)

    # training loop
    for e in epochs:
        # timestample
        timestamp("epoch " + str(e))
        epoch_loss = []


        for batch, pred in training_data_loader:

            batch /= 255

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

            # log
            epoch_loss.append(loss.detach())

        run.log(
            {
                "epoch": e,
                "loss": torch.mean(torch.tensor(epoch_loss))
            }
        )

        if e % checkpoint_steps == 0:
            file_name = "_".join(["out/ddpm_cifar10", str(e)])
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": ddpm_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss
                },
                "_".join(["out/ddpm_cifar10", str(e)])
            )
            print("saved in " + file_name)


    timestamp("end")


if __name__ == "__main__":
    unet_config = load_config(name="CIFAR10_unet_config", location="../model/config")
    ddpm_net = DDPMnet(unet_config)


