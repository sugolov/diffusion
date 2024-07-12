from diffusers import DDPMPipeline
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import wandb

from model.ddpm import DDPMnet
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

import torch.nn.functional as F

def train_loop_ddpm(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, checkpoint_path=None):

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        # project_dir=os.path.join(config.output_dir, "logs"),
        project_dir=config.output_dir
    )


    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers(
            project_name="diffusion",
            init_kwargs={"wandb": {"name": config.run_name}},
            config=config
        )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if checkpoint_path is not None:
        accelerator.load_state(checkpoint_path)

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:

            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:

                if config.push_to_hub:
                    upload_folder(
                        repo_id=config.repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

                else:
                    pipeline.save_pretrained(config.output_dir)

    accelerator.end_training()

def train_ddpm_cifar10_old(
        ddpm_net,
        data_location,
        lr=2e-4,
        n_epochs=1,
        batch_size=128,
        checkpoint_steps=10,
        save_epoch=None,
        optimizer_state_dict=None,
        model_state_dict=None
):

    n_epochs = int(n_epochs)

    # CIFAR10 training data
    dataset_train = datasets.CIFAR10(data_location, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

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


