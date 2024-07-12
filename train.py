import sys
import argparse

from diffusers import DDPMScheduler
from torchvision import datasets, transforms

from utils.utils import *
from model.ddpm import DDPMnet
from model.config import *

from train.train_ddpm import train_loop_ddpm
import torch

from dataclasses import dataclass, asdict

from datasets import load_dataset
from diffusers.optimization import get_cosine_schedule_with_warmup

from train.config import *

#===============================

parser = argparse.ArgumentParser(description="slurm training arg parser")

parser.add_argument("--data", action="store", type=str)

#parser.add_argument("--cifar10", action="store_true", default=False)
#parser.add_argument("--butterfly", action="store_true", default=False)

parser.add_argument("--checkpoint", action="store", type=int)
parser.add_argument("--out", action="store", type=str, default="out")

args = parser.parse_args()

#===============================

match args.data:
    case "butterfly":
        config = SmithsonianButterflyTrainingConfig()

        config.dataset_name = "huggan/smithsonian_butterflies_subset"
        dataset = load_dataset(config.dataset_name, split="train")

        config.output_dir = args.out

        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset = load_dataset(config.dataset_name, split="train")
        dataset.set_transform(transform)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

        model = unet_smithsonian_butterfly(config)

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        train_loop_ddpm(
            config=config,
            train_dataloader=train_dataloader,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )

    case "cifar10":

        config = SmithsonianButterflyTrainingConfig()

        config.dataset_name = "uoft-cs/cifar10"
        dataset = load_dataset(config.dataset_name, split="train")

        config.output_dir = args.out

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["img"]]
            return {"images": images}


        dataset = load_dataset(config.dataset_name, split="train")
        dataset.set_transform(transform)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

        model = unet_smithsonian_butterfly(config)

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        train_loop_ddpm(
            config=config,
            train_dataloader=train_dataloader,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )




#===============================
"""
if args.checkpoint is not None:
    print("works")
    checkpoint = torch.load("_".join(["out/ddpm_cifar10", str(args.checkpoint)]))
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    save_epoch = checkpoint["epoch"]
else:
    model_state_dict, optimizer_state_dict, save_epoch = None, None, None

if args.cifar10:
    train_config = load_config("CIFAR10_train_config", "train/config")
    unet_config = load_config("CIFAR10_unet_config", "model/config")

    ddpm = DDPMnet(unet_config, **train_config["ddpm"])
    # training

    train_loop_cifar10(
        config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler
    )
"""