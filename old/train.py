import sys
import argparse
from utils.utils import *
from model.ddpm import DDPMnet
from train.train_ddpm import train_ddpm_cifar10_old

import torch

parser = argparse.ArgumentParser(description="slurm training arg parser")
parser.add_argument("--cifar10", action="store_true", default=False)
parser.add_argument("--checkpoint", action="store", type=int)


args = parser.parse_args()

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
    
    train_ddpm_cifar10_old(ddpm,
                           data_location="data",
                           model_state_dict=model_state_dict,
                           optimizer_state_dict=optimizer_state_dict,
                           save_epoch=save_epoch,
                           **train_config["train"]
                           )
