import sys
import argparse
from utils.utils import *
from model.ddpm import DDPMnet
from train.train_ddpm import train_ddpm_cifar10

parser = argparse.ArgumentParser(description="slurm training arg parser")
parser.add_argument("--cifar10")
args = parser.parse_args()

if args.cifar10:
    train_config = load_config("CIFAR10_train_config", "train/config")
    unet_config = load_config("CIFAR10_unet_config", "model/config")

    ddpm = DDPMnet(unet_config, **train_config["ddpm"])
    # training
    # TODO: add checkpointing loading if exists
    train_ddpm_cifar10(ddpm, data_location="data", **train_config["train"])

