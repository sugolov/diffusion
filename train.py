import sys
import argparse

from model.config import *

from train.ddpm import train_loop_ddpm
import torch

from datasets import load_dataset
from diffusers.optimization import get_cosine_schedule_with_warmup

from train.prepare import prepare
from train.ddpm import train_loop_ddpm


parser = argparse.ArgumentParser(description="slurm training arg parser")
parser.add_argument("--run", action="store", type=str)
parser.add_argument("--out", action="store", type=str, default="out")
parser.add_argument("--checkpoint", action="store", type=int)
parser.add_argument("--n_subset", default=None, action="store", type=int)
args = parser.parse_args()

# prepare and run train loop
config, train_dataloader, model, noise_scheduler, optimizer, lr_scheduler = prepare(
    args.run,
    args.n_subset
)
train_loop_ddpm(
            config=config,
            train_dataloader=train_dataloader,
            model=model,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
