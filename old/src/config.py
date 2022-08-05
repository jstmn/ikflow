"""Global configuration for the experiments"""
import torch
import os

# /
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

git_branch = "master"
# print(f"device: {device}")

URDFS_DIRECTORY = "./urdfs"

is_ubuntu_dev_machine = os.path.isdir("/home/jeremysmorgan/")
WANDB_CACHE_DIR = "./wandb/"
