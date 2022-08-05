"""Global configuration"""
import torch

# /
device = "cuda:0" if torch.cuda.is_available() else "cpu"

URDFS_DIRECTORY = "./urdfs"
WANDB_CACHE_DIR = "./wandb/"
DATASET_DIR = "./datasets/"