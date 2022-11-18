"""Global configuration"""
import torch
import os

# /
device = "cuda:0" if torch.cuda.is_available() else "cpu"

DEFAULT_DATA_DIR = "~/.cache/ikflow/"

#
WANDB_CACHE_DIR = os.path.join(DEFAULT_DATA_DIR, "wandb/")
DATASET_DIR = os.path.join(DEFAULT_DATA_DIR, "datasets/")
TRAINING_LOGS_DIR = os.path.join(DEFAULT_DATA_DIR, "training_logs/")
MODELS_DIR = os.path.join(DEFAULT_DATA_DIR, "models/")

print(f"MODELS_DIR: {MODELS_DIR}")
