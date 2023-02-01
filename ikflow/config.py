"""Global configuration"""
import torch
import os

# /
device = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_TORCH_DTYPE = torch.float32

# ~/.cache/ikflow/
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache/ikflow/")

#
WANDB_CACHE_DIR = os.path.join(DEFAULT_DATA_DIR, "wandb/")
DATASET_DIR = os.path.join(DEFAULT_DATA_DIR, "datasets/")
TRAINING_LOGS_DIR = os.path.join(DEFAULT_DATA_DIR, "training_logs/")
MODELS_DIR = os.path.join(DEFAULT_DATA_DIR, "models/")

# Dataset tags
DATASET_TAG_NON_SELF_COLLIDING = "non-self-colliding"

ALL_DATASET_TAGS = [DATASET_TAG_NON_SELF_COLLIDING]


# Training
# This is the absolute maximum value that a value on the joint angle side of the network can take on when it is passed
# through the network, when sigmoid activation is enabled. Note this only applies for values that are in columns past
# the columns reserved for the joints. On other terms, this only applies for the columns with indexes greater than
# n_dofs.
SIGMOID_SCALING_ABS_MAX = 1.0
