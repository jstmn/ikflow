"""Global configuration"""
import torch
import os
from time import sleep


def _get_device() -> str:
    if not torch.cuda.is_available():
        return "cpu", -1

    def _mem_and_utilitization_ave_usage_pct(device_idx: int):
        device = torch.cuda.device(device_idx)
        mems = []
        utils = []
        for i in range(2):
            mems.append(torch.cuda.memory_usage(device=device))
            utils.append(torch.cuda.utilization(device=device))
            sleep(0.5)
        return sum(mems) / len(mems), sum(utils) / len(utils)

    n_devices = torch.cuda.device_count()
    min_mem = 100
    min_util = 100
    for i in range(n_devices):
        mem_pct, util_pct = _mem_and_utilitization_ave_usage_pct(i)
        min_mem = min(min_mem, mem_pct)
        min_util = min(min_util, util_pct)
        if mem_pct < 5.0 and util_pct < 7.5:
            return f"cuda:{i}", i
    raise EnvironmentError(f"No unused GPU's available. Minimum memory, utilization: {min_mem, min_util}%")


# /
device, GPU_IDX = _get_device()
DEFAULT_TORCH_DTYPE = torch.float32
print(f"config.py: Using device '{device}'")

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
