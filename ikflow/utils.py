from typing import Tuple, Optional, Callable, List
import pathlib
import os
import random
import pkg_resources

import numpy as np
import torch

from ikflow import config


def get_wandb_project() -> Tuple[str, str]:
    """Get the wandb entity and project. Reads from environment variables"""

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    assert (
        wandb_project is not None
    ), "The 'WANDB_PROJECT' environment variable is not set (try `export WANDB_PROJECT=<your wandb project name>`)"
    assert (
        wandb_entity is not None
    ), "The 'WANDB_ENTITY' environment variable is not set (try `export WANDB_PROJECT=<your wandb project name>`)"
    return wandb_entity, wandb_project


def get_dataset_directory(robot: str) -> str:
    """Return the path of the directory"""
    return os.path.join(config.DATASET_DIR, robot)


def get_dataset_filepaths(dataset_directory: str, tags: List[str]):
    """Return the filepaths of the tensors in a dataset"""

    def filename_w_tags(filename: str):
        for i, tag in enumerate(tags):
            filename = filename + f"__tag{i}={tag}"
        return filename

    samples_tr_file_path = os.path.join(dataset_directory, filename_w_tags("samples_tr.pt"))
    poses_tr_file_path = os.path.join(dataset_directory, filename_w_tags("endpoints_tr.pt"))
    samples_te_file_path = os.path.join(dataset_directory, filename_w_tags("samples_te.pt"))
    poses_te_file_path = os.path.join(dataset_directory, filename_w_tags("endpoints_te.pt"))
    info_filepath = os.path.join(dataset_directory, filename_w_tags("info.txt"))
    return samples_tr_file_path, poses_tr_file_path, samples_te_file_path, poses_te_file_path, info_filepath


def get_filepath(local_filepath: str):
    return pkg_resources.resource_filename(__name__, local_filepath)


# _____________
# Pytorch utils


def assert_joint_angle_tensor_in_joint_limits(
    joints_limits: List[Tuple[float, float]], x: torch.Tensor, description: str, eps: float
):
    """Validate that a tensor of joint angles is within the joint limits of the robot."""
    for i, (lower, upper) in enumerate(joints_limits):
        max_elem = torch.max(x[:, i]).item()
        min_elem = torch.min(x[:, i]).item()
        error_lower = min_elem - (lower - eps)
        error_upper = max_elem - (upper + eps)
        assert min_elem >= lower - eps, (
            f"[{description}] Joint angle {min_elem} is less than lower limit {lower} (minus eps={eps}) for joint {i} -"
            f" error = {error_lower}\n limits(joint_{i}) = ({lower}, {upper})"
        )
        assert max_elem <= upper + eps, (
            f"[{description}] Max element {max_elem} is greater than upper limit {upper} (plus eps={eps}) for joint"
            f" {i} - error = {error_upper}\n  limits(joint_{i}) = ({lower}, {upper})"
        )


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(0)
    print("set_seed() - random int: ", torch.randint(0, 1000, (1, 1)).item())


def cuda_info():
    """Printout the current cuda status"""
    cuda_available = torch.cuda.is_available()
    print(f"\n____________\ncuda_info()")
    print(f"cuda_available: {cuda_available}")

    if cuda_available:
        print(f"  current_device: {torch.cuda.current_device()}")
        print(f"  device(0): {torch.cuda.device(0)}")
        print(f"  device_count: {torch.cuda.device_count()}")
        print(f"  get_device_name(0): {torch.cuda.get_device_name(0)}")
    print()


# __________________
# Printing functions


def print_tensor_stats(
    arr,
    name="",
    writable: Optional[
        Callable[
            [
                str,
            ],
            None,
        ]
    ] = None,
):
    if writable is None:
        writable = lambda _s: None

    round_amt = 4

    s = f"\n\t\tmin,\tmax,\tmean,\tstd  - for '{name}'"
    print(s)
    writable.write(s + "\n")

    for i in range(arr.shape[1]):
        if "torch" in str(type(arr)):
            min_ = round(torch.min(arr[:, i]).item(), round_amt)
            max_ = round(torch.max(arr[:, i]).item(), round_amt)
            mean = round(torch.mean(arr[:, i]).item(), round_amt)
            std = round(torch.std(arr[:, i]).item(), round_amt)
        else:
            min_ = round(np.min(arr[:, i]), round_amt)
            max_ = round(np.max(arr[:, i]), round_amt)
            mean = round(np.mean(arr[:, i]), round_amt)
            std = round(np.std(arr[:, i]), round_amt)
        s = f"  col_{i}:\t{min_}\t{max_}\t{mean}\t{std}"
        print(s)
        writable.write(s + "\n")


def get_sum_joint_limit_range(samples):
    """Return the total joint limit range"""
    sum_joint_range = 0
    for joint_i in range(samples.shape[1]):
        min_sample = torch.min(samples[:, joint_i])
        max_sample = torch.max(samples[:, joint_i])
        sum_joint_range += max_sample - min_sample
    return sum_joint_range


# ___________________
# Scripting functions


def boolean_string(s):
    if isinstance(s, bool):
        return s
    if s.upper() not in {"FALSE", "TRUE"}:
        raise ValueError(f'input: "{s}" ("{type(s)}") is not a valid boolean string')
    return s.upper() == "TRUE"


def non_private_dict(d):
    r = {}
    for k, v in d.items():
        if k[0] == "_":
            continue
        r[k] = v
    return r


# _____________________
# File system utilities


def safe_mkdir(dir_name: str):
    """Create a directory `dir_name`. May include multiple levels of new directories"""
    pathlib.Path(dir_name).mkdir(exist_ok=True, parents=True)


# ______________
# Training utils


def grad_stats(params_trainable) -> Tuple[float, float, float]:
    """
    Return the average and max. gradient from the parameters in params_trainable
    """
    ave_grads = []
    abs_ave_grads = []
    max_grad = 0.0
    for p in params_trainable:
        if p.grad is not None:
            ave_grads.append(p.grad.mean().item())
            abs_ave_grads.append(p.grad.abs().mean().item())
            max_grad = max(max_grad, p.grad.data.max().item())
    return np.average(ave_grads), np.average(abs_ave_grads), max_grad
