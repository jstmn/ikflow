from typing import Tuple, Union, Optional, Callable, Dict
import csv
import pathlib
import os
import random
from time import time
import pickle

from src.robots import RobotModel, get_robot
import config
from src.ik_solvers import IkflowSolver, GenerativeIKSolver
from src.training_parameters import IkflowModelParameters

import numpy as np
import pandas as pd
import torch
import wandb
from wandb.wandb_run import Run

# _______________________
# Dataset utilities
def get_dataset_directory(robot: str):
    """Return the path of the directory"""
    return os.path.join(config.DATASET_DIR, robot)


# _______________________
# Model loading utilities


def get_ik_solver(
    model_weights_filepath: str, robot_name: str, model_hyperparameters: Dict
) -> Tuple[GenerativeIKSolver, IkflowModelParameters]:
    """Build and return a `IkflowSolver` using the model weights saved in the file `model_weights_filepath` for the
    given robot and with the given hyperparameters

    Args:
        model_weights_filepath (str): The filepath for the model weights
        robot_name (str): The name of the robot that the model is for
        model_hyperparameters (Dict): The hyperparameters used for the NN

    Returns:
        Tuple[GenerativeIKSolver, IkflowModelParameters]: A `IkflowSolver` solver and the corresponding
                                                            `IkflowModelParameters` parameters object
    """
    assert os.path.isfile(
        model_weights_filepath
    ), f"File '{model_weights_filepath}' was not found. Unable to load model weights"
    robot_model = get_robot(robot_name)

    # Build GenerativeIKSolver and set weights
    hyper_parameters = IkflowModelParameters()
    hyper_parameters.__dict__.update(model_hyperparameters)
    ik_solver = IkflowSolver(hyper_parameters, robot_model)
    ik_solver.load_state_dict(model_weights_filepath)
    return ik_solver, hyper_parameters


# _____________
# Pytorch utils


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


def cuda_mem_info():
    """Print out the current memory usage on the gpu"""

    print(f"\n____________\ncuda_mem_info()")

    if not torch.cuda.is_available():
        print("Cuda unavailable, returning")

    # Total memory of the gpu (MB)
    t = torch.cuda.get_device_properties(0).total_memory
    t = t / (1024 * 1024)

    # Returns the current GPU memory managed by the caching allocator in bytes for the given device
    r = torch.cuda.memory_reserved(0)
    r = r / (1024 * 1024)  # convert to MB

    # Returns the current GPU memory occupied by tensors in bytes for a given device
    a = torch.cuda.memory_allocated(0)
    a = a / (1024 * 1024)  # convert to MB
    f = r - a  # free inside reserved

    print(f"  Total device memory (MB):                      {t}")
    print(f"  Total reserved memory (MB):                    {r}")
    print(f"  Total allocated memory (MB):                   {a}")
    print(f"  Total free memory (reserved - allocated) (MB): {f}")
    print()


# __________________
# Printing functions


def pp_dict(d, space=4):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{' ' * space}{k}")
            pp_dict(v, space=space + 3)
        # elif isinstance(v, pd.DataFrame):
        #     # s = " " * 5 + d.to_string().replace("\n", "\n     ")
        #     print(f"{' ' * space}{k}\n{s}")
        else:
            print(f"{' ' * space}{k}\t{v}")


def print_sd(sd, name=""):
    print(f"\nprint_sd(name = '{name}')")
    for k, v in sd.items():
        print(" ", k, "\t", v.shape)


def print_np_array_copyable(x: np.array):
    """Prints a numpy array so that it can be copy pasted as python code"""
    print("[", end="")
    for i in range(x.shape[0]):
        print("[", end="")
        for j in range(x.shape[1] - 1):
            print(f"{x[i, j]}, ", end="")
        print(f"{x[i, x.shape[1]-1]}]", end="")
        if i < x.shape[0] - 1:
            print(",")
            continue
        print("]")


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


def decimal_range(start, stop, inc):
    while start < stop:
        yield start
        start += inc


def float_range(start, stop, inc):
    x = start
    while x < stop:
        yield x
        x += inc


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
    print(f"safe_mkdir(): Creating directory '{dir_name}'")
    pathlib.Path(dir_name).mkdir(exist_ok=True, parents=True)


def os_is_posix() -> bool:
    return os.name == "posix"


def delete_files_in_dir(dir_name: str):
    """Delete all files in directory"""
    for f in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, f))


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


def get_learning_rate(optimizer) -> float:
    """
    Get the learning rate used by the optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
