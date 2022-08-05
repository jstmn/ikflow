from typing import Tuple, Union, Optional, Callable
import csv
import pathlib
import os
import random
from time import time
import pickle

from utils.robots import RobotModel, get_robot
import config
from utils.models import IkflowModel, ModelWrapper
from training.training_parameters import IkflowModelParameters, MultiCINN_Parameters

import numpy as np
import pandas as pd
import torch
import wandb
from wandb.wandb_run import Run

MODEL_DOWNLOAD_DIR = "~/Projects/data/ikflow/saved_models"

# _______________________
# Dataset utilities


def get_dataset_directory(robot: str):
    """Return the path of the directory"""
    return os.path.join(config.DATASET_DIR, robot)


# _______________________
# Model loading utilities


def load_model_from_wandb(
    run_name: str, wandb_run: Optional[Run]
) -> Tuple[str, Union[IkflowModelParameters, MultiCINN_Parameters]]:
    """Querry wandb for the saved model from the specified run. Return the robot, and hyperparameters used for the model."""

    # Load Run
    if wandb_run is None:
        wandb_run = wandb.init(project="nnik", tags=["SAFE_TO_DELETE", "FROM_DEMO_PY"])

    # artifact_name = f"{run_name}:v0"
    artifact_name = f"{run_name}:latest"
    artifact = wandb_run.use_artifact(artifact_name)
    run = artifact.logged_by()
    robot = run.config["robot"]
    print(f"Run config: {run.config}")

    # Download saved model
    t0 = time()
    run_download_dir = os.path.join(MODEL_DOWNLOAD_DIR, run_name)
    artifact.download(root=run_download_dir)
    print(f"Downloaded nn state_dict for '{run_name}' in {round(time() - t0, 3)} seconds")

    # Set hyperparameters. Note: this will set the training values aswell. They'll be ignored so it doesn't matter
    assert run.config["model_type"] in ["ikflow", "multi_cinn"]
    if run.config["model_type"] == "ikflow":
        hyper_parameters = IkflowModelParameters()
    elif run.config["model_type"] == "multi_cinn":
        hyper_parameters = MultiCINN_Parameters()

    for k, v in run.config.items():
        hyper_parameters.__dict__[k] = v

    # Hack: Save run config so don't need to querry wandb in future runs
    run_config = {"hyper_parameters": hyper_parameters, "robot": robot}
    with open(os.path.join(run_download_dir, "run_config.pkl"), "wb") as f:
        pickle.dump(run_config, f)

    return robot, hyper_parameters


def get_model_wrapper(run_name, wandb_run: Optional[Run] = None) -> Tuple[ModelWrapper, IkflowModelParameters]:
    """Load a robot"""
    model_dir = os.path.join(MODEL_DOWNLOAD_DIR, run_name)
    model_filepath = os.path.join(model_dir, "model.pkl")
    run_config_filepath = os.path.join(model_dir, "run_config.pkl")

    if os.path.isfile(model_filepath) and os.path.isfile(run_config_filepath):
        print(f"Model found on disk, skipping download")
        with open(run_config_filepath, "rb") as f:
            cfg = pickle.load(f)
            robot = cfg["robot"]
            hyper_parameters = cfg["hyper_parameters"]
    else:
        robot, hyper_parameters = load_model_from_wandb(run_name, wandb_run=wandb_run)

    robot_model = get_robot(robot)

    # Build ModelWrapper and set weights
    if isinstance(hyper_parameters, IkflowModelParameters):
        model_wrapper = IkflowModel(hyper_parameters, robot_model)
    elif isinstance(hyper_parameters, MultiCINN_Parameters):
        model_wrapper = ModelWrapper_MultiCINN(hyper_parameters, robot_model)
    else:
        raise RuntimeError("Unknown model")

    # torch.load(model_filepath, map_location="cpu")
    # model_wrapper.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')))
    model_wrapper.load_state_dict(model_filepath)

    return model_wrapper, hyper_parameters


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
