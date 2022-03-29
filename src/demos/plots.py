from typing import Tuple, List
from time import time
import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())
print(os.getcwd())

from src.robot_models import get_robot
from src.utils import boolean_string
from src import robot_models
from src.model import ModelWrapper, ModelWrapper_cINN_Softflow
from src.training_parameters import cINN_SoftflowParameters

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


IEEE_COLUMN_WIDTH = 3.5  # inches
GOLDEN_RATIO = 1.61803398875


def plot_sum_interval_width_vs_accuracy():
    """Plot the sum of the robot's joint limit ranges vs. the lowest L2 error or the robot"""

    errors = {
        "atlas": [0.002782, 0.008908],
        "atlas_arm": [0.00343, 0.01105],
        "kuka_11dof": [0.083, 0.322],
        "panda_arm": [0.009, 0.051],
        "robonaut2": [0.00537, 0.01643],
        "robonaut2_arm": [0.002146, 0.007498],
        "robosimian": [0.02864, 0.1298],
        "ur5": [0.025, 0.17],
        "valkyrie": [0.003844, 0.01283],
        "valkrie_arm": [0.0004094, 0.003361],
        "valkrie_arm_shoulder": [0.00167, 0.00592],
    }

    sum_joint_intervals = []
    lowest_l2_errors = []
    lowest_angular_errors = []
    labels = []

    for robot in robot_models.get_all_3d_robots():
        sum_joint_range = 0
        for l, u in robot.joint_limits:
            diff = u - l
            assert diff > 0
            sum_joint_range += diff
        sum_joint_intervals.append(sum_joint_range)
        lowest_l2_errors.append(errors[robot.name][0])
        lowest_angular_errors.append(errors[robot.name][1])
        labels.append(robot.name)

    # plt.scatter(sum_joint_intervals, lowest_l2_errors)
    plt.scatter(sum_joint_intervals, lowest_angular_errors)

    for i, txt in enumerate(labels):
        # plt.annotate(txt, (sum_joint_intervals[i], lowest_l2_errors[i]))
        plt.annotate(txt, (sum_joint_intervals[i], lowest_angular_errors[i]))

    plt.xlabel("Sum of the joint limit range")
    plt.ylabel("Lowest L2 Error")
    plt.show()


def _mean_std_runtime(get_runtime_fn, k) -> Tuple[float, float]:
    runtimes = []
    for _ in range(k):
        runtimes.append(get_runtime_fn())
    return np.mean(runtimes), np.std(runtimes)


def ikflow_runtime_mean_stds(
    model_wrapper: ModelWrapper, batch_sizes: List[int], k: int, end_pose: np.array
) -> List[Tuple[float, float]]:
    """Return the mean and std of ikflow's runtime for each batch size in `batch_sizes`"""
    means = []
    stds = []
    for batch_size in tqdm(batch_sizes):
        runtime_fn_i = lambda: model_wrapper.make_samples(end_pose, batch_size)[1]
        mean, std = _mean_std_runtime(runtime_fn_i, k)
        means.append(mean)
        stds.append(std)
    return np.array(means), np.array(stds)


def klampt_ik_runtime_mean_stds(
    model_wrapper: ModelWrapper, batch_sizes: List[int], k: int, interpolate_from_first=False, seed_w_ikf=False
) -> Tuple[np.array, np.array]:
    """Return the mean and std of Klampts ik solve runtime for each batch size in `batch_sizes`. Call this function in
    torch.inference_mode()"""

    if interpolate_from_first:
        lb = "\n__________\n__________\n__________"
        print(lb, "WARNING: `interpolate_from_first` enabled", lb)

    means = []
    stds = []
    for batch_size in tqdm(batch_sizes):

        # Randomly drawn poses are held fixed for a given batch_size
        r_samples = model_wrapper.robot_model.sample(batch_size)
        end_poses = model_wrapper.robot_model.forward_kinematics(r_samples)

        def runtime_fn_i():
            """Returns the total amount of time to generate `batch_size` solutions for `batch_size` randomly drawn
            poses
            """

            # initialize seeding
            seeds = None
            seed_runtime = 0
            if seed_w_ikf:
                seeds, seed_runtime = model_wrapper.make_samples_mult_y(end_poses, latent_noise_scale=0.75)

            t_sum = 0
            for i in range(batch_size):

                # Set seed, if seeding is enabled
                seed = None
                if seeds is not None:
                    seed = np.expand_dims(seeds[i].cpu().numpy(), axis=0)

                t0 = time()
                _ = model_wrapper.robot_model.inverse_kinematics(end_poses[i], 1, seed=seed)
                t_sum += time() - t0

            return t_sum + seed_runtime

        mean, std = _mean_std_runtime(runtime_fn_i, k)
        means.append(mean)
        stds.append(std)

        if interpolate_from_first:
            time_p_batch = mean / batch_size
            for bs in batch_sizes[1:]:
                means.append(bs * time_p_batch)
                stds.append(std)
            break
    return np.array(means), np.array(stds)
