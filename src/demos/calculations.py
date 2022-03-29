from typing import Optional, Tuple

from src.math_utils import MMD_multiscale
from src.model import ModelWrapper
from src import config, utils, math_utils, model, robot_models, forward_kinematics

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim
from tqdm import tqdm


def sample_errors(model_wrapper: ModelWrapper, x: np.array, target_pose: np.array) -> Tuple[np.array, np.array]:
    """Calculate the forward kinematics given joint angles `x` and return the positional and angular errors to the
    end effector's target pose `target_pose`
    """
    n_samples = x.shape[0]
    y_returned = forward_kinematics.klampt_fk(model_wrapper.robot_model, x)

    ang_errs = np.zeros(y_returned.shape[0])

    for i in range(n_samples):
        angular_diff = math_utils.geodesic_distance_between_quaternions(
            target_pose[3:].reshape((1, 4)), y_returned[i, 3:].reshape((1, 4))
        )[0]
        ang_errs[i] = angular_diff

    return np.linalg.norm(y_returned[:, 0:3] - target_pose[0:3], axis=1), ang_errs


def mmd(sols1, sols2, n_terms=1):
    """MMD calculation wrapper

    Copied from https://github.com/JeremySMorgan/toy_inverse-kinematics-updated/blob/master/losses.py#L49
    """
    if "numpy" in str(type(sols1)):
        sols1 = torch.from_numpy(sols1).float().to(config.device)
    if "numpy" in str(type(sols2)):
        sols2 = torch.from_numpy(sols2).float().to(config.device)
    assert sols1.shape == sols2.shape
    assert 0 < n_terms and n_terms <= 3
    assert n_terms == 1, "Use `n_terms` for mmd calculations"

    rev_kernel_width = 1.1827009364464547
    rev_kernel_widths = [rev_kernel_width] * n_terms
    reverse_loss_a = [0.2, 1.0, 2.0]
    reverse_loss_a = reverse_loss_a[0:n_terms]

    return MMD_multiscale(
        sols1,
        sols2,
        rev_kernel_widths,
        reverse_loss_a,
        reduce=True,
    ).item()


def calculate_mmd(
    model_wrapper: ModelWrapper,
    n_poses: int = 100,
    n_samples_per_pose: int = 500,
    latent_noise_distribution: str = "gaussian",
    latent_noise_scale: float = 1,
    n_terms: int = 1,
    poses: Optional[np.array] = None,
    print_results=False,
    use_tqdm=False,
) -> float:
    """Calculate the MMD score between the INN and the ik generated solution distribution."""

    """ 
        - python cli_demo.py --run_name=misunderstood-serenity-7770

        Created model of type 'cinn_softflow' ( w/ 33075676 parameters)
        100%|...| 100/100 [02:05<00:00,  1.25s/it]
        Ave mmd_model:	 0.7342762497067451 	std: 0.3589206949304765
        Ave mmd_random:	 0.10092135727405548 	std: 0.013486457069768008
    """

    fk = lambda sample: model_wrapper.robot_model.forward_kinematics(sample)
    r_sample = lambda _m: model_wrapper.robot_model.sample(_m)
    gen_ik_sols = lambda pose: model_wrapper.robot_model.inverse_kinematics(pose, n_samples_per_pose, seed="random")

    mmds_model = []
    mmds_random = []

    if poses is None:
        iterator = range(n_poses)
    else:
        iterator = poses
    if use_tqdm:
        iterator = tqdm(iterator)

    for i in iterator:

        if poses is None:
            pose = fk(r_sample(1))[0]
        else:
            pose = i

        sols_ik = gen_ik_sols(pose)
        sols_inn, _ = model_wrapper.make_samples(
            pose,
            n_samples_per_pose,
            latent_noise_distribution=latent_noise_distribution,
            latent_noise_scale=latent_noise_scale,
        )
        ik_n_found = sols_ik.shape[0]
        if ik_n_found != n_samples_per_pose:
            print(f"Warning: IK only returned {ik_n_found} solutions ({n_samples_per_pose} requested)")

        sols_inn = sols_inn[0:ik_n_found, :]
        sols_random = r_sample(ik_n_found)

        mmd_model = mmd(sols_ik, sols_inn, n_terms=n_terms)
        mmd_random = mmd(sols_ik, sols_random, n_terms=n_terms)

        mmds_model.append(mmd_model)
        mmds_random.append(mmd_random)

    if print_results:
        print("Ave mmd_model:\t", np.mean(mmds_model), "\tstd:", np.std(mmds_model))
        print("Ave mmd_random:\t", np.mean(mmds_random), "\tstd:", np.std(mmds_random))
    return np.mean(mmds_model), np.std(mmds_model)


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
