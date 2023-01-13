from typing import Tuple, Union

from jkinpylib.robots import Robot
from ikflow.math_utils import rotation_matrix_from_quaternion, geodesic_distance
from ikflow import config

import numpy as np
import torch


def get_solution_errors(
    robot: Robot, solutions: Union[torch.Tensor, np.ndarray], target_pose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the L2 and angular errors of calculated ik solutions for a given target_pose. Note: this function expects
    multiple solutions but only a single target_pose. All of the solutions are assumed to be for the given target_pose

    Args:
        robot (Robot): The Robot which contains the FK function we will use
        solutions (Union[torch.Tensor, np.ndarray]): [n x 7] IK solutions for the given target pose
        target_pose (np.ndarray): [7] the target pose the IK solutions were generated for

    Returns:
        Tuple[np.ndarray, np.ndarray]: The L2, and angular (rad) errors of IK solutions for the given target_pose
    """
    ee_pose_ikflow = robot.forward_kinematics(solutions[:, 0 : robot.n_dofs].cpu().detach().numpy())

    # Positional Error
    l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_pose[0:3], axis=1)

    # Angular Error
    n_solutions = solutions.shape[0]
    rot_target = np.tile(target_pose[3:], (n_solutions, 1))
    rot_output = ee_pose_ikflow[:, 3:]

    output_R9 = rotation_matrix_from_quaternion(torch.tensor(rot_output, device=config.device))
    target_R9 = rotation_matrix_from_quaternion(torch.tensor(rot_target, device=config.device))
    ang_errors = geodesic_distance(target_R9, output_R9).detach().cpu().numpy()
    assert l2_errors.shape == ang_errors.shape
    return l2_errors, ang_errors
