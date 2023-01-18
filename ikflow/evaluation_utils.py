from typing import Tuple, Union, List
from time import time

from jkinpylib.robot import Robot
from jkinpylib.robots import Panda
from jkinpylib.conversions import (
    quaternion_to_rotation_matrix,
    geodesic_distance_between_rotation_matrices,
    PT_NP_TYPE,
    DEFAULT_TORCH_DTYPE,
)
from ikflow import config

import numpy as np
import torch

DEVICE_TEMP = "cuda"


def get_solution_errors(
    robot: Robot, solutions: torch.Tensor, target_pose: PT_NP_TYPE
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
    assert isinstance(solutions, torch.Tensor)
    is_multi_y = len(target_pose.shape) == 2

    if isinstance(target_pose, torch.Tensor):
        target_pose = target_pose.detach().cpu().numpy()

    if solutions.shape[0] >= 1000:
        print("Heads up: It may be faster to run get_solution_errors() with pytorch directly on the cpu/gpu")

    ee_pose_ikflow = robot.forward_kinematics(solutions[:, 0 : robot.n_dofs].detach().cpu().numpy())

    print("ee_pose_ikflow:", ee_pose_ikflow)

    # Positional Error
    if is_multi_y:
        l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_pose[:, 0:3], axis=1)
    else:
        l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_pose[0:3], axis=1)

    # Angular Error
    n_solutions = solutions.shape[0]
    if is_multi_y:
        rot_target = target_pose[:, 3:]
    else:
        rot_target = np.tile(target_pose[3:], (n_solutions, 1))
    rot_output = ee_pose_ikflow[:, 3:]

    # TODO: 'rot_target: [[1. 0. 0. 0.]]' from tests/evaluation_utils_test.py is wrong.
    print("rot_target:", rot_target)

    # Surprisingly, this is almost always faster to calculate on the gpu than on the cpu. I would expect the opposite
    # for low number of solutions (< 200).
    output_R9 = quaternion_to_rotation_matrix(torch.tensor(rot_output, device=config.device, dtype=torch.float32))
    target_R9 = quaternion_to_rotation_matrix(torch.tensor(rot_target, device=config.device, dtype=torch.float32))
    ang_errors = geodesic_distance_between_rotation_matrices(target_R9, output_R9).detach().cpu().numpy()
    assert l2_errors.shape == ang_errors.shape
    return l2_errors, ang_errors


def calculate_joint_limits_respected(configs: torch.Tensor, joint_limits: List[Tuple[float, float]]) -> torch.Tensor:
    """Calculate if the given configs are within the specified joint limits

    Args:
        configs (torch.Tensor): [batch x n_dofs] tensor of robot configurations
        joint_limits (List[Tuple[float, float]]): The joint limits for the robot. Should be a list of tuples, where each
                                                    tuple contains (lower, upper).
    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs are within the specified joint limits
    """
    return torch.zeros(configs.shape[0], dtype=torch.bool)


def calculate_self_collisions_respected(robot: Robot, configs: torch.Tensor) -> torch.Tensor:
    """Calculate if the given configs will cause the robot to collide with itself

    Args:
        robot (Robot): The robot
        configs (torch.Tensor): [batch x n_dofs] tensor of robot configurations

    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs will self-collide
    """
    return torch.zeros(configs.shape[0], dtype=torch.bool)


""" Benchmarking get_solution_errors():

python ikflow/evaluation_utils.py

"""

if __name__ == "__main__":
    robot = Panda()

    print("\n=====================")
    print("Implementation: numpy")
    print("\nNumber of solutions | Time to calculate")
    print("------------------- | -----------------")
    for n_solutions in range(0, 2000, 100):
        solutions_np = np.random.randn(n_solutions, 7)
        solutions_pt = torch.tensor(solutions_np, device=DEVICE_TEMP, dtype=torch.float32)
        target_pose = robot.forward_kinematics(robot.sample_joint_angles(1))[0]

        runtimes = []
        for _ in range(5):
            t0 = time()
            l2_errors, ang_errors = get_solution_errors(robot, solutions_pt, target_pose)
            runtimes.append(time() - t0)
        runtime = float(np.mean(runtimes))
        print(f"{n_solutions}  \t| {round(1000*(runtime), 5)}ms")

        # Benchmarking result: it is faster to convert solutions to numpy to calculate errors when n_solutions < 1000.
        # after that, it becomes faster to calculate errors with pytorch on the CPU. At some point longer in the future
        # it will be faster to calculate errors with pytorch on the GPU.

        # Current implementation:
        # =====================
        # Implementation: numpy

        # Number of solutions | Time to calculate
        # ------------------- | -----------------
        # 0	    | 0.52063ms
        # 100	| 2.02473ms
        # 200	| 4.13291ms
        # 300	| 4.01187ms
        # 400	| 5.37546ms
        # 500	| 5.68899ms
        # 600	| 6.61222ms
        # 700	| 7.88768ms
        # 800	| 8.47348ms
        # 900	| 9.6999ms
        # 1000	| 10.7499ms
        # 1100	| 11.24612ms
        # 1200	| 12.73076ms
        # 1300	| 13.11556ms
        # 1400	| 15.02244ms
        # 1500	| 15.03865ms
        # 1600	| 16.01609ms
        # 1700	| 16.87686ms
        # 1800	| 17.74081ms
        # 1900	| 19.45353ms
