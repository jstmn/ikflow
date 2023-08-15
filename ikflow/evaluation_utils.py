from typing import Tuple, List, Optional

import numpy as np
import torch

from jrl.robot import Robot
from jrl.conversions import geodesic_distance_between_quaternions, enforce_pt_np_input, PT_NP_TYPE
from jrl.config import DEVICE

""" Description of 'SOLUTION_EVALUATION_RESULT_TYPE':
- torch.Tensor: [n] tensor of positional errors of the IK solutions. The error is the L2 norm of the realized poses of
                    the solutions and the target pose.
- torch.Tensor: [n] tensor of angular errors of the IK solutions. The error is the angular geodesic distance between the
                    realized orientation of the robot's end effector from the IK solutions and the targe orientation.
- torch.Tensor: [n] tensor of bools indicating whether each IK solutions has exceeded the robots joint limits.
- torch.Tensor: [n] tensor of bools indicating whether each IK solutions is self colliding.
- float: Runtime
"""
SOLUTION_EVALUATION_RESULT_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]


def _get_target_pose_batch(target_pose: PT_NP_TYPE, n_solutions: int) -> torch.Tensor:
    """Return a batch of target poses. If the target_pose is a single pose, then it will be tiled to match the number of
    solutions. If the target_pose is a batch of poses, then it will be returned as is.

    Args:
        target_pose (PT_NP_TYPE): [7] or [n x 7] the target pose(s) to calculate errors for
        n_solutions (int): The number of solutions we are calculating errors for
    """
    if target_pose.shape[0] == 7:
        if isinstance(target_pose, torch.Tensor):
            return target_pose.repeat(n_solutions, 1)
        return np.tile(target_pose, (n_solutions, 1))
    return target_pose


@enforce_pt_np_input
def pose_errors(
    poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE, acos_epsilon: Optional[float] = None
) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses."""
    assert poses_1.shape == poses_2.shape, f"Poses are of different shape: {poses_1.shape} != {poses_2.shape}"

    if isinstance(poses_1, torch.Tensor):
        l2_errors = torch.norm(poses_1[:, 0:3] - poses_2[:, 0:3], dim=1)
    else:
        l2_errors = np.linalg.norm(poses_1[:, 0:3] - poses_2[:, 0:3], axis=1)
    angular_errors = geodesic_distance_between_quaternions(
        poses_1[:, 3 : 3 + 4], poses_2[:, 3 : 3 + 4], acos_epsilon=acos_epsilon
    )
    assert l2_errors.shape == angular_errors.shape
    return l2_errors, angular_errors


@enforce_pt_np_input
def pose_errors_cm_deg(
    poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE, acos_epsilon: Optional[float] = None
) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses in cm and degrees"""
    assert poses_1.shape == poses_2.shape, f"Poses are of different shape: {poses_1.shape} != {poses_2.shape}"
    l2_errors, angular_errors = pose_errors(poses_1, poses_2, acos_epsilon=acos_epsilon)
    if isinstance(poses_1, torch.Tensor):
        return 100 * l2_errors, torch.rad2deg(angular_errors)
    return 100 * l2_errors, np.rad2deg(angular_errors)


def solution_pose_errors(
    robot: Robot, solutions: torch.Tensor, target_poses: PT_NP_TYPE
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
    assert isinstance(
        target_poses, (np.ndarray, torch.Tensor)
    ), f"target_poses must be a torch.Tensor or np.ndarray (got {type(target_poses)})"
    assert isinstance(solutions, torch.Tensor), f"solutions must be a torch.Tensor (got {type(solutions)})"
    n_solutions = solutions.shape[0]
    if n_solutions >= 1000:
        print("Heads up: It may be faster to run solution_pose_errors() with pytorch directly on the cpu/gpu")

    if isinstance(target_poses, torch.Tensor):
        target_poses = target_poses.detach().cpu().numpy()
    target_poses = _get_target_pose_batch(target_poses, solutions.shape[0])

    ee_pose_ikflow = robot.forward_kinematics(solutions[:, 0 : robot.ndof].detach().cpu().numpy())
    rot_output = ee_pose_ikflow[:, 3:]

    # Positional Error
    l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_poses[:, 0:3], axis=1)
    rot_target = target_poses[:, 3:]
    assert rot_target.shape == rot_output.shape

    # Surprisingly, this is almost always faster to calculate on the gpu than on the cpu. I would expect the opposite
    # for low number of solutions (< 200).
    q_target_pt = torch.tensor(rot_target, device=DEVICE, dtype=torch.float32)
    q_current_pt = torch.tensor(rot_output, device=DEVICE, dtype=torch.float32)
    ang_errors = geodesic_distance_between_quaternions(q_target_pt, q_current_pt).detach().cpu().numpy()
    return l2_errors, ang_errors


def calculate_joint_limits_exceeded(configs: torch.Tensor, joint_limits: List[Tuple[float, float]]) -> torch.Tensor:
    """Calculate if the given configs have exceeded the specified joint limits

    Args:
        configs (torch.Tensor): [batch x ndof] tensor of robot configurations
        joint_limits (List[Tuple[float, float]]): The joint limits for the robot. Should be a list of tuples, where each
                                                    tuple contains (lower, upper).
    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs have exceeded the specified joint limits
    """
    toolarge = configs > torch.tensor([x[1] for x in joint_limits], dtype=torch.float32, device=configs.device)
    toosmall = configs < torch.tensor([x[0] for x in joint_limits], dtype=torch.float32, device=configs.device)
    return torch.logical_or(toolarge, toosmall).any(dim=1)


def calculate_self_collisions(robot: Robot, configs: torch.Tensor) -> torch.Tensor:
    """Calculate if the given configs will cause the robot to collide with itself

    Args:
        robot (Robot): The robot
        configs (torch.Tensor): [batch x ndof] tensor of robot configurations

    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs will self-collide
    """
    is_self_colliding = [robot.config_self_collides(config) for config in configs]
    return torch.tensor(is_self_colliding, dtype=torch.bool)


# TODO: What type should this return? Right now its a mix of torch.Tensor and np.ndarray
def evaluate_solutions(
    robot: Robot, target_poses: PT_NP_TYPE, solutions: torch.Tensor
) -> SOLUTION_EVALUATION_RESULT_TYPE:
    """Return detailed information about a batch of solutions. See the comment at the top of this file for a description
    of the SOLUTION_EVALUATION_RESULT_TYPE type - that's the type that this function returns

    Example usage:
        l2_errors, ang_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
            robot, ee_pose_target, samples
        )
    """
    assert isinstance(
        target_poses, (np.ndarray, torch.Tensor)
    ), f"target_poses must be a torch.Tensor or np.ndarray (got {type(target_poses)})"
    assert isinstance(solutions, torch.Tensor), f"solutions must be a torch.Tensor (got {type(solutions)})"
    target_poses = _get_target_pose_batch(target_poses, solutions.shape[0])
    l2_errors, angular_errors = solution_pose_errors(robot, solutions, target_poses)
    joint_limits_exceeded = calculate_joint_limits_exceeded(solutions, robot.actuated_joints_limits)
    self_collisions_respected = calculate_self_collisions(robot, solutions)
    return l2_errors, angular_errors, joint_limits_exceeded, self_collisions_respected
