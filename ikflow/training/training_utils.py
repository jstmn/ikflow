from datetime import datetime
from time import time
import os

import numpy as np
import torch
import wandb

from ikflow import config


def get_softflow_noise(x: torch.Tensor, softflow_noise_scale: float):
    """
    Return noise and noise magnitude for softflow. See https://arxiv.org/abs/2006.04604
    Return:
        c (torch.Tensor): An (batch_sz x 1) array storing the magnitude of the sampled gausian `eps` along each row
        eps (torch.Tensor): a (batch_sz x dim_x) array, where v[i] is a sample drawn from a zero mean gaussian with variance = c[i]**2
    """
    dim_x = x.shape[1]

    # (batch_size x 1) from uniform(0, 1)
    c = torch.rand_like(x[:, 0]).unsqueeze(1)

    # (batch_size x dim_y) *  (batch_size x dim_y)  | element wise multiplication
    eps = torch.randn_like(x) * torch.cat(dim_x * [c], dim=1) * softflow_noise_scale
    return c, eps


def _np_to_str(x: np.ndarray) -> str:
    assert x.ndim == 1
    return str([round(x_i, 4) for x_i in x.tolist()])


def _datetime_str() -> str:
    now = datetime.now()
    return now.strftime("%b.%d.%Y_%I:%-M:%p")


def get_checkpoint_dir(robot_name: str) -> str:
    if wandb.run is not None:
        ckpt_dir_name = f"{robot_name}--{_datetime_str()}--wandb-run-name:{wandb.run.name}/"
    else:
        ckpt_dir_name = f"{robot_name}--{_datetime_str()}/"
    dir_filepath = os.path.join(config.TRAINING_LOGS_DIR, ckpt_dir_name)
    return dir_filepath


# def log_ik_solutions_to_wandb_table(self, k: int = 3):
#     """Log `k` solutions for each of the target poses in ROBOT_TARGET_POSE_POINTS_FOR_PLOTTING to wandb"""
#     if wandb.run is None:
#         print("wandb not initialized, skipping logging solution table")
#         return

#     # TODO:
#     # see https://github.com/wandb/wandb/issues/1826
#     #
#     robot = self.ik_solver.robot
#     if robot.name not in ROBOT_TARGET_POSE_POINTS_FOR_PLOTTING:
#         return

#     target_poses = ROBOT_TARGET_POSE_POINTS_FOR_PLOTTING[robot.name]
#     joint_types = robot.actuated_joint_types

#     for i, target_pose in enumerate(target_poses):
#         solutions = self.generate_ik_solutions(target_pose, k)  # (k, n_dofs)
#         realized_poses = robot.forward_kinematics(solutions.cpu().detach().numpy())
#         errors_xyz = realized_poses[:, 0:3] - target_pose[0:3]
#         q_target = np.tile(target_pose[3:7], (k, 1))
#         q_realized = realized_poses[:, 3:7]
#         # see https://gamedev.stackexchange.com/a/68184
#         q_error = quaternion_product(q_target, quaternion_inverse(q_realized))
#         errors = np.hstack([errors_xyz, q_error])
#         assert errors.shape == (k, 7)
#         for j in range(k):
#             ik_solution_table.add_data(
#                 int(self.global_step),
#                 _np_to_str(target_pose),
#                 _np_to_str(solutions[j]),
#                 _np_to_str(realized_poses[j]),
#                 _np_to_str(errors[j]),
#                 str(joint_types),
#             )

#     new_table = wandb.Table(data=ik_solution_table.data, columns=_IK_SOLUTION_TABLE_COLUMNS)
#     wandb.log({f"{self.ik_solver.robot.name}_solution_table": new_table, "global_step": float(self.global_step)})
