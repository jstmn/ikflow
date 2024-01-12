from time import time
from typing import Callable
import argparse

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from jrl.utils import set_seed
from jrl.math_utils import geodesic_distance_between_quaternions

from ikflow.model_loading import get_ik_solver
from ikflow.evaluation_utils import solution_pose_errors
from ikflow.ikflow_solver import IKFlowSolver

# torch.set_printoptions(linewidth=200, precision=5, sci_mode=False)
torch.set_printoptions(linewidth=200, precision=6, sci_mode=False)

set_seed()


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


POS_ERROR_THRESHOLD = 0.001
ROT_ERROR_THRESHOLD = 0.01


def solve_ikflow(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):
    ikflow_solver.solve_n_poses(ys=target_poses)


def solve_klampt(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):
    ikflow_solver.solve_n_poses(ys=target_poses)
    for i in range(len(target_poses)):
        q_i = ikflow_solver.robot.inverse_kinematics_klampt(target_poses[i])
        assert q_i is not None


def check_converged(robot, qs, target_poses, i):
    # check error
    pose_realized = robot.forward_kinematics_batch(qs)
    pos_errors = torch.norm(pose_realized[:, 0:3] - target_poses[:, 0:3], dim=1)
    rot_errors = geodesic_distance_between_quaternions(target_poses[:, 3:], pose_realized[:, 3:])

    #
    # converged = pos_errors.max().item() < POS_ERROR_THRESHOLD and rot_errors.max().item() < ROT_ERROR_THRESHOLD
    # print("solve_lma", i, converged, round(pos_errors.max().item(), 6), "\t", round(rot_errors.max().item(), 6), "\t", pos_errors.data)
    #
    converged = pos_errors.mean().item() < POS_ERROR_THRESHOLD and rot_errors.mean().item() < ROT_ERROR_THRESHOLD
    print(
        "solve_lma",
        i,
        converged,
        round(pos_errors.mean().item(), 6),
        "\t",
        round(rot_errors.mean().item(), 6),
        "\t",
        pos_errors.data,
        rot_errors.data,
    )

    return converged, pos_errors.cpu().numpy(), rot_errors.cpu().numpy()


def plot_errors(pos_errors, rot_errors, i):

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Errors at iteration {i}")
    axl.set_title(f"Positional errors")
    axl.grid(alpha=0.2)
    axr.set_title(f"Rotational errors")
    axr.grid(alpha=0.2)
    axl.scatter(list(range(len(pos_errors))), pos_errors, label="position")
    axl.plot([0, len(pos_errors)], [POS_ERROR_THRESHOLD, POS_ERROR_THRESHOLD], label="threshold", color="green")
    axr.scatter(list(range(len(pos_errors))), rot_errors, label="rotation")
    axr.plot([0, len(pos_errors)], [ROT_ERROR_THRESHOLD, ROT_ERROR_THRESHOLD], label="threshold", color="green")
    plt.savefig(f"lma_errors_{i}.pdf", bbox_inches="tight")
    plt.show()


def solve_lma(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):

    qs = ikflow_solver.solve_n_poses(ys=target_poses)
    robot = ikflow_solver.robot
    converged = False
    i = 0
    print()
    _, pos_errors, rot_errors = check_converged(robot, qs, target_poses, "init")
    plot_errors(pos_errors, rot_errors, "init")

    while not converged:

        qs = robot.inverse_kinematics_single_step_levenburg_marquardt(target_poses, qs)
        converged, _, _ = check_converged(robot, qs, target_poses, i)

        # print(i, converged)
        i += 1
        assert i < 20, f"Failed to converge after {i} iterations"


def solve_j_pinv(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):

    qs = ikflow_solver.solve_n_poses(ys=target_poses)

    converged = False
    i = 0
    while not converged:
        qs = ikflow_solver.robot.inverse_kinematics_single_step_batch_pt(target_poses, qs, alpha=0.25)
        pos_errors, rot_errors = solution_pose_errors(ikflow_solver.robot, qs, target_poses)
        converged = np.all(pos_errors < POS_ERROR_THRESHOLD) and np.all(rot_errors < ROT_ERROR_THRESHOLD)
        print("solve_j_pinv", i, converged, pos_errors.max().item(), rot_errors.max().item())
        i += 1


""" Example 

python scripts/benchmark_runtime.py --model_name=panda__full__lp191_5.25m

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the saved model (see ikflow/model_descriptions.yaml)"
    )
    parser.add_argument("--k", type=int, default=3, help="Number of repeats per batch size")
    args = parser.parse_args()

    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (ms)", "runtime std", "runtime per solution (ms)"]
    )
    method_names = [
        "ikflow",
        "ikflow + klampt ik",
        "ikflow + levenberg-marquardt",
        # "ikflow + J psuedo-inverse",
    ]

    ikflow_solver, _ = get_ik_solver(args.model_name)

    for batch_size in [1, 5, 10, 50, 100]:
        # for batch_size in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        # for batch_size in [1, 5, 10]:
        print(f"\n---\nBatch size: {batch_size}")

        _, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(
            batch_size, only_non_self_colliding=True, tqdm_enabled=False
        )
        target_poses_pt = torch.tensor(target_poses, dtype=torch.float32, device="cuda:0")

        lambdas = [
            lambda: solve_ikflow(ikflow_solver, target_poses_pt),
            lambda: solve_klampt(ikflow_solver, target_poses),
            lambda: solve_lma(ikflow_solver, target_poses_pt),
            # lambda: solve_j_pinv(ikflow_solver, target_poses_pt),
        ]
        for lambda_, method_name in zip(lambdas, method_names):
            mean_runtime_ms, std_runtime = fn_mean_std(lambda_, args.k)
            new_row = [method_name, batch_size, mean_runtime_ms, std_runtime, mean_runtime_ms / batch_size]
            df.loc[len(df)] = new_row

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)

    # Plot
    max_runtime = -1
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.grid(alpha=0.2)
    for method_name in method_names:
        df_method = df[df["method"] == method_name]
        n_solutions = df_method["number of solutions"]
        total_runtime_ms = df_method["total runtime (ms)"]
        std = df_method["runtime std"]
        ax.plot(n_solutions, total_runtime_ms, label=method_name)
        ax.fill_between(n_solutions, total_runtime_ms - std, total_runtime_ms + std, alpha=0.2)
        max_runtime = max(max_runtime, total_runtime_ms.to_numpy()[-1])

    ax.set_ylim(-0.1, max_runtime + 0.5)
    ax.set_title("Number of solutions vs runtime for geodesic_distance_between_quaternions()")
    ax.set_xlabel("Number of solutions")
    ax.set_ylabel("Total runtime (ms)")
    ax.legend()
    fig.savefig("rotational_distance_runtime_comparison.pdf", bbox_inches="tight")
    plt.show()
