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
    ikflow_solver.generate_ik_solutions(target_poses)

def solve_klampt(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):
    n_failed = 0
    for i in range(len(target_poses)):
        q_i = ikflow_solver.robot.inverse_kinematics_klampt(target_poses[i])
        if q_i is None:
            n_failed += 1

def solve_klampt_ikflow_seed(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):
    qs = ikflow_solver.generate_ik_solutions(target_poses).cpu().numpy()
    target_poses = target_poses.cpu().numpy()
    n_failed = 0
    for i in range(len(target_poses)):
        q_i = ikflow_solver.robot.inverse_kinematics_klampt(target_poses[i], seed=qs[i, :])
        if q_i is None:
            n_failed += 1

def solve_lma(ikflow_solver: IKFlowSolver, target_poses: torch.Tensor):
    ikflow_solver.generate_exact_ik_solutions(target_poses, run_lma_on_cpu=True)


""" Example 

python scripts/benchmark_runtime.py --model_name=panda__full__lp191_5.25m

TODO: report success pct as well. klampt / lma are going to have a different % exact solutions generated
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
        "ikflow - NOT EXACT",
        "klampt",
        "ikflow with levenberg-marquardt",
        "klampt with ikflow seeds"
    ]

    ikflow_solver, _ = get_ik_solver(args.model_name)

    for batch_size in [2, 5, 10, 50, 100, 500, 1000, 2500]:
        print(f"\n---\nBatch size: {batch_size}")

        _, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(
            batch_size, only_non_self_colliding=True, tqdm_enabled=False
        )
        target_poses = torch.tensor(target_poses, dtype=torch.float32, device="cuda:0")

        lambdas = [
            lambda: solve_ikflow(ikflow_solver, target_poses.clone()),
            lambda: solve_klampt(ikflow_solver, target_poses.cpu().numpy()),
            lambda: solve_lma(ikflow_solver, target_poses.clone()),
            lambda: solve_klampt_ikflow_seed(ikflow_solver, target_poses.clone()),
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
    fig.savefig("ik_benchmark_runtime.pdf", bbox_inches="tight")
    plt.show()
