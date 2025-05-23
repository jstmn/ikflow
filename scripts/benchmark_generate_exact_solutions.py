from typing import Callable, Optional
from argparse import ArgumentParser
from time import time
import pickle

import torch
import numpy as np
from jrl.utils import set_seed, evenly_spaced_colors
from jrl.robots import Panda
import matplotlib.pyplot as plt

from ikflow.model import TINY_MODEL_PARAMS
from ikflow.ikflow_solver import IKFlowSolver
from ikflow.model_loading import get_ik_solver

set_seed()

POS_ERROR_THRESHOLD = 0.001
ROT_ERROR_THRESHOLD = 0.01


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


def benchmark_get_pose_error():
    """Benchmarks running IKFlowSolver._calculate_pose_error() on the CPU vs. the GPU"""

    robot = Panda()
    ikflow_solver = IKFlowSolver(TINY_MODEL_PARAMS, robot)

    retry_k = 5
    batch_sizes = [1, 10, 100, 500, 750, 1000, 2000, 3000, 4000, 5000]
    ts_cpu = []
    ts_cu = []
    stds_cpu = []
    stds_cu = []

    print("Number of solutions, CPU, CUDA")
    for n_solutions in batch_sizes:
        qs, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(n_solutions)
        qs_cpu = torch.tensor(qs, device="cpu", dtype=torch.float32)
        qs_cu = torch.tensor(qs, device="cuda:0", dtype=torch.float32)
        target_poses_cpu = torch.tensor(target_poses, device="cpu", dtype=torch.float32)
        target_poses_cu = torch.tensor(target_poses, device="cuda:0", dtype=torch.float32)

        t_cpu, t_cpu_std = fn_mean_std(lambda: ikflow_solver._calculate_pose_error(qs_cpu, target_poses_cpu), retry_k)
        t_cu, t_cu_std = fn_mean_std(lambda: ikflow_solver._calculate_pose_error(qs_cu, target_poses_cu), retry_k)
        ts_cpu.append(t_cpu)
        ts_cu.append(t_cu)
        stds_cpu.append(t_cpu_std)
        stds_cu.append(t_cu_std)
        print(n_solutions, "\t", round(t_cpu, 6), "\t", round(t_cu, 6))

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle("Levenberg-Marquardt IK Convergence")

    ax.set_title("Runtime")
    ax.grid(alpha=0.2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("runtime (s)")

    colors = evenly_spaced_colors(3)
    ts_cpu = np.array(ts_cpu)
    ts_cu = np.array(ts_cu)
    stds_cpu = np.array(stds_cpu)
    stds_cu = np.array(stds_cu)

    ax.plot(batch_sizes, ts_cpu, label="cpu", color=colors[0])
    ax.fill_between(batch_sizes, ts_cpu - stds_cpu, ts_cpu + stds_cpu, alpha=0.15, color=colors[0])
    ax.scatter(batch_sizes, ts_cpu, s=15, color=colors[0])

    ax.plot(batch_sizes, ts_cu, label="cuda", color=colors[1])
    ax.fill_between(batch_sizes, ts_cu - stds_cu, ts_cu + stds_cu, alpha=0.15, color=colors[1])
    ax.scatter(batch_sizes, ts_cu, s=15, color=colors[1])

    ax.legend()
    plt.savefig("get_pose_error_runtime_curve.pdf", bbox_inches="tight")
    plt.show()


def get_stats(batch_sizes, k_retry, ikflow_solver, device, repeat_counts, run_lma_on_cpu):
    runtimes = []
    success_pcts = []
    runtime_stds = []
    success_pct_stds = []
    n_found = []
    n_found_stds = []

    for batch_size in batch_sizes:

        sub_runtimes = []
        sub_success_pcts = []
        sub_n_found = []

        for _ in range(k_retry):
            _, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(
                batch_size, only_non_self_colliding=True
            )
            target_poses = torch.tensor(target_poses, device=device, dtype=torch.float32)
            t0 = time()
            solutions, valid_solution_idxs = ikflow_solver.generate_exact_ik_solutions(
                target_poses,
                pos_error_threshold=POS_ERROR_THRESHOLD,
                rot_error_threshold=ROT_ERROR_THRESHOLD,
                repeat_counts=repeat_counts,
                run_lma_on_cpu=run_lma_on_cpu,
            )
            sub_runtimes.append(time() - t0)
            sub_success_pcts.append(100 * valid_solution_idxs.sum().item() / batch_size)
            sub_n_found.append(valid_solution_idxs.sum().item())
            print("should not be equal, if not all successful:", solutions.shape[0], valid_solution_idxs.sum().item())

        runtimes.append(np.mean(sub_runtimes))
        runtime_stds.append(np.std(sub_runtimes))

        success_pcts.append(np.mean(sub_success_pcts))
        success_pct_stds.append(np.std(sub_success_pcts))

        # TODO: this isn't getting properly plotted
        n_found.append(np.mean(sub_n_found))
        n_found_stds.append(np.std(sub_n_found))
        print(f"batch_size: {batch_size}, runtime: {runtimes[-1]:.3f}s")
        print(f"batch_size: {batch_size}, runtime: {runtimes[-1]:.3f}s")
    return (
        np.array(batch_sizes),
        np.array(runtimes),
        np.array(runtime_stds),
        np.array(success_pcts),
        np.array(success_pct_stds),
        np.array(n_found),
        np.array(n_found_stds),
    )


def benchmark_get_exact_ik(model_name: str, comparison_data_filepath: Optional[str] = None):
    ikflow_solver, _ = get_ik_solver(model_name)

    # batch_sizes = [1, 10, 100, 500, 750, 1000, 2000, 3000, 4000, 5000]
    batch_sizes = [1, 10, 100, 500, 750, 1000]
    k_retry = 3
    # all_repeat_counts = [(1, 3, 10), (1, 5, 10)]
    # all_repeat_counts = [(1, 3, 10), (1, 3)]
    all_repeat_counts = [(1, 3, 10), (1,)]
    device = "cuda:0"

    curves = []
    labels = []

    # for run_lma_on_cpu in [False, True]:
    for run_lma_on_cpu in [True]:
        for repeat_counts in all_repeat_counts:
            curves.append(get_stats(batch_sizes, k_retry, ikflow_solver, device, repeat_counts, run_lma_on_cpu))
            # labels.append(f"repeat_counts: {repeat_counts}, run_lma_on_cpu: {run_lma_on_cpu}")
            labels.append(f"ikflow + lma - {repeat_counts} iterations")

    # Load comparison data from a pickle file
    if comparison_data_filepath is not None:
        with open(comparison_data_filepath, "rb") as f:
            data = pickle.load(f)
            assert abs(data["position_threshold"] - POS_ERROR_THRESHOLD) < 1e-7
            assert abs(data["rotation_threshold"] - ROT_ERROR_THRESHOLD) < 1e-7
            batch_sizes = data["batch_sizes"]
            runtimes = data["runtimes"]
            runtime_stds = data["runtime_stds"]
            success_pcts = data["success_pcts"]
            success_pct_stds = data["success_pct_stds"]
            n_found = data["n_found"]
            n_found_stds = data["n_found_stds"]
            curves.append((
                np.array(batch_sizes),
                np.array(runtimes),
                np.array(runtime_stds),
                np.array(success_pcts),
                np.array(success_pct_stds),
                np.array(n_found),
                np.array(n_found_stds),
            ))
            labels.append(data["label"])

    # Plot results
    #
    fig, (axl, axr, axrr) = plt.subplots(1, 3, figsize=(24, 8))
    fig.tight_layout()
    fig.suptitle(
        "Number of Solutions vs. Runtime for IKFlow + Levenberg-Marquardt IK Solution Generation"
        f" ({POS_ERROR_THRESHOLD} m, {ROT_ERROR_THRESHOLD} rad tol)",
        fontsize=16,
    )

    axl.set_title("Runtime")
    axl.grid(alpha=0.2)
    axl.set_xlabel("Number of Solutions")
    axl.set_ylabel("Runtime (s)")

    axr.set_title("Success Pct")
    axr.set_xlabel("Number of Solutions")
    axr.set_ylabel("Success percentage (%)")

    axrr.set_title("Number of solutions found")
    axrr.grid(alpha=0.2)
    axrr.set_xlabel("Number of Solutions Requested")
    axrr.set_ylabel("Number of Solutions Returned")

    colors = evenly_spaced_colors(int(1.5 * len(curves)))
    max_runtime = 0

    for (
        (batch_sizes, runtimes, runtime_stds, success_pcts, success_pct_stds, n_found, n_found_stds),
        label,
        color,
    ) in zip(curves, labels, colors):
        max_runtime = max(max_runtime, runtimes.max())
        axl.plot(batch_sizes, runtimes, label=label, color=color)
        axl.fill_between(batch_sizes, runtimes - runtime_stds, runtimes + runtime_stds, alpha=0.15, color=color)
        axl.scatter(batch_sizes, runtimes, s=15, color=color)

        axr.plot(batch_sizes, success_pcts, label=label, color=color)
        axr.fill_between(
            batch_sizes, success_pcts - success_pct_stds, success_pcts + success_pct_stds, alpha=0.15, color=color
        )
        axr.scatter(batch_sizes, success_pcts, s=15, color=color)

        axrr.plot(batch_sizes, n_found, label=label, color=color)
        axrr.fill_between(batch_sizes, n_found - n_found_stds, n_found + n_found_stds, alpha=0.15, color=color)
        axrr.scatter(batch_sizes, n_found, s=15, color=color)

    # axl.legend()
    axl.set_ylim(-0.01, max_runtime * 1.1)
    axr.legend()
    axrr.legend()
    plt.savefig(f"exact_ik_runtime__model:{model_name}.pdf", bbox_inches="tight")
    plt.savefig(f"exact_ik_runtime__model:{model_name}.png", bbox_inches="tight")
    plt.show()


""" Example usage

uv run python scripts/benchmark_generate_exact_solutions.py --model_name=panda__full__lp191_5.25m
uv run python scripts/benchmark_generate_exact_solutions.py --model_name=panda__full__lp191_5.25m --comparison_data_filepath=ik_runtime_results__curobo.pkl

"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--comparison_data_filepath", type=str, required=False)
    args = parser.parse_args()

    # benchmark_get_pose_error()
    benchmark_get_exact_ik(args.model_name, comparison_data_filepath=args.comparison_data_filepath)
