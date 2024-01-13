from typing import List, Optional
import argparse
from time import time
import sys
from collections import namedtuple
from datetime import datetime

import torch
import numpy as np
from jrl.robots import Robot
import pandas as pd

from ikflow.ikflow_solver import IKFlowSolver
from ikflow.utils import set_seed, boolean_string
from ikflow.model_loading import get_ik_solver, get_all_model_names
from ikflow.evaluation_utils import evaluate_solutions


set_seed()


PD_DATAFRAME_COLUMNS = [
    "Robot",
    "Model name",
    "Mean positional error (cm)",
    "Mean rotational error (deg)",
    "Joint limits exceeded %",
    "Self-colliding %",
    "Mean runtime - UPDATE_THIS_VALUE",
    # "Runtime std - UPDATE_THIS_VALUE",
    "Number of coupling layers",
]
_K_FOR_RUNTIME_STATS = 5
_DEFAULT_LATENT_DISTRIBUTION = "gaussian"
_DEFAULT_LATENT_SCALE = 0.75
ErrorStats = namedtuple(
    "ErrorStats", "mean_l2_error_mm mean_angular_error_deg pct_joint_limits_exceeded pct_self_colliding"
)
RuntimeStats = namedtuple("RuntimeStats", "mean_runtime_ms runtime_std nb_solutions")


def calculate_error_stats(
    ik_solver: IKFlowSolver,
    robot: Robot,
    testset: np.ndarray,
    latent_distribution: str,
    latent_scale: float,
    samples_per_pose: int,
    refine_solutions: bool,
    clamp_to_joint_limits: bool,
) -> ErrorStats:
    """Evaluate the given `ik_solver` on the provided `testset`.

    NOTE: Returns positional error in millimeters and rotational error in degrees
    """
    ik_solver.nn_model.eval()

    pos_errs = []
    rot_errs = []
    jlims_exceeded_count = 0
    self_collisions_count = 0

    with torch.inference_mode():
        for i in range(testset.shape[0]):
            ee_pose_target = testset[i]
            samples = ik_solver.generate_ik_solutions(
                ee_pose_target,
                samples_per_pose,
                latent_distribution=latent_distribution,
                latent_scale=latent_scale,
                clamp_to_joint_limits=clamp_to_joint_limits,
                refine_solutions=refine_solutions,
            )
            pos_errors_i, rot_errors_i, joint_limits_exceeded, self_collisions = evaluate_solutions(
                robot, ee_pose_target, samples
            )
            pos_errs.extend(pos_errors_i)
            rot_errs.extend(rot_errors_i)
            jlims_exceeded_count += joint_limits_exceeded.sum().item()
            self_collisions_count += self_collisions.sum().item()
    n_total = testset.shape[0] * samples_per_pose
    return ErrorStats(
        float(1000 * np.mean(pos_errs)),
        float(np.rad2deg(np.mean(rot_errs))),
        100 * (jlims_exceeded_count / n_total),
        100 * (self_collisions_count / n_total),
    )


def calculate_runtime_stats(ik_solver: IKFlowSolver, n_solutions: int, k: int, refine_solutions: bool) -> RuntimeStats:
    """Collect runtime statistics for the given `ik_solver`. NOTE: Returns runtime in milliseconds

    Returns:
        RuntimeStats: Mean, std of the runtime, number of solutions the runtime is for
    """
    sample_times = []
    poses = ik_solver.robot.forward_kinematics_klampt(ik_solver.robot.sample_joint_angles(n_solutions * k))
    with torch.inference_mode():
        for k_i in range(k):
            target_poses = poses[k_i * n_solutions : (k_i + 1) * n_solutions]
            assert target_poses.shape == (n_solutions, 7)
            t0 = time()
            ik_solver.generate_ik_solutions(target_poses, None, refine_solutions=refine_solutions)[1]
            sample_times.append(time() - t0)
    return RuntimeStats(np.mean(sample_times) * 1000, np.std(sample_times), n_solutions)


def pp_results(args: argparse.Namespace, error_stats: ErrorStats, runtime_stats: RuntimeStats):
    print(f"\n----------------------------------------")
    print(f"> Results for {args.model_name.upper()}")
    print(f"\n  solutions clamped to joint limits: {args.clamp_to_joint_limits}")
    print(f"  non-self-colliding testset:        {args.non_self_colliding_dataset}")
    print(f"  solutions refined:                 {args.do_refinement}")

    print(f"\n  Average positional error:              {round(error_stats.mean_l2_error_mm, 4)} mm")
    print(f"  Average rotational error:         {round(error_stats.mean_angular_error_deg, 4)} deg")
    print(f"  Percent joint limits exceeded: {round(error_stats.pct_joint_limits_exceeded, 4)} %")
    print(f"  Percent self-colliding:        {round(error_stats.pct_self_colliding, 4)} %")
    print(
        f"  Average runtime:               {round(runtime_stats.mean_runtime_ms, 4)} +/-"
        f" {round(runtime_stats.runtime_std, 4)} ms for {runtime_stats.nb_solutions} solutions"
    )
    print(
        f"                                 {round(runtime_stats.mean_runtime_ms/runtime_stats.nb_solutions, 4)} ms"
        " per solution"
    )


def evaluate_model(
    model_name: str, args: argparse.Namespace, print_results: bool = True, pd_table: Optional[pd.DataFrame] = None
):
    ik_solver, hparams = get_ik_solver(model_name)
    robot = ik_solver.robot
    _, testset = robot.sample_joint_angles_and_poses(
        args.testset_size, only_non_self_colliding=args.non_self_colliding_dataset, tqdm_enabled=False
    )

    error_stats = calculate_error_stats(
        ik_solver,
        robot,
        testset,
        _DEFAULT_LATENT_DISTRIBUTION,
        _DEFAULT_LATENT_SCALE,
        args.samples_per_pose,
        refine_solutions=args.do_refinement,
        clamp_to_joint_limits=args.clamp_to_joint_limits,
    )
    runtime_stats = calculate_runtime_stats(
        ik_solver, n_solutions=runtime_n, k=_K_FOR_RUNTIME_STATS, refine_solutions=False
    )
    if print_results:
        pp_results(args, error_stats, runtime_stats)

    if pd_table is not None:
        new_row = [
            robot.name,
            model_name,
            error_stats.mean_l2_error_mm / 10.0,
            error_stats.mean_angular_error_deg,
            error_stats.pct_joint_limits_exceeded,
            error_stats.pct_self_colliding,
            runtime_stats.mean_runtime_ms,
            # runtime_stats.runtime_std,
            hparams.nb_nodes,
        ]
        pd_table.loc[len(pd_table)] = new_row


""" Example usage

python scripts/evaluate.py --testset_size=500 --n_solutions_for_runtime=100 --all

python scripts/evaluate.py --testset_size=500 --model_name=fetch_arm_full_temp
python scripts/evaluate.py --testset_size=5 --model_name=panda__full__lp191_5.25m --samples_per_pose=3
python scripts/evaluate.py --testset_size=5 --model_name=fetch_full_temp_nsc_tpm
python scripts/evaluate.py --testset_size=5 --model_name=fetch_full_temp_nsc_tpm
python scripts/evaluate.py --testset_size=500 --model_name=panda__full__lp191_5.25m --do_refinement
python scripts/evaluate.py --testset_size=500 --model_name=rizon4__snowy-brook-208__global_step=2.75M
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--samples_per_pose", default=50, type=int)
    parser.add_argument(
        "--n_solutions_for_runtime",
        default=100,
        type=int,
        help="Check the average runtime to get this number of solutions",
    )
    parser.add_argument("--non_self_colliding_dataset", type=str, default="true")
    parser.add_argument("--testset_size", default=500, type=int)
    parser.add_argument("--model_name", type=str, help="Name of the saved model (see ikflow/model_descriptions.yaml)")
    parser.add_argument("--do_refinement", action="store_true", help="Run refine solutions")
    parser.add_argument(
        "--clamp_to_joint_limits",
        type=str,
        default="true",
        help="Clamp the solutions to the robots joint limits (this is a bool, should be 'true' or 'false')",
    )
    parser.add_argument("--all", action="store_true", help="Run for all robots in tpms")
    args = parser.parse_args()
    args.non_self_colliding_dataset = boolean_string(args.non_self_colliding_dataset)
    args.do_refinement = boolean_string(args.do_refinement)
    args.clamp_to_joint_limits = boolean_string(args.clamp_to_joint_limits)
    assert not (args.all and args.model_name), "Cannot specify both --all and --model_name"
    runtime_n = args.n_solutions_for_runtime

    # Build IKFlowSolver and set weights
    if args.all:
        df = pd.DataFrame(columns=PD_DATAFRAME_COLUMNS)
        df = df.rename(
            columns={
                "Mean runtime - UPDATE_THIS_VALUE": f"Mean runtime for {args.n_solutions_for_runtime} solutions (ms)",
                # "Runtime std - UPDATE_THIS_VALUE": f"Runtime std (k={_K_FOR_RUNTIME_STATS})",
            }
        )

        for model_name in get_all_model_names():
            evaluate_model(model_name, args, print_results=False, pd_table=df)
        df.sort_values(by=["Robot"])

        with open("model_performances.md", "r") as f:
            current_lines = f.readlines()

        with open("model_performances.md", "w") as f:
            for line in current_lines:
                f.write(line)
            cli_input = "python " + " ".join(sys.argv)
            dt = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            f.write(f"\n\n**{dt}** | Generated with `{cli_input}`\n\n")
            f.write(df.to_markdown())

        print(df)
    else:
        print("\n-------------")
        print(f"Evaluating model '{args.model_name}'")

        evaluate_model(args.model_name, args)
