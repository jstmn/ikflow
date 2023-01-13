from typing import List, Tuple
import argparse
from time import time

import torch
import numpy as np
from jkinpylib.robots import Robot

from ikflow.ikflow_solver import IKFlowSolver
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.evaluation_utils import get_solution_errors


set_seed()


def error_stats(
    ik_solver: IKFlowSolver,
    robot: Robot,
    testset: np.ndarray,
    latent_distribution: str,
    latent_scale: float,
    samples_per_pose: int,
    refine_solutions: bool,
) -> Tuple[float, float]:
    """Evaluate the given `ik_solver` on the provided `testset`.

    NOTE: Returns L2 error in millimeters and angular error in degrees

    Args:
        ik_solver (IKFlowSolver): _description_
        testset (np.ndarray): _description_
        latent_distribution (str): _description_
        latent_scale (float): _description_
        samples_per_pose (int): _description_

    Returns:
        Tuple[float, float]: _description_
    """
    ik_solver.nn_model.eval()

    l2_errs: List[List[float]] = []
    ang_errs: List[List[float]] = []

    with torch.inference_mode():
        for i in range(testset.shape[0]):
            ee_pose_target = testset[i]
            samples, _ = ik_solver.solve(
                ee_pose_target,
                samples_per_pose,
                latent_distribution=latent_distribution,
                latent_scale=latent_scale,
                refine_solutions=refine_solutions,
            )
            l2_errors, ang_errors = get_solution_errors(robot, samples, ee_pose_target)
            l2_errs.append(l2_errors)
            ang_errs.append(ang_errors)
    return 1000 * np.mean(l2_errs), float(np.rad2deg(np.mean(ang_errors)))


def runtime_stats(ik_solver: IKFlowSolver, n_solutions: int, k: int, refine_solutions: bool) -> Tuple[float, float]:
    """Collect runtime statistics for the given `ik_solver`. NOTE: Returns runtime in milliseconds

    Returns:
        Tuple[float, float]: Mean, std of the runtime
    """
    sample_times = []
    poses = ik_solver.robot.forward_kinematics_klampt(ik_solver.robot.sample_joint_angles(n_solutions * k))
    with torch.inference_mode():
        for k_i in range(k):
            target_poses = poses[k_i * n_solutions : (k_i + 1) * n_solutions]
            assert target_poses.shape == (n_solutions, 7)
            t0 = time()
            ik_solver.solve_n_poses(target_poses, refine_solutions=refine_solutions)[1]
            sample_times.append(time() - t0)
    return np.mean(sample_times) * 1000, np.std(sample_times)


def pp_results(title, mean_l2_error, mean_angular_error, mean_runtime, runtime_std, runtime_n):
    print(f"\n----------------------------------------")
    print(f"> {title}")
    print(f"\n    Average L2 error:      {round(mean_l2_error, 4)} mm")
    print(f"    Average angular error: {round(mean_angular_error, 4)} deg")
    print(
        f"    Average runtime:       {round(mean_runtime, 4)} +/- {round(runtime_std, 4)} ms (for {runtime_n} samples)"
    )
    print(f"                           {round(mean_runtime/runtime_n, 4)} ms per solution")


""" Usage 

python scripts/evaluate.py --testset_size=500 --model_name=panda_full_tpm
python scripts/evaluate.py --testset_size=500 --model_name=panda_liteplus_tpm
python scripts/evaluate.py --testset_size=500 --model_name=panda_lite_tpm




python scripts/evaluate.py \
    --samples_per_pose=50 \
    --testset_size=500 \
    --n_samples_for_runtime=512 \
    --do_refinement \
    --model_name=panda_full_tpm


Expected output:

    ...
    ----------------------------------------
    > IKFlow with solution refinement

        Average L2 error:      0.5837 mm
        Average angular error: 0.0075 deg
        Average runtime:       76.1609 +/- 0.0112 ms (for 512 samples)
                            0.1488 ms per solution

    ----------------------------------------
    > Vanilla IKFlow

        Average L2 error:      7.5254 mm
        Average angular error: 1.6287 deg
        Average runtime:       10.5191 +/- 0.0 ms (for 512 samples)
                            0.0205 ms per solution

    Done
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--samples_per_pose", default=50, type=int)
    parser.add_argument(
        "--n_samples_for_runtime",
        default=100,
        type=int,
        help="Check the average runtime to get this number of solutions",
    )
    parser.add_argument("--testset_size", default=500, type=int)
    parser.add_argument("--model_name", type=str, help="Name of the saved model to look for in trained_models/")
    parser.add_argument("--do_refinement", action="store_true", help="Run refine solutions")
    parser.add_argument("--all", action="store_true", help="Run for all robots in tpms")
    args = parser.parse_args()

    assert args.all is False, f"--all currently unimplemented"

    # Get latent distribution parameters
    latent_distribution = "gaussian"
    latent_scale = 0.75
    runtime_n = args.n_samples_for_runtime

    print("\n-------------")
    print(f"Evaluating model '{args.model_name}'")

    # Build IKFlowSolver and set weights
    ik_solver, hyper_parameters = get_ik_solver(args.model_name)
    robot = ik_solver.robot
    testset = robot.forward_kinematics_klampt(robot.sample_joint_angles(args.testset_size))

    # ------------------------
    # With solution refinement
    #
    if args.do_refinement:
        mean_l2_error, mean_angular_error = error_stats(
            ik_solver,
            robot,
            testset,
            latent_distribution,
            latent_scale,
            args.samples_per_pose,
            refine_solutions=True,
        )
        mean_runtime, runtime_std = runtime_stats(ik_solver, n_solutions=runtime_n, k=5, refine_solutions=True)
        pp_results(
            "IKFlow with solution refinement", mean_l2_error, mean_angular_error, mean_runtime, runtime_std, runtime_n
        )

    # ---------------------------
    # Without solution refinement
    #
    mean_l2_error, mean_angular_error = error_stats(
        ik_solver,
        robot,
        testset,
        latent_distribution,
        latent_scale,
        args.samples_per_pose,
        refine_solutions=False,
    )
    mean_runtime, runtime_std = runtime_stats(ik_solver, n_solutions=runtime_n, k=5, refine_solutions=False)
    pp_results("Vanilla IKFlow", mean_l2_error, mean_angular_error, mean_runtime, runtime_std, runtime_n)
