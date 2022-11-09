from typing import List, Tuple, Optional
import argparse
import sys
import os

sys.path.append(os.getcwd())

from src.ik_solvers import GenerativeIKSolver
from src.math_utils import rotation_matrix_from_quaternion, geodesic_distance
from src.utils import get_ik_solver, set_seed, get_solution_errors
import config

from tqdm import tqdm
import torch
import numpy as np
import yaml

set_seed()

with open("model_descriptions.yaml", "r") as f:
    MODEL_DESCRIPTIONS = yaml.safe_load(f)


def error_stats(
    ik_solver: GenerativeIKSolver,
    testset: np.ndarray,
    latent_noise_distribution: str,
    latent_noise_scale: float,
    samples_per_pose: int,
) -> Tuple[float, float]:
    """Evaluate the given `ik_solver` on the provided `testset`.

    Args:
        ik_solver (GenerativeIKSolver): _description_
        testset (np.ndarray): _description_
        latent_noise_distribution (str): _description_
        latent_noise_scale (float): _description_
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
            samples, _ = ik_solver.make_samples(
                ee_pose_target,
                samples_per_pose,
                latent_noise_distribution=latent_noise_distribution,
                latent_noise_scale=latent_noise_scale,
            )
            l2_errors, ang_errors = get_solution_errors(ik_solver.robot, samples, ee_pose_target)
            l2_errs.append(l2_errors)
            ang_errs.append(ang_errors)

    return np.mean(l2_errs), np.mean(ang_errors)


""" Usage 

python evaluate.py \
    --samples_per_pose=50 \
    --testset_size=500 \
    --n_samples_for_runtime=512 \
    --model_name=panda2_lite 


python evaluate.py \
    --samples_per_pose=50 \
    --testset_size=500 \
    --n_samples_for_runtime=512 \
    --model_name=panda_tpm 

	Average L2 error:      3.738 mm
	Average angular error: 0.6555 deg
	Average runtime:       6.6501 +/- 0.0006 ms (for 100 samples)


python evaluate.py \
    --samples_per_pose=50 \
    --testset_size=50 \
    --all
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
    parser.add_argument("--all", action="store_true", help="Run for all robots in tpms")
    args = parser.parse_args()

    assert (args.model_name is not None) != (args.all), "Only one of 'model_name' or 'all' can be provided"

    if args.model_name is not None:
        model_names = [args.model_name]
    else:
        model_names = [model_name for model_name in MODEL_DESCRIPTIONS]

    results = []
    for model_name in model_names:
        print("\n-------------")
        print(f"Evaluating model '{model_name}'")

        model_weights_filepath = MODEL_DESCRIPTIONS[model_name]["model_weights_filepath"]
        robot_name = MODEL_DESCRIPTIONS[model_name]["robot_name"]
        hparams = MODEL_DESCRIPTIONS[model_name]

        # Build GenerativeIKSolver and set weights
        ik_solver, hyper_parameters = get_ik_solver(model_weights_filepath, robot_name, hparams)

        # Get latent distribution parameters
        latent_noise_distribution = "gaussian"
        latent_noise_scale = 0.75

        # Calculate final L2 error
        samples = ik_solver.robot_model.sample(args.testset_size)
        testset = ik_solver.robot_model.forward_kinematics(samples)
        ave_l2_error, ave_angular_error = error_stats(
            ik_solver, testset, latent_noise_distribution, latent_noise_scale, args.samples_per_pose
        )
        ave_l2_error = ave_l2_error * 1000  # to millimeters
        ave_angular_error = float(np.rad2deg(ave_angular_error))  # to degrees

        # Calculate average runtime for 100 samples
        sample_times = []
        n_samples = args.n_samples_for_runtime
        with torch.inference_mode():
            for i in range(50):
                pose = np.random.random(7)
                sample_times.append(
                    ik_solver.make_samples(
                        pose,
                        n_samples,
                        latent_noise_distribution=latent_noise_distribution,
                        latent_noise_scale=latent_noise_scale,
                    )[1]
                )
        sample_time_100 = 1000 * np.mean(sample_times)  # to milleseconds
        sample_time_100_std = np.std(sample_times)

        print(f"\n\tAverage L2 error:      {round(ave_l2_error, 4)} mm")
        print(f"\tAverage angular error: {round(ave_angular_error, 4)} deg")
        print(
            f"\tAverage runtime:       {round(sample_time_100, 4)} +/- {round(sample_time_100_std, 4)} ms (for"
            f" {n_samples} samples)"
        )

    print("Done")
