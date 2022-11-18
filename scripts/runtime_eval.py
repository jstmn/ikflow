import argparse

# Update sys.path so you can run this file from the projects root directory
import sys
import os


from ikflow.supporting_types import IkflowModelParameters
from ikflow.ikflow_solver import IkflowSolver
from ikflow.robots import get_robot

import torch
import numpy as np


def calculate_ave_runtime(ik_solver: IkflowSolver, n_samples: int):
    """_summary_

    Args:
        ik_solver (IkflowSolver): _description_
        n_samples (int): _description_

    Returns:
        _type_: _description_
    """

    # Calculate average runtime for 100 samples
    sample_times = []
    with torch.inference_mode():
        for i in range(30):
            pose = np.random.random(7)
            sample_times.append(
                ik_solver.make_samples(
                    pose,
                    n_samples,
                )[1]
            )
    sample_time_100 = 1000 * np.mean(sample_times)  # to milleseconds
    sample_time_100_std = np.std(sample_times)
    return sample_time_100, sample_time_100_std


""" Example usage

python tools/runtime_eval.py \
    --nb_nodes_range 4 6 8 10 12 14
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Runtime tester")

    # Note: WandB saves artifacts by the run ID (i.e. '34c2gimi') not the run name ('dashing-forest-33'). This is
    # slightly annoying because you need to click on a run to get its ID.
    parser.add_argument("--wandb_run_id_to_load_checkpoint", type=str, help="Example: '34c2gimi'")
    parser.add_argument("--robot_name", type=str)

    # Model parameters
    parser.add_argument("--dim_latent_space", type=int, default=9)
    parser.add_argument("--coeff_fn_config", type=int, default=3)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=2)
    parser.add_argument("--nb_nodes_range", nargs="+", type=int)

    parser.add_argument(
        "--n_samples_for_runtime",
        default=100,
        type=int,
        help="Check the average runtime to get this number of solutions",
    )

    args = parser.parse_args()

    hparams = IkflowModelParameters()
    hparams.dim_latent_space = args.dim_latent_space
    hparams.coeff_fn_config = args.coeff_fn_config
    hparams.coeff_fn_internal_size = args.coeff_fn_internal_size
    robot = get_robot("panda_arm")

    print(f"Average runtime for {args.n_samples_for_runtime} samples")

    # TODO(@jstmn): Format output
    print("\nNumber of nodes\t| Runtime (ms)")
    print("----------------|-------------------")
    for nb_nodes in args.nb_nodes_range:
        hparams.nb_nodes = nb_nodes
        ik_solver = IkflowSolver(hparams, robot, verbosity=0)

        ave_runtime, std = calculate_ave_runtime(ik_solver, args.n_samples_for_runtime)
        print(f"{nb_nodes}\t\t| {ave_runtime:.2f}")
