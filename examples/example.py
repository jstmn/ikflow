import argparse

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import device

import torch

set_seed()


""" Usage 

uv run python examples/example.py --model_name=panda__full__lp191_5.25m

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="minimial example of running IKFlow")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "Name of the saved model ('panda__full__lp191_5.25m' should work). Defined in"
            " ikflow/model_descriptions.yaml"
        ),
    )
    args = parser.parse_args()

    # Build IKFlowSolver and set weights
    ik_solver, hyper_parameters = get_ik_solver(args.model_name)
    robot = ik_solver.robot

    """SINGLE TARGET-POSE

    The following code is for when you want to run IKFlow on a single target poses. In this example we are getting 
    number_of_solutions=5 solutions for the target pose.
    """
    target_pose = torch.tensor(
        [0.5, 0.5, 0.5, 1, 0, 0, 0], device=device
    )  # Note: quaternions format for ikflow is [w x y z] (I have learned this the hard way...)
    number_of_solutions = 5

    # -> Get some approximate solutions
    solutions, positional_errors, rotational_errors, joint_limits_exceeded, self_colliding, runtime = (
        ik_solver.generate_ik_solutions(target_pose, n=number_of_solutions, return_detailed=True)
    )
    print(
        "\nGot {} ikflow solutions in {} ms. Solution positional errors = {} (cm)".format(
            number_of_solutions, round(runtime * 1000, 3), positional_errors * 100
        )
    )

    # -> Get some exact solutions
    solutions, _ = ik_solver.generate_exact_ik_solutions(target_pose.expand((number_of_solutions, 7)))
    print(
        "Got {} exact ikflow solutions in {} ms. Solution positional errors = {} (cm)".format(
            len(solutions), round(runtime * 1000, 3), "(TODO)"
        )
    )

    """MULTIPLE TARGET-POSES

    The following code is for when you want to run IKFlow on multiple target poses at once.
    """
    target_poses = torch.tensor(
        [
            [0.25, 0, 0.5, 1, 0, 0, 0],
            [0.35, 0, 0.5, 1, 0, 0, 0],
            [0.45, 0, 0.5, 1, 0, 0, 0],
            [0.55, 0, 0.5, 1, 0, 0, 0],
            [0.65, 0, 0.5, 1, 0, 0, 0],
        ],
        device=device,
    )

    # -> approximate solutions
    solutions, positional_errors, rotational_errors, joint_limits_exceeded, self_colliding, runtime = (
        ik_solver.generate_ik_solutions(target_poses, n=len(target_poses), refine_solutions=False, return_detailed=True)
    )
    print(
        "\nGot {} ikflow solutions in {} ms. The positional error of the solutions = {} (cm)".format(
            target_poses.shape[0], round(runtime * 1000, 3), positional_errors * 100
        )
    )

    # -> exact solutions
    solutions, _ = ik_solver.generate_exact_ik_solutions(target_poses)
    print(
        "Got {} exact ikflow solutions in {} ms. The positional error of the solutions = {} (cm)".format(
            len(solutions), round(runtime * 1000, 3), "(TODO)"
        )
    )

    print("\nDone")
