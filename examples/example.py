import argparse

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver

import torch

set_seed()


""" Usage 

python examples/example.py --model_name=panda__full__lp191_5.25m

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
        [0.5, 0.5, 0.5, 1, 0, 0, 0], device="cpu"
    )  # Note: quaternions format for ikflow is [w x y z] (I have learned this the hard way...)
    number_of_solutions = 5

    # -> Get some unrefined solutions
    solutions, l2_errors, angular_errors, joint_limits_exceeded, self_colliding, runtime = (
        ik_solver.generate_ik_solutions(target_pose, number_of_solutions, return_detailed=True)
    )
    print(
        "\nGot {} ikflow solutions in {} ms. Solution L2 errors = {} (cm)".format(
            number_of_solutions, round(runtime * 1000, 3), l2_errors * 100
        )
    )

    # -> Get some refined solutions
    solutions, l2_errors, angular_errors, joint_limits_exceeded, self_colliding, runtime = (
        ik_solver.generate_exact_ik_solutions(target_pose[None, :], number_of_solutions)
    )
    print(
        "Got {} refined ikflow solutions in {} ms. Solution L2 errors = {} (cm)".format(
            number_of_solutions, round(runtime * 1000, 3), l2_errors * 100
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
        device="cpu",
    )

    # -> unrefined solutions
    solutions, l2_errors, angular_errors, joint_limits_exceeded, self_colliding, runtime = (
        ik_solver.generate_ik_solutions(target_poses, refine_solutions=False, return_detailed=True)
    )
    print(
        "\nGot {} ikflow solutions in {} ms. The L2 error of the solutions = {} (cm)".format(
            target_poses.shape[0], round(runtime * 1000, 3), l2_errors * 100
        )
    )

    # -> refined solutions
    solutions, l2_errors, angular_errors, joint_limits_exceeded, self_colliding, runtime = (
        ik_solver.generate_exact_ik_solutions(target_poses, return_detailed=True)
    )
    print(
        "Got {} refined ikflow solutions in {} ms. The L2 error of the solutions = {} (cm)".format(
            target_poses.shape[0], round(runtime * 1000, 3), l2_errors * 100
        )
    )

    print("\nDone")
