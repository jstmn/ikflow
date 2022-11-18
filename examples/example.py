import argparse
import yaml


from ikflow.utils import set_seed
from ikflow.ikflow_solver import get_ik_solver
from ikflow.robots import get_robot

import numpy as np

set_seed()

with open("model_descriptions.yaml", "r") as f:
    MODEL_DESCRIPTIONS = yaml.safe_load(f)


""" Usage 

python examples/example.py --model_name=panda_tpm

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="example.py - brief example of using IKFlow")
    parser.add_argument("--model_name", type=str, help="Name of the saved model to look for in trained_models/")
    args = parser.parse_args()

    assert args.model_name in MODEL_DESCRIPTIONS

    # Build IkflowSolver and set weights
    ik_solver, hyper_parameters = get_ik_solver(args.model_name)
    robot_model = ik_solver.robot

    """SINGLE TARGET-POSE

    The following code is for when you want to run IKFlow on a single target poses. In this example we are getting 
    number_of_solutions=5 solutions for the target pose.
    """
    target_pose = np.array(
        [0.5, 0.5, 0.5, 1, 0, 0, 0]
    )  # Note: quaternions format for ikflow is [w x y z] (I have learned this the hard way...)
    number_of_solutions = 3

    # -> Get some unrefined solutions
    solution, solution_runtime = ik_solver.make_samples(target_pose, number_of_solutions, refine_solutions=False)
    realized_ee_pose = robot_model.forward_kinematics_klampt(solution.cpu().detach().numpy())
    l2_errors = np.linalg.norm(realized_ee_pose[:, 0:3] - target_pose[0:3], axis=1)
    print(
        "\nGot {} ikflow solutions in {} ms. The L2 error of the solutions = {} (mm)".format(
            number_of_solutions, round(solution_runtime * 1000, 3), l2_errors * 1000
        )
    )

    # -> Get some refined solutions
    solution, solution_runtime = ik_solver.make_samples(target_pose, number_of_solutions, refine_solutions=True)
    realized_ee_pose = robot_model.forward_kinematics_klampt(solution.cpu().detach().numpy())
    l2_errors = np.linalg.norm(realized_ee_pose[:, 0:3] - target_pose[0:3], axis=1)
    print(
        "Got {} refined ikflow solutions in {} ms. The L2 error of the solutions = {} (mm)".format(
            number_of_solutions, round(solution_runtime * 1000, 3), l2_errors * 1000
        )
    )

    """MULTIPLE TARGET-POSES

    The following code is for when you want to run IKFlow on multiple target poses at once. The only difference is that 
    you need to call `make_samples_mult_y` instead of `make_samples`.
    """
    target_poses = np.array(
        [
            [0.25, 0, 0.5, 1, 0, 0, 0],
            [0.35, 0, 0.5, 1, 0, 0, 0],
            [0.45, 0, 0.5, 1, 0, 0, 0],
        ]
    )

    # -> unrefined solutions
    solution, solution_runtime = ik_solver.make_samples_mult_y(target_poses, refine_solutions=False)
    realized_ee_pose = robot_model.forward_kinematics_klampt(solution.cpu().detach().numpy())
    l2_errors = np.linalg.norm(realized_ee_pose[:, 0:3] - target_poses[:, 0:3], axis=1)
    print(
        "\nGot {} ikflow solutions in {} ms. The L2 error of the solutions = {} (mm)".format(
            target_poses.shape[0], round(solution_runtime * 1000, 3), l2_errors * 1000
        )
    )

    # -> refined solutions
    solution, solution_runtime = ik_solver.make_samples_mult_y(target_poses, refine_solutions=True)
    realized_ee_pose = robot_model.forward_kinematics_klampt(solution.cpu().detach().numpy())
    l2_errors = np.linalg.norm(realized_ee_pose[:, 0:3] - target_poses[:, 0:3], axis=1)
    print(
        "Got {} refined ikflow solutions in {} ms. The L2 error of the solutions = {} (mm)".format(
            target_poses.shape[0], round(solution_runtime * 1000, 3), l2_errors * 1000
        )
    )

    print("\nDone")
