import argparse
from typing import Optional
import os

import config
from utils.utils import get_dataset_directory, safe_mkdir, print_tensor_stats, get_sum_joint_limit_range
from utils.robots import get_robot, RobotModel, KlamptRobotModel, RobotModel2d

import numpy as np
import torch
import tqdm

TRAINING_SET_SIZE_SMALL = int(1e5)
TEST_SET_SIZE = 15000


def print_saved_datasets_stats(robots: Optional[RobotModel] = []):
    """Printout summary statistics for each dataset. Optionaly print out the default joint limits of all robots in
    `robots`
    """

    print("\tSummary info on all saved datasets:")

    def print_joint_limits(limits, robot=""):
        print(f"print_joint_limits('{robot}')")
        sum_range = 0
        for idx, (l, u) in enumerate(limits):
            sum_range += u - l
            print(f"  joint_{idx}: ({l},\t{u})")
        print(f"  sum_range: {sum_range}")

    for robot in robots:
        print_joint_limits(robot.actuated_joints_limits, robot=robot)

    sp = "\t"
    print(f"\nrobot {sp} dataset_name {sp} sum_joint_range")
    print(f"----- {sp} ------------ {sp} ---------------")

    # For each child directory (for each robot) in config.DATASET_DIR:
    for dataset_directory, dirs, files in os.walk(config.DATASET_DIR):

        # Ignore the config.DATASET_DIR itself. os.walk returns the parent direcory followed by all child
        # directories
        if len(dirs) > 0:
            continue

        dataset_name = dataset_directory.split("/")[-1]
        robot_name = dataset_name.split("_")[0]
        samples_tr = torch.load(os.path.join(dataset_directory, "samples_tr.pt"))
        sum_joint_range = get_sum_joint_limit_range(samples_tr)
        print(f"{robot_name} {sp} {dataset_name} {sp} {sum_joint_range}")


# TODO(@jeremysm): Consider adding plots from generated datasets. See `plot_dataset_samples()` in 'Dataset visualization and validation.ipynb'
def save_dataset_to_disk(
    robot: RobotModel,
    dataset_directory: str,
    training_set_size: int,
    test_set_size: int,
    progress_bar=True,
    solver="klampt",
    return_if_existing_dir=False,
):
    """
    Save training & testset numpy arrays to the provided directory
    """
    INTERNAL_BATCH_SIZE = 5000

    assert solver in ["klampt", "kinpy", "batch_fk"]
    assert (
        training_set_size % INTERNAL_BATCH_SIZE == 0
    ), f"Datset size must be divisble by {INTERNAL_BATCH_SIZE} (because I am lazy programmer)"

    dir_exists = os.path.isdir(dataset_directory)
    if return_if_existing_dir and dir_exists:
        print(
            f"save_dataset_to_disk(): '{dataset_directory}' exists already and return_if_existing_dir is enabled, returning"
        )
        return
    if dir_exists:
        print(f"Warning, directory '{dataset_directory}' already exists. Overwriting")
    safe_mkdir(dataset_directory)

    def endpoints_from_samples_3d(sample):
        if solver == "klampt":
            return robot.forward_kinematics_klampt(sample)
        elif solver == "kinpy":
            return robot.forward_kinematics_kinpy(sample)
        elif solver == "batch_fk":
            return robot.forward_kinematics_batch(sample)
        raise ValueError("Shouldn't be here")

    # Build testset
    te_samples = robot.sample(test_set_size, solver=solver)
    if isinstance(robot, KlamptRobotModel):
        te_endpoints = endpoints_from_samples_3d(te_samples)
    elif isinstance(robot, RobotModel2d):
        te_endpoints = robot.forward_kinematics(te_samples)
    else:
        raise ValueError(f"robot: {robot} / {type(robot)} not recognized")

    samples = np.zeros((training_set_size, robot.dim_x))
    end_points = np.zeros((training_set_size, robot.dim_y))

    # Build training set
    iterator = range(0, training_set_size, INTERNAL_BATCH_SIZE)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)

    for i in iterator:
        samples_i = robot.sample(INTERNAL_BATCH_SIZE)

        if isinstance(robot, KlamptRobotModel):
            end_points_i = endpoints_from_samples_3d(samples_i)
        elif isinstance(robot, RobotModel2d):
            end_points_i = robot.forward_kinematics(samples_i)
        else:
            raise ValueError("ummmm.... case not handled")

        samples[i : i + INTERNAL_BATCH_SIZE] = samples_i
        end_points[i : i + INTERNAL_BATCH_SIZE] = end_points_i

    # Save training set
    samples_tr_file_path = os.path.join(dataset_directory, "samples_tr.pt")
    endpoints_tr_file_path = os.path.join(dataset_directory, "endpoints_tr.pt")
    samples_te_file_path = os.path.join(dataset_directory, "samples_te.pt")
    endpoints_te_file_path = os.path.join(dataset_directory, "endpoints_te.pt")
    info_filepath = os.path.join(dataset_directory, "info.txt")

    samples_tr = torch.from_numpy(samples[0 : training_set_size - 1 - INTERNAL_BATCH_SIZE, :]).float()
    endpoints_tr = torch.from_numpy(end_points[0 : training_set_size - 1 - INTERNAL_BATCH_SIZE, :]).float()
    samples_te = torch.from_numpy(te_samples).float()
    endpoints_te = torch.from_numpy(te_endpoints).float()

    for arr in [samples_tr, samples_te]:
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} in samples has zero stdev"

    for arr in [endpoints_tr, endpoints_te]:
        for i in range(arr.shape[1]):
            if torch.std(arr[:, i]) < 0.001:
                print(f"Warning: Array column {i} has non zero stdev")

    with open(info_filepath, "w") as f:
        f.write("Dataset info")
        f.write(f"  robot:             {robot.name}\n")
        f.write(f"  dataset_directory: {dataset_directory}\n")
        f.write(f"  training_set_size: {training_set_size}\n")
        f.write(f"  test_set_size:     {test_set_size}\n")
        f.write(f"  solver:            {solver}\n")

        # Sanity check arrays.
        print_tensor_stats(samples_tr, writable=f, name="samples_tr")
        print_tensor_stats(endpoints_tr, writable=f, name="endpoints_tr")
        print_tensor_stats(samples_te, writable=f, name="samples_te")
        print_tensor_stats(endpoints_te, writable=f, name="endpoints_te")

    torch.save(samples_tr, samples_tr_file_path)
    torch.save(endpoints_tr, endpoints_tr_file_path)
    torch.save(samples_te, samples_te_file_path)
    torch.save(endpoints_te, endpoints_te_file_path)


"""
# Build dataset

python build_dataset.py --robot_name=panda_arm --training_set_size=2500000
python build_dataset.py --robot_name=planar_3dof --training_set_size=100000

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str)
    parser.add_argument("--training_set_size", type=int, default=int(2.5 * 1e6))
    args = parser.parse_args()

    robot = get_robot(args.robot_name)

    # Build dataset
    print(f"Building dataset for robot: {robot}")
    dset_directory = get_dataset_directory(robot.name)
    save_dataset_to_disk(
        robot, dset_directory, args.training_set_size, TEST_SET_SIZE, solver="klampt", progress_bar=True
    )
    print_saved_datasets_stats()
