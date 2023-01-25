import argparse
from typing import Optional, List
import os
from time import time

from ikflow.config import DATASET_DIR, ALL_DATASET_TAGS, DATASET_TAG_NON_SELF_COLLIDING
from ikflow.utils import (
    get_dataset_directory,
    safe_mkdir,
    print_tensor_stats,
    get_sum_joint_limit_range,
    get_dataset_filepaths,
    assert_joint_angle_tensor_in_joint_limits,
)
from jkinpylib.robots import get_robot, Robot

import numpy as np
import torch
import tqdm

TRAINING_SET_SIZE_SMALL = int(1e5)
TEST_SET_SIZE = 15000


def print_saved_datasets_stats(tags: List[str], robots: Optional[Robot] = []):
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
    for dataset_directory, dirs, files in os.walk(DATASET_DIR):
        # Ignore the config.DATASET_DIR itself. os.walk returns the parent direcory followed by all child
        # directories
        if len(dirs) > 0:
            continue

        dataset_name = dataset_directory.split("/")[-1]
        robot_name = dataset_name.split("_")[0]

        try:
            samples_tr_file_path, _, _, _, _ = get_dataset_filepaths(dataset_directory, tags)
            samples_tr = torch.load(samples_tr_file_path)
            sum_joint_range = get_sum_joint_limit_range(samples_tr)
            print(f"{robot_name} {sp} {dataset_name} {sp} {sum_joint_range}")
        except FileNotFoundError:
            samples_tr_file_path, _, _, _, _ = get_dataset_filepaths(dataset_directory, ALL_DATASET_TAGS)
            samples_tr = torch.load(samples_tr_file_path)
            sum_joint_range = get_sum_joint_limit_range(samples_tr)
            print(f"{robot_name} {sp} {dataset_name} {sp} {sum_joint_range}")


# TODO(@jeremysm): Consider adding plots from generated datasets. See `plot_dataset_samples()` in 'Dataset visualization
# and validation.ipynb'
def save_dataset_to_disk(
    robot: Robot,
    dataset_directory: str,
    training_set_size: int,
    test_set_size: int,
    only_non_self_colliding: bool,
    tags: List[str],
    joint_limit_eps: float = 1e-6,
):
    """
    Save training & testset numpy arrays to the provided directory
    """

    safe_mkdir(dataset_directory)
    if only_non_self_colliding:
        assert DATASET_TAG_NON_SELF_COLLIDING in tags, (
            f"Error: If `only_non_self_colliding` is True, then the tag '{DATASET_TAG_NON_SELF_COLLIDING}' should be in"
            " `tags`"
        )

    # Build testset
    samples_te, poses_te = robot.sample_joint_angles_and_poses(
        test_set_size,
        joint_limit_eps=joint_limit_eps,
        only_non_self_colliding=only_non_self_colliding,
        tqdm_enabled=True,
    )
    samples_tr, poses_tr = robot.sample_joint_angles_and_poses(
        training_set_size,
        joint_limit_eps=joint_limit_eps,
        only_non_self_colliding=only_non_self_colliding,
        tqdm_enabled=True,
    )

    # Save training set
    (
        samples_tr_file_path,
        poses_tr_file_path,
        samples_te_file_path,
        poses_te_file_path,
        info_filepath,
    ) = get_dataset_filepaths(dataset_directory, tags)

    samples_tr = torch.tensor(samples_tr, dtype=torch.float32)
    poses_tr = torch.tensor(poses_tr, dtype=torch.float32)
    samples_te = torch.tensor(samples_te, dtype=torch.float32)
    poses_te = torch.tensor(poses_te, dtype=torch.float32)

    # Sanity check
    for arr in [samples_tr, samples_te, poses_tr, poses_te]:
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} in samples has zero stdev"
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples_tr, "samples_tr", 0.0)
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples_te, "samples_te", 0.0)

    with open(info_filepath, "w") as f:
        f.write("Dataset info")
        f.write(f"  robot:             {robot.name}\n")
        f.write(f"  dataset_directory: {dataset_directory}\n")
        f.write(f"  training_set_size: {training_set_size}\n")
        f.write(f"  test_set_size:     {test_set_size}\n")

        # Sanity check arrays.
        print_tensor_stats(samples_tr, writable=f, name="samples_tr")
        print_tensor_stats(poses_tr, writable=f, name="poses_tr")
        print_tensor_stats(samples_te, writable=f, name="samples_te")
        print_tensor_stats(poses_te, writable=f, name="poses_te")

    torch.save(samples_tr, samples_tr_file_path)
    torch.save(poses_tr, poses_tr_file_path)
    torch.save(samples_te, samples_te_file_path)
    torch.save(poses_te, poses_te_file_path)


def _get_tags(args):
    tags = []
    if args.only_non_self_colliding:
        tags.append(DATASET_TAG_NON_SELF_COLLIDING)
    else:
        print("====" * 10)
        print("WARNING: Saving dataset with self-colliding configurations. This is not recommended.")
    tags.sort()
    for tag in tags:
        assert tag in ALL_DATASET_TAGS
    return tags


"""
# Build dataset

python scripts/build_dataset.py --robot_name=fetch --training_set_size=25000000 --only_non_self_colliding
python scripts/build_dataset.py --robot_name=panda --training_set_size=25000000 --only_non_self_colliding
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str)
    parser.add_argument("--training_set_size", type=int, default=int(2.5 * 1e6))
    parser.add_argument("--only_non_self_colliding", action="store_true")
    args = parser.parse_args()

    robot = get_robot(args.robot_name)
    tags = _get_tags(args)

    # Build dataset
    print(f"Building dataset for robot: {robot}")
    dset_directory = get_dataset_directory(robot.name)
    t0 = time()
    save_dataset_to_disk(
        robot,
        dset_directory,
        args.training_set_size,
        TEST_SET_SIZE,
        args.only_non_self_colliding,
        tags,
        joint_limit_eps=0.004363323129985824,  # np.deg2rad(0.25)
    )
    print(f"Saved dataset with {args.training_set_size} samples in {time() - t0:.2f} seconds")
    print_saved_datasets_stats(tags)
