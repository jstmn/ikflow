from typing import Optional

import os
import sys
import tqdm

sys.path.append(os.getcwd())

from src.dataset_tags import DATASET_TAGS
from src.dataset_utils import get_artifact_name
from src import config
from src.utils import get_sum_joint_limit_range, print_tensor_stats, safe_mkdir
from src.robot_models import RobotModel, KlamptRobotModel, RobotModel2d
from src import robot_models
import wandb
import numpy as np
import torch


# Root directory for where datasets will be stored. Save format is BUILD_DATASET_SAVE_DIR/robot_name/
#  in robot_name/ is: samples_tr.pt, endpoints_tr.pt, samples_te.pt, endpoints_te.pt
BUILD_DATASET_SAVE_DIR = "~/Projects/data/ikflow/datasets/model_refinement"

DEFAULT_TEST_SET_SIZE = 25000
TR_SET_SIZE_SMALL = int(1 * 1e5)


def delete_tagged_datasets():

    # Not working. getting a `wandb.errors.CommError: <Response [400]>` error when calling `artifact.delete()`
    raise NotImplementedError()

    api = wandb.Api()
    tags = SJR_EXP
    robots = [robot_models.Valkyrie.name, robot_models.Atlas.name, robot_models.Robonaut2.name]

    for tag in tags:
        for robot in robots:
            for small in [True, False]:
                artifact_name = get_artifact_name(robot, small, tag)

                alias = artifact_name
                print(artifact_name)
                # api.artifact(f'nnik/artifact:{alias}')
                artifact = api.artifact(f"nnik/{alias}:latest")
                artifact.delete()


def upload_datasets_to_wandb(project="nnik"):
    """Save all datasets in the BUILD_DATASET_SAVE_DIR directory to wandb. Reuploading the same information will
    not increase the amount of stored data on the Weights and Biases server.

    https://docs.wandb.ai/guides/artifacts/artifacts-faqs:

        "What happens if I log a new version of same artifact with only minor changes? Does the storage used double?
        No need to worry! When logging a new version of an artifact, W&B only uploads the files that changed between
        the last version and the new version. For example, if your first version uploads 100 files and the second
        version adds 10 more files, the second artifact will only consume the space of the 10 new files."
    """

    wandb_run = wandb.init(
        project=project,
        job_type="dataset-creation",
        notes="from data.py",
        tags=["DONT_DELETE", "SAVING_ARTIFACT"],
    )

    # For each child directory (for each robot) in BUILD_DATASET_SAVE_DIR:
    for dataset_directory, dirs, files in os.walk(BUILD_DATASET_SAVE_DIR):

        # Ignore the BUILD_DATASET_SAVE_DIR itself. os.walk returns the parent direcory followed by all
        # child directories
        if len(files) == 0:
            continue

        dataset_name = dataset_directory.split("/")[-1]
        print(f"root: {dataset_directory}")
        print(f"  dirs:         {dirs}")
        print(f"  files:        {files}")
        print(f"  dataset_name: {dataset_name}")

        assert len(dataset_name) > 0, f"dataset_name: '{dataset_name}' invalid"

        artifact = wandb.Artifact(dataset_name, type="dataset")
        artifact.add_dir(dataset_directory)
        wandb_run.log_artifact(artifact)

    wandb_run.join()


def print_saved_datasets_stats(robots: Optional[RobotModel] = []):
    """Printout summary statistics for each dataset. Optionaly print out the default joint limits of all robots in
    `robots`
    """

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

    # For each child directory (for each robot) in BUILD_DATASET_SAVE_DIR:
    for dataset_directory, dirs, files in os.walk(BUILD_DATASET_SAVE_DIR):

        # Ignore the BUILD_DATASET_SAVE_DIR itself. os.walk returns the parent direcory followed by all child
        # directories
        if len(files) == 0:
            continue

        dataset_name = dataset_directory.split("/")[-1]
        robot_name = dataset_name.split("_")[0]
        samples_tr = torch.load(os.path.join(dataset_directory, "samples_tr.pt"))
        sum_joint_range = get_sum_joint_limit_range(samples_tr)
        print(f"{robot_name} {sp} {dataset_name} {sp} {sum_joint_range}")


# TODO(@jeremysm): Consider adding plots from generated datasets. See `plot_dataset_samples()` in 'Dataset visualization and validation.ipynb'
def save_dataset_to_disk(
    robot: RobotModel,
    training_set_size: int,
    test_set_size: int,
    artifact_name: str,
    limit_increase=0,
    progress_bar=True,
    solver="klampt",
    return_if_existing_dir=False,
    include_l2_loss_pts=False,
):
    """
    Save training & testset numpy arrays to the BUILD_DATASET_SAVE_DIR/artifact_name/ directory
    """

    assert solver in ["klampt", "kinpy", "batch_fk"]

    n_poses_per_sample = 1
    if include_l2_loss_pts:
        assert isinstance(robot, KlamptRobotModel)
        assert solver in ["klampt", "batch_fk"]
        n_poses_per_sample += len(robot.l2_loss_pts)

    save_dir = os.path.join(BUILD_DATASET_SAVE_DIR, artifact_name)
    dir_exists = os.path.isdir(save_dir)
    if return_if_existing_dir and dir_exists:
        print(f"save_dataset_to_disk(): '{save_dir}' exists already and return_if_existing_dir is enabled, returning")
        return
    if dir_exists:
        print(f"Warning, directory '{save_dir}' already exists. Overwriting")
    safe_mkdir(save_dir)

    def endpoints_from_samples_3d(sample):

        if include_l2_loss_pts:
            if solver == "klampt":
                return robot.forward_kinematics_klampt(sample, with_l2_loss_pts=True)
            else:
                return robot.forward_kinematics_batch(sample, with_l2_loss_pts=True)
        if solver == "klampt":
            return robot.forward_kinematics_klampt(sample)
        elif solver == "kinpy":
            return robot.forward_kinematics_kinpy(sample)
        elif solver == "batch_fk":
            return robot.forward_kinematics_batch(sample)
        raise ValueError("Shouldn't be here")

    # Build testset
    te_samples = robot.sample(test_set_size, limit_increase=limit_increase, solver=solver)
    if isinstance(robot, KlamptRobotModel):
        te_endpoints = endpoints_from_samples_3d(te_samples)
    elif isinstance(robot, RobotModel2d):
        te_endpoints = robot.forward_kinematics(te_samples)
    else:
        raise ValueError(f"robot: {robot} / {type(robot)} not recognized")

    batch = 10000
    samples = np.zeros((training_set_size, robot.dim_x))
    end_points = np.zeros((training_set_size, robot.dim_y * n_poses_per_sample))

    # Build training set
    iterator = range(0, training_set_size, batch)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)

    for i in iterator:
        samples_i = robot.sample(batch, limit_increase=limit_increase)

        if isinstance(robot, KlamptRobotModel):
            end_points_i = endpoints_from_samples_3d(samples_i)
        elif isinstance(robot, RobotModel2d):
            end_points_i = robot.forward_kinematics(samples_i)
        else:
            raise ValueError("ummmm.... case not handled")

        samples[i : i + batch] = samples_i
        end_points[i : i + batch] = end_points_i

    # Save training set
    samples_tr_file_path = os.path.join(save_dir, "samples_tr.pt")
    endpoints_tr_file_path = os.path.join(save_dir, "endpoints_tr.pt")
    samples_te_file_path = os.path.join(save_dir, "samples_te.pt")
    endpoints_te_file_path = os.path.join(save_dir, "endpoints_te.pt")
    info_filepath = os.path.join(save_dir, "info.txt")

    samples_tr = torch.from_numpy(samples[0 : training_set_size - 1 - batch, :]).float()
    endpoints_tr = torch.from_numpy(end_points[0 : training_set_size - 1 - batch, :]).float()
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
        f.write(f"  training_set_size: {training_set_size}\n")
        f.write(f"  test_set_size:     {test_set_size}\n")
        f.write(f"  artifact_name:     {artifact_name}\n")
        f.write(f"  solver:            {solver}\n")
        f.write(f"  limit_increase:    {limit_increase}\n")

        # Sanity check arrays.
        print_tensor_stats(samples_tr, writable=f, name="samples_tr")
        print_tensor_stats(endpoints_tr, writable=f, name="endpoints_tr")
        print_tensor_stats(samples_te, writable=f, name="samples_te")
        print_tensor_stats(endpoints_te, writable=f, name="endpoints_te")

    torch.save(samples_tr, samples_tr_file_path)
    torch.save(endpoints_tr, endpoints_tr_file_path)
    torch.save(samples_te, samples_te_file_path)
    torch.save(endpoints_te, endpoints_te_file_path)


# # TODO(@jeremysm): Consider adding plots from generated datasets. See `plot_dataset_samples()` in 'Dataset visualization and validation.ipynb'
# def save_dataset_to_disk(
#     robot: RobotModel,
#     training_set_size: int,
#     test_set_size: int,
#     artifact_name: str,
#     limit_increase=0,
#     progress_bar=True,
#     solver="klampt",
#     return_if_existing_dir=False,
# ):
#     """
#     Save training & testset numpy arrays to the BUILD_DATASET_SAVE_DIR/artifact_name/ directory
#     """

#     assert solver in ["klampt", "kinpy", "batch_fk"]

#     save_dir = os.path.join(BUILD_DATASET_SAVE_DIR, artifact_name)
#     dir_exists = os.path.isdir(save_dir)
#     if return_if_existing_dir and dir_exists:
#         print(
#             f"save_dataset_to_disk(): '{save_dir}' exists already and return_if_existing_dir is enabled, returning"
#         )
#         return
#     if dir_exists:
#         print(f"Warning, directory '{save_dir}' already exists. Overwriting")
#     safe_mkdir(save_dir)

#     def endpoints_from_samples_3d(sample):
#         if solver == "klampt":
#             return robot.forward_kinematics_klampt(sample)
#         elif solver == "kinpy":
#             return robot.forward_kinematics_kinpy(sample)
#         elif solver == "batch_fk":
#             return robot.forward_kinematics_batch(sample)
#         raise ValueError("Shouldn't be here")

#     # Build testset
#     te_samples = robot.sample(test_set_size, limit_increase=limit_increase, solver=solver)
#     if isinstance(robot, KlamptRobotModel):
#         te_endpoints = endpoints_from_samples_3d(te_samples)
#     elif isinstance(robot, RobotModel2d):
#         te_endpoints = robot.forward_kinematics(te_samples)
#     else:
#         raise ValueError(f"robot: {robot} / {type(robot)} not recognized")

#     batch = 100000
#     samples = np.zeros((training_set_size, robot.dim_x))
#     end_points = np.zeros((training_set_size, robot.dim_y))

#     # Build training set
#     iterator = range(0, training_set_size, batch)
#     if progress_bar:
#         iterator = tqdm.tqdm(iterator)

#     for i in iterator:
#         samples_i = robot.sample(batch, limit_increase=limit_increase)

#         if isinstance(robot, KlamptRobotModel):
#             end_points_i = endpoints_from_samples_3d(samples_i)
#         elif isinstance(robot, RobotModel2d):
#             end_points_i = robot.forward_kinematics(samples_i)
#         else:
#             raise ValueError("ummmm.... case not handled")

#         samples[i : i + batch] = samples_i
#         end_points[i : i + batch] = end_points_i

#     # Save training set
#     samples_tr_file_path = os.path.join(save_dir, "samples_tr.pt")
#     endpoints_tr_file_path = os.path.join(save_dir, "endpoints_tr.pt")
#     samples_te_file_path = os.path.join(save_dir, "samples_te.pt")
#     endpoints_te_file_path = os.path.join(save_dir, "endpoints_te.pt")
#     info_filepath = os.path.join(save_dir, "info.txt")

#     samples_tr = torch.from_numpy(samples[0 : training_set_size - 1 - batch, :]).float()
#     endpoints_tr = torch.from_numpy(end_points[0 : training_set_size - 1 - batch, :]).float()
#     samples_te = torch.from_numpy(te_samples).float()
#     endpoints_te = torch.from_numpy(te_endpoints).float()

#     for arr in [samples_tr, samples_te]:
#         for i in range(arr.shape[1]):
#             assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} in samples has zero stdev"

#     for arr in [endpoints_tr, endpoints_te]:
#         for i in range(arr.shape[1]):
#             if torch.std(arr[:, i]) < 0.001:
#                 print(f"Warning: Array column {i} has non zero stdev")

#     with open(info_filepath, "w") as f:
#         f.write("Dataset info")
#         f.write(f"  robot:             {robot.name}\n")
#         f.write(f"  training_set_size: {training_set_size}\n")
#         f.write(f"  test_set_size:     {test_set_size}\n")
#         f.write(f"  artifact_name:     {artifact_name}\n")
#         f.write(f"  solver:            {solver}\n")
#         f.write(f"  limit_increase:    {limit_increase}\n")

#         # Sanity check arrays.
#         print_tensor_stats(samples_tr, writable=f, name="samples_tr")
#         print_tensor_stats(endpoints_tr, writable=f, name="endpoints_tr")
#         print_tensor_stats(samples_te, writable=f, name="samples_te")
#         print_tensor_stats(endpoints_te, writable=f, name="endpoints_te")

#     torch.save(samples_tr, samples_tr_file_path)
#     torch.save(endpoints_tr, endpoints_tr_file_path)
#     torch.save(samples_te, samples_te_file_path)
#     torch.save(endpoints_te, endpoints_te_file_path)


def save_tagged_dataset_to_disk(
    robot: RobotModel,
    tag: str,
    progress_bar=True,
    solver="klampt",
    return_if_existing_dir=False,
    include_l2_loss_pts=False,
):
    """Build datasets with increased/decreased joint limit ranges"""

    limit_delta = DATASET_TAGS[tag]["joint_limit_delta"]
    tr_set_size = DATASET_TAGS[tag]["tr_set_size"]

    limit_delta_valid = robot.joint_limit_delta_valid(limit_delta, debug=True)
    assert limit_delta_valid, f"Tag '{tag}' is invalid for robot '{robot.name}'"

    # Build regular dataset
    print(f"Building dataset with tag '{tag}' for robot: {robot} with limit_delta: {limit_delta}")

    artifact_name = get_artifact_name(robot.name, False, tag)
    save_dataset_to_disk(
        robot,
        tr_set_size,
        DEFAULT_TEST_SET_SIZE,
        artifact_name,
        limit_increase=limit_delta,
        progress_bar=progress_bar,
        solver=solver,
        return_if_existing_dir=return_if_existing_dir,
        include_l2_loss_pts=include_l2_loss_pts,
    )

    # Build small dataset
    artifact_name = get_artifact_name(robot.name, True, tag)
    save_dataset_to_disk(
        robot,
        TR_SET_SIZE_SMALL,
        DEFAULT_TEST_SET_SIZE,
        artifact_name,
        limit_increase=limit_delta,
        progress_bar=progress_bar,
        solver=solver,
        return_if_existing_dir=return_if_existing_dir,
        include_l2_loss_pts=include_l2_loss_pts,
    )
