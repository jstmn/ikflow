import argparse
from src.dataset_tags import SJR_EXP, SJR_EXP_2
from src.dataset_utils import get_artifact_name
from src.dataset_building import (
    save_dataset_to_disk,
    save_tagged_dataset_to_disk,
    upload_datasets_to_wandb,
    print_saved_datasets_stats,
    DEFAULT_TEST_SET_SIZE,
)
from src.robot_models import get_robot


def build_regular_dataset(robot_name: str):
    """Build regular (non tagged) datasets"""
    robot = get_robot(robot_name)
    tr_set_size = int(2.5 * 1e6)
    tr_set_size_small = int(1e5)

    # Build regular dataset
    print(f"Building dataset for robot: {robot}")
    artifact_name = get_artifact_name(robot=robot.name, is_small=False, tag=None)
    save_dataset_to_disk(robot, tr_set_size, DEFAULT_TEST_SET_SIZE, artifact_name, solver="klampt", progress_bar=True)

    # Build small dataset
    print(f"Building small dataset for robot: {robot}")
    artifact_name = get_artifact_name(robot=robot.name, is_small=True, tag=None)
    save_dataset_to_disk(
        robot, tr_set_size_small, DEFAULT_TEST_SET_SIZE, artifact_name, solver="klampt", progress_bar=True
    )


def build_tagged_datasets(robot_name: str, meta_tag: str, return_if_existing_dir=False):
    """Builds a tagged dataset"""

    assert meta_tag.upper() in ["SJR_EXP", "SJR_EXP_2"]

    robot = get_robot(robot_name)

    if meta_tag.upper() == "SJR_EXP":
        tags = SJR_EXP
    elif meta_tag.upper() == "SJR_EXP_2":
        tags = SJR_EXP_2

    # Build regular dataset
    print(f"Building tagged dataset from metatags '{meta_tag}' for {robot.name}")

    for tag in tags:
        print("\ntag:", tag)
        try:
            save_tagged_dataset_to_disk(robot, tag, progress_bar=True, return_if_existing_dir=return_if_existing_dir)
        except AssertionError:
            print(f"Tag: '{tag}' invalid for robot '{robot.name}'")


def build_tagged_dataset(robot_name: str, tag: str, return_if_existing_dir=False, include_l2_loss_pts=False):
    """Builds a tagged dataset"""

    robot = get_robot(robot_name)

    if include_l2_loss_pts:
        #         tag = "includes_l2_loss_pts"
        #         tag = "includes_l2_loss_pts1pt"
        tag = "includes_l2_loss_pts2pt"

    # Build regular dataset
    print(f"Building tagged dataset '{tag}' for {robot.name}")
    try:
        save_tagged_dataset_to_disk(
            robot,
            tag,
            progress_bar=True,
            return_if_existing_dir=return_if_existing_dir,
            include_l2_loss_pts=include_l2_loss_pts,
        )
    except AssertionError:
        print(f"Tag: '{tag}' invalid for robot '{robot.name}'")


"""
# Relogin to wandb:
python3 -m wandb login --relogin

# Build regular dataset
python3 cli_dataset.py --build_regular_dataset --robot=baxter

# Build dataset that includes l2_loss_pts
python3 cli_dataset.py --include_l2_loss_pts --robot=robosimian


# Build single tagged dataset
python3 cli_dataset.py --tag="LARGE50" --robot=robosimian


# Build tagged datasets via. meta tag
python3 cli_dataset.py --meta_tag=SJR_EXP_2 --robot=pr2 --return_if_existing_dir

# Upload datasets
python3 cli_dataset.py --upload_datasets_to_wandb --print_dataset_stats

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str)
    parser.add_argument("--meta_tag", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--project", default="nnik", type=str)
    parser.add_argument("--build_regular_dataset", action="store_true")
    parser.add_argument("--print_dataset_stats", action="store_true")
    parser.add_argument("--upload_datasets_to_wandb", action="store_true")
    parser.add_argument("--return_if_existing_dir", action="store_true")
    parser.add_argument("--include_l2_loss_pts", action="store_true")
    args = parser.parse_args()

    assert not (
        (args.meta_tag is not None) and args.build_regular_dataset
    ), "Can only pass one of `--meta_tag`, `--build_regular_dataset`"
    assert not ((args.meta_tag is not None) and (args.tag is not None)), "Can only pass one of `--meta_tag`, `--tag`"

    if args.tag is not None or args.include_l2_loss_pts:
        build_tagged_dataset(
            args.robot,
            args.tag,
            return_if_existing_dir=args.return_if_existing_dir,
            include_l2_loss_pts=args.include_l2_loss_pts,
        )

    if args.meta_tag is not None:
        build_tagged_datasets(args.robot, args.meta_tag, return_if_existing_dir=args.return_if_existing_dir)

    elif args.build_regular_dataset:
        build_regular_dataset(args.robot)

    if args.upload_datasets_to_wandb:
        upload_datasets_to_wandb(args.project)
        # Afterwards: Move all datasets in BUILD_DATASET_SAVE_DIR elsewhere so that they aren't reuploaded
        print("Reminder - move uploaded datasets out of DATASET_SAVE_DIR after upload")

    if args.print_dataset_stats:
        print_saved_datasets_stats()
