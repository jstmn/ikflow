from typing import Optional
import os
import sys
from time import time

sys.path.append(os.getcwd())

from src import config
from src.utils import get_sum_joint_limit_range
from src.dataset_utils import get_artifact_name
from src.dataset_tags import DATASET_TAGS
import wandb
import numpy as np
import torch


from wandb.wandb_run import Run

# Root directory for where datasets will be stored. Save format is BUILD_DATASET_SAVE_DIR/robot_name/
#  in robot_name/ is: samples_tr.pt, endpoints_tr.pt, samples_te.pt, endpoints_te.pt
DATASET_DOWNLOAD_DIR = "/tmp/ikflow_datasets"


class Dataset:
    def __init__(
        self,
        wandb_run: Run,
        robot_name: str,
        batch_size: int,
        verbosity: Optional[int] = 0,
        use_small_dataset: Optional[bool] = False,
        tag: Optional[str] = None,
    ):
        """Initialize a Dataset object.

        Args:
            robot_name (str): Name of the robot
            batch_size (str): The batch size of the training set
            verbosity (Optional[int]): Verbosity level. Currently unused
            use_small_dataset (Optional[bool]): Boolean indicating whether to use the small dataset saved for
                                                each robot or the regular dataset
        """

        if use_small_dataset:
            print("______________________________________________________________")
            print("______________________________________________________________")
            print("                Heads up: using small dataset.")
            print("______________________________________________________________")
            print("______________________________________________________________")

        if not tag is None and len(tag) > 0:
            assert tag in DATASET_TAGS

        artifact_name = get_artifact_name(robot_name, use_small_dataset, tag) + ":latest"
        artifact = wandb_run.use_artifact(artifact_name)

        t0 = time()
        download_root = os.path.join(DATASET_DOWNLOAD_DIR, artifact_name)
        print(f"Starting download of artifact '{artifact_name}' to '{download_root}'")
        artifact_dir_from_wandb = artifact.download(root=download_root)
        print(f"Downloaded dataset directory '{artifact_dir_from_wandb}' in {round(time() - t0, 3)} seconds")

        try:
            print("\ninfo.txt:")
            with open(os.path.join(artifact_dir_from_wandb, "info.txt"), "r") as f:
                print(f.read())
        except FileNotFoundError:
            print("No info.txt")

        self._samples_tr = torch.load(os.path.join(artifact_dir_from_wandb, "samples_tr.pt"))
        self._endpoints_tr = torch.load(os.path.join(artifact_dir_from_wandb, "endpoints_tr.pt"))
        self._samples_te = torch.load(os.path.join(artifact_dir_from_wandb, "samples_te.pt"))
        self._endpoints_te = torch.load(os.path.join(artifact_dir_from_wandb, "endpoints_te.pt"))
        self._endpoints_te_np = self._endpoints_te.cpu().detach().numpy()

        self._batch_size = batch_size
        self._sum_joint_limit_range = get_sum_joint_limit_range(self._samples_tr)

        # TODO(@jeremysm): Should a new DataLoader be created every epoch?
        self._train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._samples_tr, self._endpoints_tr),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

    def log_dataset_sizes(self, epoch=0, batch_nb=0):
        """Log the training and testset size to wandb"""
        assert self._samples_tr.shape[0] == self._endpoints_tr.shape[0]
        assert self._samples_te.shape[0] == self._endpoints_te.shape[0]
        wandb.log(
            {
                "epoch": epoch,
                "batch_nb": batch_nb,
                "sum_joint_limit_range": self._sum_joint_limit_range,
                "dataset_size_tr": self._samples_tr.shape[0],
                "dataset_size_te": self._samples_te.shape[0],
            }
        )

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        return self._train_loader

    @property
    def endpoints_te(self) -> torch.Tensor:
        return self._endpoints_te

    @property
    def endpoints_te_numpy(self) -> np.array:
        return self._endpoints_te_np

    @property
    def endpoints_tr(self) -> torch.Tensor:
        return self._endpoints_tr

    @property
    def samples_te(self) -> torch.Tensor:
        return self._samples_te

    @property
    def samples_tr(self) -> torch.Tensor:
        return self._samples_tr

    @property
    def sum_joint_limit_range(self) -> float:
        return self._sum_joint_limit_range
