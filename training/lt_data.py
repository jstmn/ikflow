from typing import Optional
import os

import config
from utils.utils import get_sum_joint_limit_range

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.core.datamodule import LightningDataModule

import wandb
import torch


def _get_dataset_directory(robot: str, is_small: bool):
    """ Return the path of the directory
    """
    subdir_name = f"{robot}"
    if is_small:
        subdir_name += "__SMALL" 
    return os.path.join(config.DATASET_DIR, subdir_name) 

class IkfLitDataset(LightningDataModule):
    def __init__(
        self,
        robot_name: str,
        batch_size: int,
        use_small_dataset: bool = False,
        val_set_size: int = 500,
    ):
        self._robot_name = robot_name
        self._batch_size = batch_size
        self._use_small_dataset = use_small_dataset
        self._val_set_size = val_set_size

        dataset_directory = _get_dataset_directory(self._robot_name, self._use_small_dataset)
        assert os.path.isdir(dataset_directory), f"Directory '{dataset_directory}' doesn't exist - have you created the dataset for this robot yet?"

        self._samples_tr = torch.load(os.path.join(dataset_directory, "samples_tr.pt")).to("cuda:0")
        self._endpoints_tr = torch.load(os.path.join(dataset_directory, "endpoints_tr.pt")).to("cuda:0")
        self._samples_te = torch.load(os.path.join(dataset_directory, "samples_te.pt")).to("cuda:0")
        self._endpoints_te = torch.load(os.path.join(dataset_directory, "endpoints_te.pt")).to("cuda:0")

        self._sum_joint_limit_range = get_sum_joint_limit_range(self._samples_tr)

    def log_dataset_sizes(self, epoch=0, batch_nb=0):
        """Log the training and testset size to wandb"""
        assert self._samples_tr.shape[0] == self._endpoints_tr.shape[0]
        assert self._samples_te.shape[0] == self._endpoints_te.shape[0]
        wandb.log(
            {
                "epoch": epoch,
                "batch_nb": batch_nb,
                "small_dataset": self._use_small_dataset,
                "sum_joint_limit_range": self._sum_joint_limit_range,
                "dataset_size_tr": self._samples_tr.shape[0],
                "dataset_size_te": self._samples_te.shape[0],
            }
        )

    def train_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(self._samples_tr, self._endpoints_tr),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(
                self._samples_te[0 : self._val_set_size], self._endpoints_te[0 : self._val_set_size]
            ),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )
