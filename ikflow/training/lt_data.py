import os
from typing import List, Dict

from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
import wandb
import torch

from ikflow.config import ALL_DATASET_TAGS, device
from ikflow.utils import get_sum_joint_limit_range, get_dataset_directory, get_dataset_filepaths


class IkfLitDataset(LightningDataModule):
    def __init__(
        self, robot_name: str, batch_size: int, val_set_size: int, dataset_tags: List[str], prepare_data_per_node=True
    ):
        for tag in dataset_tags:
            assert tag in ALL_DATASET_TAGS

        self._robot_name = robot_name
        self._batch_size = batch_size
        self._val_set_size = val_set_size

        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node. If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        self.prepare_data_per_node = prepare_data_per_node
        self._log_hyperparams = True

        dataset_directory = get_dataset_directory(self._robot_name)
        assert os.path.isdir(dataset_directory), (
            f"Directory '{dataset_directory}' doesn't exist - have you created the dataset for this robot yet? (try"
            f" `python scripts/build_dataset.py --robot_name={robot_name} --training_set_size=10000000`)"
        )
        samples_tr_file_path, poses_tr_file_path, samples_te_file_path, poses_te_file_path, _ = get_dataset_filepaths(
            dataset_directory, dataset_tags
        )
        self._samples_tr = torch.load(samples_tr_file_path).to(device)
        self._endpoints_tr = torch.load(poses_tr_file_path).to(device)
        self._samples_te = torch.load(samples_te_file_path).to(device)
        self._endpoints_te = torch.load(poses_te_file_path).to(device)

        self._sum_joint_limit_range = get_sum_joint_limit_range(self._samples_tr)

    def add_dataset_hashes_to_cfg(self, cfg: Dict):
        cfg.update(
            {
                "dataset_hashes": str(
                    [
                        self._samples_tr.sum().item(),
                        self._endpoints_tr.sum().item(),
                        self._samples_te.sum().item(),
                        self._endpoints_te.sum().item(),
                    ]
                )
            }
        )

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
            # see https://github.com/dbolya/yolact/issues/664#issuecomment-975051339
            generator=torch.Generator(device=device),
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
