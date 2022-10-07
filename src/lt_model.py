from datetime import datetime
from multiprocessing.sharedctypes import Value
import os
from typing import Tuple, Dict, Optional
from time import time, sleep

import config
from src.ik_solvers import IkflowSolver, draw_latent_noise
from src.training_parameters import IkflowModelParameters
from src.math_utils import rotation_matrix_from_quaternion, geodesic_distance
from src.utils import grad_stats, non_private_dict, safe_mkdir

import wandb
import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule

from thirdparty.ranger import RangerVA  # from ranger913A.py

zeros_noise_scale = 0.0001
device = "cuda"


def _datetime_str() -> str:
    now = datetime.now()
    return now.strftime("%b.%d.%Y_%I:%-M:%p")


def checkpoint_dir(robot_name: str) -> str:
    if wandb.run is not None:
        ckpt_dir_name = f"{robot_name}--{_datetime_str()}--wandb-run-name:{wandb.run.name}/"
    else:
        ckpt_dir_name = f"{robot_name}--{_datetime_str()}/"
    dir_filepath = os.path.join(config.TRAINING_LOGS_DIRECTORY, ckpt_dir_name)
    return dir_filepath


class IkfLitModel(LightningModule):
    def __init__(
        self,
        ik_solver: IkflowSolver,
        base_hparams: IkflowModelParameters,
        learning_rate: float,
        checkpoint_every: int,
        gamma: float = 0.975,
        samples_per_pose: int = 100,
        log_every: int = 1e10,
        gradient_clip: float = float("inf"),
        lambd: float = 1,
        step_lr_every: int = int(int(2.5 * 1e6) / 64),  # #batches to see 2.5million datapoints w/ batch size=64
        weight_decay: float = 1.8e-05,
        optimizer_name: str = "ranger",
    ):
        # `learning_rate`, `gamma`, `samples_per_pose`, etc. saved to self.hparams

        super().__init__()

        self.ik_solver = ik_solver
        self.nn_model = ik_solver.nn_model
        self.base_hparams = base_hparams
        self.dim_x = self.ik_solver.robot_model.dim_x
        self.dim_tot = self.base_hparams.dim_latent_space
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        self.save_hyperparameters(ignore=["ik_solver"])

        # The Ranger optimizer is incompatable with pytorch's calling method so we need to manually call the
        # optimization steps (also need to call .step() on the learning rate scheduler)
        if self.hparams.optimizer_name == "ranger":
            # Important: This property activates manual optimization.
            self.automatic_optimization = False

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler"""
        if self.hparams.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(
                self.nn_model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_name == "ranger":
            optimizer = RangerVA(
                self.nn_model.parameters(),
                lr=self.hparams.learning_rate,
                betas=(0.95, 0.999),
                eps=1e-04,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.nn_model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: '{self.hparams.optimizer_name}'")

        """
        Create a learning rate scheduler and configure it to update every k optimization steps instead of k epochs. 
        
        Quick rant: updating the lr scheduler based on epochs is a pain and dumb and bad engineering practice.
            Complaint #1: It makes the learning rate decay dependent on your dataset size. If you change your dataset 
                all the sudden your learning processes is fundamentally altered. Dataset size will always be a 
                hyperparameter of sorts, but if also is a key factor in your learning rate decay now you've added extra 
                complexity into hyperparameter tuning. For example, if you change your dataset size and your training 
                improves, is that because you have more data to learn from? Or because your learning rate is decreasing 
                more slowly?! 

            Complaint #2: Why do we define an epoch at all when our data is continuously being drawn uniformly at 
                random from some distribution (D)? To be fair there is one dataset file in this project. But if I have 
                10 million poses that are being sampled randomly then that is approximately equal to random sampling 
                from the distribution. Its a meaningless distinction to choose a fixed number and call a set with size 
                of that fixed number of randomly drawn samples from D an epoch. The bottom line is that what we care 
                about in training is our generalization error. Epoch is an unneccessary construct.   
        """
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_lr_every, gamma=self.hparams.gamma, verbose=False
        )

        # See 'configure_optimizers' in these docs to see the format of this dict: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, either 'step' or 'epoch'.
            # 'epoch' updates the scheduler on epoch end whereas 'step' updates it after a optimizer update.
            "interval": "step",
        }
        return [optimizer], [lr_scheduler_config]

    def safe_log_metrics(self, vals: Dict):
        assert isinstance(vals, dict)
        try:
            self.logger.log_metrics(vals, step=self.global_step)
        except AttributeError:
            pass

    def ml_loss_fn(self, batch):
        """Maximum likelihood loss"""
        x, y = batch
        y = y.to(device)
        x = x.to(device)

        batch_size = y.shape[0]
        if self.dim_tot > self.dim_x:
            pad_x = 0.001 * torch.randn((batch_size, self.dim_tot - self.dim_x)).to(device)
            x = torch.cat([x, pad_x], dim=1)

        conditional = torch.cat([y, torch.zeros((batch_size, 1)).to(device)], dim=1)

        output, jac = self.nn_model.forward(x, c=conditional, jac=True)
        zz = torch.sum(output**2, dim=1)
        neg_log_likeli = 0.5 * zz - jac
        loss = torch.mean(neg_log_likeli)

        loss_is_nan = torch.isnan(loss)
        if loss_is_nan:
            print("loss is Nan\n output:")
            print(output)

        loss_data = {
            "tr/output_max": torch.max(output).item(),
            "tr/output_abs_ave": torch.mean(torch.abs(output)).item(),
            "tr/output_ave": torch.mean(output).item(),
            "tr/output_std": torch.std(output).item(),
            "tr/loss_is_nan": int(loss_is_nan),
            "tr/loss_ml": loss.item(),
        }
        force_log = loss_is_nan
        return loss, loss_data, force_log

    def get_lr(self) -> float:
        """Returns the current learning rate"""
        optimizer = self.optimizers().optimizer
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group["lr"])
        assert len(set(lrs)) == 1, f"Error: Multiple learning rates found. There should only be one. lrs: '{lrs}'"
        return lrs[0]

    def training_step(self, batch, batch_idx):
        del batch_idx
        t0 = time()
        loss, loss_data, force_log = self.ml_loss_fn(batch)

        if (self.global_step % self.log_every == 0 and self.global_step > 0) or force_log:
            log_data = {"tr/loss": loss.item(), "tr/time_p_batch": time() - t0, "tr/learning_rate": self.get_lr()}
            log_data.update(loss_data)
            self.safe_log_metrics(log_data)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        # The `step()` function of the Ranger optimizer doesn't follow the convention pytorch lightning expects.
        # Because of this we manually perform the optimization step. See https://stackoverflow.com/a/73267631/5191069
        if self.hparams.optimizer_name == "ranger":
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()

            optimizer.zero_grad()
            # TODO: Verify `manual_backward()` works as well.
            # self.manual_backward(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss

    def on_after_backward(self):
        torch.nn.utils.clip_grad_value_(self.parameters(), self.hparams.gradient_clip)

        if self.trainer.global_step % self.log_every == 0:
            ave_grad, ave_abs_grad, max_grad = grad_stats(self.parameters())
            self.safe_log_metrics(
                {
                    "tr/grad_ave": ave_grad,
                    "tr/grad_abs_ave": ave_abs_grad,
                    "tr/grad_max": max_grad,
                }
            )

    def validation_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        ee_pose_target = y.cpu().detach().numpy()[0]
        # TODO(@jeremysm): Move this error calculation to evaluation.py
        samples, model_runtime = self.make_samples(ee_pose_target, self.hparams.samples_per_pose)
        ee_pose_ikflow = self.ik_solver.robot_model.forward_kinematics(
            samples[:, 0 : self.ik_solver.robot_model.dim_x].cpu().detach().numpy()
        )
        # Positional Error
        pos_l2errs = np.linalg.norm(ee_pose_ikflow[:, 0:3] - ee_pose_target[0:3], axis=1)
        # Angular Error
        rot_target = np.tile(ee_pose_target[3:], (self.hparams.samples_per_pose, 1))
        rot_output = ee_pose_ikflow[:, 3:]

        output_R9 = rotation_matrix_from_quaternion(torch.Tensor(rot_output).to(device))
        target_R9 = rotation_matrix_from_quaternion(torch.Tensor(rot_target).to(device))
        angular_rad_errs = geodesic_distance(target_R9, output_R9).cpu().data.numpy()

        return {"l2_errs": pos_l2errs, "angular_errs": angular_rad_errs, "model_runtime": model_runtime}

    def validation_epoch_end(self, validation_step_outputs):
        spp = self.hparams.samples_per_pose
        n_val_steps = len(validation_step_outputs)
        l2_errs = np.zeros(n_val_steps * spp)
        max_l2_errs = np.zeros(n_val_steps)
        angular_errs = np.zeros(n_val_steps * spp)
        max_angular_errs = np.zeros(n_val_steps)
        model_runtimes = np.zeros(n_val_steps)

        for i, pred in enumerate(validation_step_outputs):
            l2_errs[i * spp : i * spp + spp] = pred["l2_errs"]
            max_l2_errs[i] = max(pred["l2_errs"])
            angular_errs[i * spp : i * spp + spp] = pred["angular_errs"]
            max_angular_errs[i] = max(pred["angular_errs"])
            model_runtimes[i] = pred["model_runtime"]

        self.safe_log_metrics(
            {
                "val/l2_error": np.mean(l2_errs),
                "val/l2_error_std": np.std(l2_errs),
                "val/l2_ave_max_error": np.mean(max_l2_errs),
                "val/l2_ave_max_error_std": np.std(max_l2_errs),
                "val/angular_error": np.mean(angular_errs),
                "val/angular_error_std": np.std(angular_errs),
                "val/angular_ave_max_error": np.mean(max_angular_errs),
                "val/angular_ave_max_error_std": np.std(max_angular_errs),
                "val/ave_model_runtime": np.mean(model_runtimes),
            }
        )

        # B/c pytorch lightning is dumb (https://github.com/Lightning-AI/lightning/issues/12724)
        self.log("global_step", self.global_step)

    def make_samples(self, y: Tuple[float], m: int) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y
                y: endpose [x, y, z, q0, q1, q2, q3]
                m: Number of samples
        """
        assert len(y) == 7

        # Note: No code change required here to handle using/not using softflow.
        conditional = torch.zeros(m, self.ik_solver.dim_cond)
        # (:, 0:3) is x, y, z
        conditional[:, 0:3] = torch.FloatTensor(y[:3])
        # (:, 3:7) is quat
        conditional[:, 3 : 3 + 4] = torch.FloatTensor(np.array([y[3:]]))
        # (:, 7: 7 + ndim_tot) is the sigma of the normal distribution that noise is drawn from and added to x. 0 in this case
        conditional = conditional.to(config.device)

        shape = (m, self.dim_tot)
        latent_noise = draw_latent_noise(None, "gaussian", 1, shape)
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.dim_tot

        t0 = time()
        output_rev, jac = self.nn_model(latent_noise, c=conditional, rev=True)
        return output_rev[:, 0 : self.dim_x], time() - t0
