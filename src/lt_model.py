from datetime import datetime
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
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

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
    ):
        # `learning_rate`, `gamma`, `samples_per_pose`, etc. saved to self.hparams

        super().__init__()

        self.ik_solver = ik_solver
        self.nn_model = ik_solver.nn_model
        self.base_hparams = base_hparams
        self.dim_x = self.ik_solver.robot_model.dim_x
        self.dim_tot = self.base_hparams.dim_latent_space
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir(self.ik_solver.robot.name)
        if self.checkpoint_every > 0:
            safe_mkdir(self.checkpoint_dir)
        self.log_every = log_every

        self.save_hyperparameters(ignore=["ik_solver"])

    def _checkpoint(self):
        filename = f"epoch:{self.current_epoch}_batch:{self.global_step}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "nn_model_type": self.ik_solver.nn_model_type,
            "model": self.ik_solver.nn_model.state_dict(),
            "robot_model": self.ik_solver.robot_model.name,
            "hparams": non_private_dict(self.hparams.__dict__),
            "epoch": self.current_epoch,
            "batch_nb": self.global_step,
        }
        torch.save(state, filepath)

        # Upload to wandb
        if self.logger is not None:
            artifact_name = wandb.run.name
            artifact = wandb.Artifact(artifact_name, type="model", description="checkpoint")
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
            artifact.wait()

    def safe_log_metrics(self, vals: Dict):
        assert isinstance(vals, dict)
        try:
            self.logger.log_metrics(vals, step=self.global_step)
        except AttributeError:
            pass

    def ml_loss_fn(self, batch):
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
            print("loss is nan")
            print("output:")
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

    def training_step(self, batch, batch_idx):
        del batch_idx
        t0 = time()
        loss, loss_data, force_log = self.ml_loss_fn(batch)

        if (self.global_step % self.log_every == 0 and self.global_step > 0) or force_log:
            log_data = {"tr/loss": loss.item(), "tr/time_p_batch": time() - t0}
            log_data.update(loss_data)
            self.safe_log_metrics(log_data)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        if self.checkpoint_every > 0 and self.global_step % self.checkpoint_every == 0 and self.global_step > 0:
            self._checkpoint()

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.nn_model.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [lr_scheduler]

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
        conditional[:, 3 : 3 + 4] = torch.FloatTensor([y[3:]])
        # (:, 7: 7 + ndim_tot) is the sigma of the normal distribution that noise is drawn from and added to x. 0 in this case
        conditional = conditional.to(config.device)

        shape = (m, self.dim_tot)
        latent_noise = draw_latent_noise(None, "gaussian", 1, shape)
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.dim_tot

        t0 = time()
        output_rev, jac = self.nn_model(latent_noise, c=conditional, rev=True)
        return output_rev[:, 0 : self.dim_x], time() - t0
