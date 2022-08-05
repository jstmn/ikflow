from datetime import datetime
import os
from typing import Tuple, Dict
from time import time, sleep

from src import config
from src.model import IkflowModel, draw_latent_noise
from src.training_parameters import IkflowModelParameters
from src.math_utils import rotation_matrix_from_quaternion, geodesic_distance
from src.utils import grad_stats, non_private_dict, safe_mkdir

import wandb
import numpy as np
import torch
from pytorch_lightning.core.lightning import LightningModule

zeros_noise_scale = 0.0001
device = "cuda"


CHECKPOINT_BASE_DIR = "~/Projects/data/ikflow/training_checkpoints/"



def _datetime_str() -> str:
    now = datetime.now()
    return now.strftime("%b%d_%Y_%I:%-M:%p")


def checkpoint_dir(verbose=True) -> str:
    if wandb.run is not None:
        ckpt_dir_name = f"{_datetime_str()}__{wandb.run.name}/"
    else:
        ckpt_dir_name = f"{_datetime_str()}__None/"
    dir_filepath = os.path.join(CHECKPOINT_BASE_DIR, ckpt_dir_name)
    if verbose:
        print(f"checkpoint_dir(): Directory: '{dir_filepath}'")
    return dir_filepath


class IkfLitModel(LightningModule):

    AVAILABLE_LOSSES = ["l2", "l2+ml", "ml", "l2-multi"]

    def __init__(
        self,
        model_wrapper: IkflowModel,
        base_hparams: IkflowModelParameters,
        loss_fn: str,
        learning_rate: float,
        checkpoint_every: int,
        gamma: float = 0.975,
        samples_per_pose: int = 100,
        log_every: int = 1e10,
        gradient_clip: float = float("inf"),
        lambd: float = 1,
    ):
        assert loss_fn in IkfLitModel.AVAILABLE_LOSSES
        # `learning_rate`, `gamma`, `samples_per_pose`, etc. saved to self.hparams

        super().__init__()

        self.model_wrapper = model_wrapper
        self.nn_model = model_wrapper.nn_model
        self.base_hparams = base_hparams
        self.dim_x = self.model_wrapper.robot_model.dim_x
        self.dim_tot = self.base_hparams.dim_latent_space
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir(verbose=self.checkpoint_every > 0)
        if self.checkpoint_every > 0:
            safe_mkdir(self.checkpoint_dir)
        self.log_every = log_every

        self.save_hyperparameters(ignore=["model_wrapper"])

    def _checkpoint(self):
        filename = f"training_checkpoint__epoch:{self.current_epoch}__batch:{self.global_step}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "nn_model_type": self.model_wrapper.nn_model_type,
            "model": self.model_wrapper.nn_model.state_dict(),
            "robot_model": self.model_wrapper.robot_model.name,
            "hparams": non_private_dict(self.hparams.__dict__),
            "epoch": self.current_epoch,
            "batch_nb": self.global_step,
        }
        torch.save(state, filepath)

        # Upload to wandb
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
        zz = torch.sum(output ** 2, dim=1)
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

    def l2_plus_ml_loss_fn(self, batch):
        x, y = batch
        y = y.to(device)
        x = x.to(device)

        batch_size = y.shape[0]
        if self.dim_tot > self.dim_x:
            pad_x = 0.001 * torch.randn((batch_size, self.dim_tot - self.dim_x)).to(device)
            x = torch.cat([x, pad_x], dim=1)

        conditional = torch.cat([y, torch.zeros((batch_size, 1)).to(device)], dim=1)

        # MSE
        latent_noise = torch.randn((batch_size, self.dim_tot)).to(device)
        samples, _ = self.nn_model(latent_noise, c=conditional, rev=True)
        samples_fk_t, _ = self.model_wrapper.robot_model.forward_kinematics_batch(samples)
        loss_mse = torch.mean(torch.pow(samples_fk_t - y[:, 0:3], 2))
        #         Note: Whenever the computation graph connects through batch_fk to `loss`, nans are ultimately thrown (hypothesis based on evidence)
        #         Result: Loss curves are different when using samples = torch.zeros(5), vs. setting lambda to 0

        output, jac = self.nn_model.forward(x, c=conditional, jac=True)
        zz = torch.sum(output ** 2, dim=1)
        neg_log_likeli = 0.5 * zz - jac
        loss_ml = torch.mean(neg_log_likeli)

        loss = self.hparams.lambd * loss_mse + ((1 - self.hparams.lambd) * loss_ml)

        loss_is_nan = torch.isnan(loss)
        if loss_is_nan:
            print("loss is nan")
            for i in range(samples_fk_t.shape[0]):
                if torch.any(torch.isnan(samples_fk_t[i, :])):
                    print("\nsamples[i]:     ", samples[i].data)
                    print("\nsamples_fk_t[i]:", samples_fk_t[i].data)
                    print("\nx[i]:           ", x[i].data)
                    print("\ny[i]:           ", y[i].data)
                    print("\ny[i] - sfk:     ", (y[i, 0:3] - samples_fk_t[i]).data)
        loss_data = {
            "tr/samples_max": torch.max(samples).item(),
            "tr/samples_abs_ave": torch.mean(torch.abs(samples)).item(),
            "tr/samples_ave": torch.mean(samples).item(),
            "tr/samples_std": torch.std(samples).item(),
            "tr/loss_is_nan": int(loss_is_nan),
            "tr/loss_ml_scaled": (1 - self.hparams.lambd) * loss_ml.item(),
            "tr/loss_ml": loss_ml.item(),
            "tr/loss_mse_scaled": self.hparams.lambd * loss_mse.item(),
            "tr/loss_mse": loss_mse.item(),
        }
        force_log = loss_is_nan
        return loss, loss_data, force_log

    def l2_loss_fn(self, batch):
        x, y = batch
        batch_size = y.shape[0]
        y = y.to(device)

        conditional = torch.cat([y, torch.zeros((batch_size, 1)).to(device)], dim=1)
        latent_noise = torch.randn((batch_size, self.dim_tot)).to(device)
        samples, _ = self.nn_model(latent_noise, c=conditional, rev=True)
        samples_fk_t, _ = self.model_wrapper.robot_model.forward_kinematics_batch(samples)

        # MSE
        loss = torch.mean(torch.pow(samples_fk_t - y[:, 0:3], 2))
        loss_is_nan = torch.isnan(loss)
        if loss_is_nan:
            print("loss is nan")
            for i in range(samples_fk_t.shape[0]):
                if torch.any(torch.isnan(samples_fk_t[i, :])):
                    print("\nsamples[i]:     ", samples[i])
                    print("\nsamples_fk_t[i]:", samples_fk_t[i])
                    print("\nx[i]:           ", x[i])
                    print("\ny[i]:           ", y[i])
                    print("\ny[i] - sfk:     ", y[i, 0:3] - samples_fk_t[i])
        loss_data = {
            "tr/samples_max": torch.max(samples).item(),
            "tr/samples_abs_ave": torch.mean(torch.abs(samples)).item(),
            "tr/samples_ave": torch.mean(samples).item(),
            "tr/samples_std": torch.std(samples).item(),
            "tr/loss_is_nan": int(loss_is_nan),
        }
        force_log = loss_is_nan
        return loss, loss_data, force_log

    def l2_multi_loss_fn(self, batch):
        x, y_full = batch
        n_poses = 1 + len(self.model_wrapper.robot_model.l2_loss_pts)
        assert y_full.shape[1] == 7 * n_poses

        batch_size = y_full.shape[0]
        y_full = y_full.to(device)
        y = y_full[:, 0:7]

        y_just_translations = torch.zeros((batch_size, 3 * n_poses)).to(device)

        for i in range(n_poses):
            base_idx_y_full = i * 7
            t_i = y_full[:, base_idx_y_full : base_idx_y_full + 3]
            base_idx_y_just_t = i * 3
            y_just_translations[:, base_idx_y_just_t : base_idx_y_just_t + 3] = t_i

        # Sample from model
        conditional = torch.cat([y, torch.zeros((batch_size, 1)).to(device)], dim=1)
        latent_noise = torch.randn((batch_size, self.dim_tot)).to(device)
        samples, _ = self.nn_model(latent_noise, c=conditional, rev=True)

        # FK on samples
        samples_fk_t = self.model_wrapper.robot_model.forward_kinematics_batch(samples, with_l2_loss_pts=True)

        # MSE
        loss = torch.mean(torch.pow(samples_fk_t - y_just_translations, 2))

        # Log info
        loss_is_nan = torch.isnan(loss)
        if loss_is_nan:
            print("loss is nan")
            for i in range(samples_fk_t.shape[0]):
                if torch.any(torch.isnan(samples_fk_t[i, :])):
                    print("\nsamples[i]:     ", samples[i])
                    print("\nsamples_fk_t[i]:", samples_fk_t[i])
                    print("\nx[i]:           ", x[i])
                    print("\ny[i]:           ", y[i])
                    print("\ny[i] - sfk:     ", y[i, 0:3] - samples_fk_t[i])
        loss_data = {
            "tr/samples_max": torch.max(samples).item(),
            "tr/samples_abs_ave": torch.mean(torch.abs(samples)).item(),
            "tr/samples_ave": torch.mean(samples).item(),
            "tr/samples_std": torch.std(samples).item(),
            "tr/loss_is_nan": int(loss_is_nan),
        }
        force_log = loss_is_nan
        return loss, loss_data, force_log

    def training_step(self, batch, batch_idx):
        del batch_idx
        t0 = time()
        if self.hparams.loss_fn == "l2":
            loss, loss_data, force_log = self.l2_loss_fn(batch)
        elif self.hparams.loss_fn == "l2-multi":
            loss, loss_data, force_log = self.l2_multi_loss_fn(batch)
        elif self.hparams.loss_fn == "l2+ml":
            loss, loss_data, force_log = self.l2_plus_ml_loss_fn(batch)
        elif self.hparams.loss_fn == "ml":
            loss, loss_data, force_log = self.ml_loss_fn(batch)
        else:
            raise ValueError("I shouldn't be here")

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
        if self.hparams.loss_fn == "l2-multi":
            y = y[:, 0:7]
        ee_pose_target = y.cpu().detach().numpy()[0]
        # TODO(@jeremysm): Move this error calculation to evaluation.py
        samples, model_runtime = self.make_samples(ee_pose_target, self.hparams.samples_per_pose)
        ee_pose_ikflow = self.model_wrapper.robot_model.forward_kinematics(
            samples[:, 0 : self.model_wrapper.robot_model.dim_x].cpu().detach().numpy()
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
        conditional = torch.zeros(m, self.model_wrapper.dim_cond)
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
