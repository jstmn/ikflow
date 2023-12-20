from typing import Tuple, Dict, List
from time import time

from jrl.robots import Fetch
import wandb
import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule

from ikflow.training.training_utils import get_softflow_noise
from ikflow import config
from ikflow.config import SIGMOID_SCALING_ABS_MAX
from ikflow.ikflow_solver import IKFlowSolver, draw_latent
from ikflow.model import IkflowModelParameters
from ikflow.utils import grad_stats
from ikflow.evaluation_utils import evaluate_solutions
from ikflow.thirdparty.ranger import RangerVA  # from ranger913A.py

zeros_noise_scale = 0.0001

# TODO: the pose for Fetch doesn't look right
ROBOT_TARGET_POSE_POINTS_FOR_PLOTTING = {Fetch.name: [np.array([0, 1.5707, 0, 0, 0, 3.141592, 0])]}

_IK_SOLUTION_TABLE_COLUMNS = ["global_step", "target_pose", "solution", "realized_pose", "error", "joint_types"]
ik_solution_table = wandb.Table(data=[], columns=_IK_SOLUTION_TABLE_COLUMNS)


class IkfLitModel(LightningModule):
    def __init__(
        self,
        ik_solver: IKFlowSolver,
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
        sigmoid_on_output: bool = False,
    ):
        # `learning_rate`, `gamma`, `samples_per_pose`, etc. saved to self.hparams

        super().__init__()

        self.ik_solver = ik_solver
        self.nn_model = ik_solver.nn_model
        self.nn_model.to(config.device)
        self.base_hparams = base_hparams
        self.ndof = self.ik_solver.robot.ndof
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
                all the sudden your learning processes is altered. Therefor dataset size becomes another hyperparameter,
                which adds extra complexity into hyperparameter tuning. For example, if you change your dataset size and
                your training improves, is that because you have more data to learn from? Or because your learning rate 
                is decreasing at a slower rate?! 

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
        y = y.to(config.device)
        x = x.to(config.device)

        batch_size = y.shape[0]
        if self.dim_tot > self.ndof:
            pad_x = 0.001 * torch.randn((batch_size, self.dim_tot - self.ndof)).to(config.device)

            # padding must be in (-SIGMOID_SCALING_ABS_MAX, SIGMOID_SCALING_ABS_MAX). This value will be scaled to these
            # bounds and then passed through inverse sigmoid. If they are outside of these bounds, inverse-sigmoid will
            # return NaNs and everything will explode. And then I will cry tears of sadness.
            if hasattr(self.base_hparams, "sigmoid_on_output") and self.base_hparams.sigmoid_on_output:
                eps = 1e-5
                pad_x = torch.clamp(pad_x, -SIGMOID_SCALING_ABS_MAX + eps, SIGMOID_SCALING_ABS_MAX - eps)

            x = torch.cat([x, pad_x], dim=1)

        # Add softflow noise
        if self.base_hparams.softflow_enabled:
            c, v = get_softflow_noise(x, self.base_hparams.softflow_noise_scale)
            x = x + v
            conditional = torch.cat([y, c], dim=1)
        else:
            conditional = y

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
            time_per_batch = time() - t0
            log_data = {
                "tr/loss": loss.item(),
                "tr/time_p_batch": time_per_batch,
                "tr/batches_p_sec": 1.0 / time_per_batch,
                "tr/learning_rate": self.get_lr(),
            }
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
            self.safe_log_metrics({
                "tr/grad_ave": ave_grad,
                "tr/grad_abs_ave": ave_abs_grad,
                "tr/grad_max": max_grad,
            })

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        del batch_idx
        _, y = batch
        ee_pose_target = y.cpu().detach().numpy()[0]
        solutions, model_runtime = self.solve(ee_pose_target, self.hparams.samples_per_pose, return_runtime=True)
        l2_errors, ang_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
            self.ik_solver.robot, ee_pose_target, solutions
        )
        solutions_clamped = self.ik_solver.robot.clamp_to_joint_limits(solutions)
        (
            l2_errors_clamped,
            ang_errors_clamped,
            joint_limits_exceeded_clamped,
            self_collisions_clamped,
        ) = evaluate_solutions(self.ik_solver.robot, ee_pose_target, solutions_clamped)
        result = {
            "model_runtime": model_runtime,
        }
        result["non_clamped"] = {
            # TODO: Fix this hackiness. evaluate_solutions() should return all of one type
            "l2_errs": torch.tensor(l2_errors, dtype=torch.float32),
            "angular_errs": torch.tensor(ang_errors, dtype=torch.float32),
            "joint_limits_exceeded": joint_limits_exceeded,
            "self_collisions": self_collisions,
        }
        result["clamped"] = {
            # TODO: Fix this hackiness. evaluate_solutions() should return all of one type
            "l2_errs": torch.tensor(l2_errors_clamped, dtype=torch.float32),
            "angular_errs": torch.tensor(ang_errors_clamped, dtype=torch.float32),
            "joint_limits_exceeded": joint_limits_exceeded_clamped,
            "self_collisions": self_collisions_clamped,
        }
        return result

    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, torch.Tensor]]):
        """_summary_

        Args:
            validation_step_outputs (_type_): _description_
        """
        results_clamped = [result["clamped"] for result in validation_step_outputs]
        results_non_clamped = [result["non_clamped"] for result in validation_step_outputs]
        model_runtimes = [pred["model_runtime"] for pred in validation_step_outputs]
        metrics = {"val/ave_model_runtime": np.mean(model_runtimes)}

        def get_stats(_results):
            n_total = len(_results) * len(_results[0]["l2_errs"])
            l2_errs = torch.cat([pred["l2_errs"] for pred in _results])
            max_l2_errs = [pred["l2_errs"].max().item() for pred in _results]
            angular_errs = torch.cat([pred["angular_errs"] for pred in _results])
            max_angular_errs = [pred["angular_errs"].max().item() for pred in _results]
            joint_limits_exceeded = torch.cat([pred["joint_limits_exceeded"] for pred in _results])
            self_collisions = torch.cat([pred["self_collisions"] for pred in _results])
            return (
                l2_errs.mean().item(),
                l2_errs.std().item(),
                np.mean(max_l2_errs),
                np.std(max_l2_errs),
                angular_errs.mean().item(),
                angular_errs.mean().item(),
                np.mean(max_angular_errs),
                np.std(max_angular_errs),
                100 * (joint_limits_exceeded.sum().item() / n_total),
                100 * (self_collisions.sum().item() / n_total),
            )

        for prefix, results in zip(["val", "val_clamped"], [results_non_clamped, results_clamped]):
            (
                l2_error,
                l2_error_std,
                l2_ave_max_error,
                l2_ave_max_error_std,
                angular_error,
                angular_error_std,
                angular_ave_max_error,
                angular_ave_max_error_std,
                joint_limits_exceeded,
                self_collisions,
            ) = get_stats(results)
            metrics[f"{prefix}/l2_error"] = l2_error
            metrics[f"{prefix}/l2_error_std"] = l2_error_std
            metrics[f"{prefix}/l2_ave_max_error"] = l2_ave_max_error
            metrics[f"{prefix}/l2_ave_max_error_std"] = l2_ave_max_error_std
            metrics[f"{prefix}/angular_error"] = angular_error
            metrics[f"{prefix}/angular_error_std"] = angular_error_std
            metrics[f"{prefix}/angular_ave_max_error"] = angular_ave_max_error
            metrics[f"{prefix}/angular_ave_max_error_std"] = angular_ave_max_error_std
            metrics[f"{prefix}/joint_limits_exceeded"] = joint_limits_exceeded
            metrics[f"{prefix}/self_collisions"] = self_collisions

        self.safe_log_metrics(metrics)

        # B/c pytorch lightning is dumb (https://github.com/Lightning-AI/lightning/issues/12724)
        # self.global_step converted to float because of the following warning:
        #   "UserWarning: You called `self.log('global_step', ...)` in your `validation_epoch_end` but the value needs to be floating point. Converting it to torch.float32."
        self.log("global_step", float(self.global_step))

        # Ignoring table logging for now
        # self.log_ik_solutions_to_wandb_table()

    def solve(self, y: Tuple[float], m: int, return_runtime: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y
                y: endpose [x, y, z, q0, q1, q2, q3]
                m: Number of samples
        """
        assert len(y) == 7

        # Note: No code change required here to handle using/not using softflow.
        conditional = torch.zeros(m, self.ik_solver.dim_cond)
        conditional[:, 0:3] = torch.FloatTensor(y[:3])
        conditional[:, 3 : 3 + 4] = torch.FloatTensor(np.array([y[3:]]))
        conditional = conditional.to(config.device)

        shape = (m, self.dim_tot)
        latent = draw_latent(None, "gaussian", 1, shape)
        assert latent.shape[0] == m
        assert latent.shape[1] == self.dim_tot

        t0 = time()
        output_rev, _ = self.nn_model(latent, c=conditional, rev=True)
        if return_runtime:
            return output_rev[:, 0 : self.ndof], time() - t0
        return output_rev[:, 0 : self.ndof]
