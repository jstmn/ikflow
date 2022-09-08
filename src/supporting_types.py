from typing import Dict, List, Optional
import numpy as np
import torch
from math import inf
from time import time


class Config:
    """Convenience class that overwrites the default print function"""

    ROUND_AMT = 7

    def __init__(self, class_name: str):
        self.class_name = class_name

    def __str__(self) -> str:
        ret_str = f"{self.class_name}()\n"
        longest_key_len = len(max([key for key in self.__dict__], key=len))
        for key in self.__dict__:
            n_spaces = longest_key_len - len(key) + 1
            val = self.__dict__[key]
            if isinstance(val, float):
                ret_str += f"  {key}:{n_spaces*' '}{round(val, Config.ROUND_AMT)}\n"
            elif isinstance(val, int):
                ret_str += f"  {key}:{n_spaces*' '}{val}\n"
            elif isinstance(val, str):
                ret_str += f"  {key}:{n_spaces*' '}'{val}'\n"
            else:
                ret_str += f"  {key}:{n_spaces*' '} {type(val)} {val}\n"
        return ret_str


class EvaluationSettings(Config):
    def __init__(
        self,
        eval_enabled: bool = True,
        n_poses_per_evaluation: int = 750,
        n_samples_per_pose: int = 20,
    ):
        """This class configures how a model is evaluated

        Args:
            eval_enabled (bool): Flag for enabling evaluation during training
            n_poses_per_evaluation (int): Sets the number of target poses that solutions are generated for during an evaluation
            n_samples_per_pose (int): Sets the number of solutions for each target pose
        """
        assert isinstance(eval_enabled, bool)
        assert isinstance(n_poses_per_evaluation, int)
        assert isinstance(n_samples_per_pose, int)
        self.eval_enabled = eval_enabled
        self.n_poses_per_evaluation = n_poses_per_evaluation
        self.n_samples_per_pose = n_samples_per_pose

        Config.__init__(self, "EvaluationSettings")


class LoggingSettings(Config):
    def __init__(
        self,
        wandb_enabled: bool = False,
        eval_every_k: int = 25000,
        log_loss_every_k: int = 5000,
        log_dist_plot_every: int = 39062,
    ):
        assert isinstance(wandb_enabled, bool)
        assert isinstance(eval_every_k, int)
        assert isinstance(log_loss_every_k, int)
        assert isinstance(log_dist_plot_every, int)
        self.wandb_enabled = wandb_enabled
        self.eval_every_k = eval_every_k
        self.log_loss_every_k = log_loss_every_k
        self.log_dist_plot_every = log_dist_plot_every
        Config.__init__(self, "LoggingSettings")


class SaveSettings(Config):
    def __init__(
        self,
        save_model: bool = True,
        save_to_cpu: bool = True,
        save_if_mean_l2_error_below: float = inf,
        save_if_mean_angular_error_below: float = inf,
    ):
        assert isinstance(save_model, bool)
        assert isinstance(save_to_cpu, bool)
        assert isinstance(save_if_mean_l2_error_below, float)
        assert isinstance(save_if_mean_angular_error_below, float)
        self.save_model = save_model
        self.save_to_cpu = save_to_cpu
        self.save_if_mean_l2_error_below = save_if_mean_l2_error_below
        self.save_if_mean_angular_error_below = save_if_mean_angular_error_below
        Config.__init__(self, "SaveSettings")


class TrainingResult(Config):
    """Store results from src"""

    def __init__(self):
        self.model_saved = False
        self.nan_during_training = False
        self.wandb_run_name = None
        self.n_epochs = None
        self.tr_time_minutes = None
        self.keyboard_interupt_thrown = False
        Config.__init__(self, "TrainingResult")


class TrainingStepResult:
    def __init__(self):
        self.l_fit = 0.0
        self.runtime = 0.0
        self.lr = 0.0
        self.max_grad = 0.0
        self.ave_grad = 0.0
        self.f_output = None
        self.rev_output = None
        self.nan_in_loss = False

    def get_dict(self) -> Dict:
        d = {
            "l_fit": self.l_fit,
            "time_p_batch": self.runtime,
            "lr": self.lr,
            "nan_in_loss": self.nan_in_loss,
        }
        if self.rev_output is not None:
            d["ave(rev_output)"] = torch.mean(self.rev_output).data.item()
            d["max(rev_output)"] = torch.max(torch.abs(self.rev_output)).data.item()
        if self.f_output is not None:
            d["ave(f_output)"] = torch.mean(self.f_output).data.item()
            d["max(f_output)"] = torch.max(torch.abs(self.f_output)).data.item()
        if self.max_grad is not None:
            d["max_grad"] = self.max_grad
        if self.ave_grad is not None:
            d["ave_grad"] = self.ave_grad
        return d

    def print(self, epoch: int, batch_nb: int, t_start: float):
        print(f"{epoch}:{batch_nb}", end="\t\t")
        max_rev = 0 if self.rev_output is None else torch.max(torch.abs(self.rev_output)).data.item()
        print("%.5f   \t%.5f   \t%.3f   \t%.3f" % (time() - t_start, self.lr, self.l_fit, max_rev))


class ModelAccuracyResults(Config):
    def __init__(
        self,
        l2_errors: List[np.array],
        angular_errors: List[np.array],
        model_runtimes: List[float],
    ):
        assert len(l2_errors) == len(angular_errors)
        assert len(l2_errors) == len(model_runtimes)
        samples_per_endpoint = l2_errors[0].shape[0]
        for l2_errors_i, ang_errs_i in zip(l2_errors, angular_errors):
            assert l2_errors_i.shape == ang_errs_i.shape
            assert l2_errors_i.shape[0] == samples_per_endpoint
        testset_size = len(l2_errors)

        union_l2_errs = np.array(l2_errors)
        union_angular_errs = np.array(angular_errors)
        assert union_l2_errs.size == testset_size * samples_per_endpoint
        assert union_angular_errs.size == testset_size * samples_per_endpoint

        max_l2_errors = [np.max(errs_i) for errs_i in l2_errors]
        max_ang_errors = [np.max(errs_i) for errs_i in angular_errors]
        assert len(max_l2_errors) == testset_size
        assert len(max_ang_errors) == testset_size

        # L2 Error
        self.ave_l2err = np.mean(union_l2_errs)
        self.median_l2err = np.median(union_l2_errs)
        self.std_l2errs = np.std(union_l2_errs)
        self.max_l2err = np.max(union_l2_errs)
        # The average max l2 error on a returned batch of solutions
        self.ave_max_l2err = np.mean(max_l2_errors)
        # The standard deviation of the max l2 error for a returned batch of solutions
        self.std_max_l2err = np.std(max_l2_errors)

        # Angular error
        self.ave_angular_err = np.mean(union_angular_errs)
        self.median_angular_err = np.median(union_angular_errs)
        self.std_angular_err = np.std(union_angular_errs)
        self.max_angular_err = np.max(union_angular_errs)
        # The average max angular error on a returned batch of solutions
        self.ave_max_angular_err = np.mean(max_ang_errors)
        # The stnadard deviation of the max angular error for a returned batch of solutions
        self.std_max_angular_errs = np.std(max_ang_errors)

        # Evaluation diagnostics
        self.test_set_size = testset_size
        self.samples_per_endpoint = samples_per_endpoint
        self.ave_model_runtime = np.average(model_runtimes)

    def get_dict(self):
        return self.__dict__

    def __str__(self) -> str:
        nl = "\n"
        rnd = 4
        ret_str = nl + "ModelAccuracyResults()" + nl
        ret_str += f" ave_l2err:     {round(self.ave_l2err, rnd)} \t (std: {round(self.std_l2errs, rnd)})" + nl
        ret_str += f" max_l2err:   {round(self.max_l2err, rnd)}" + nl
        ret_str += f" ave_max_l2err: {round(self.ave_max_l2err, rnd)} \t (std: {round(self.std_max_l2err, rnd)})" + nl
        ret_str += " ----------" + nl
        ret_str += (
            f" ave_angular_err: {round(self.ave_angular_err, rnd)} \t (std: {round(self.std_angular_err, rnd)})" + nl
        )
        ret_str += f" max_angular_err:   {round(self.max_angular_err, rnd)}" + nl
        ret_str += (
            f" ave_max_angular_err: {round(self.ave_max_angular_err, rnd)} \t (std: {round(self.std_max_angular_errs, rnd)})"
            + nl
        )
        ret_str += (
            f" mean model runtime: {round(self.ave_model_runtime, 3)} \t ({self.test_set_size} target poses in testset, {self.samples_per_endpoint} samples generated per pose)"
            + nl
        )
        return ret_str
