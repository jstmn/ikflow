from typing import List
import pickle

from src.dataset import Dataset
from src.model import ModelWrapper
from src.evaluation import test_model_accuracy
from src.utils import non_private_dict
from src.training_parameters import TrainingConfig
from src.supporting_types import SaveSettings, TrainingResult

import wandb
import torch
import numpy as np


def get_softflow_noise(x: torch.Tensor, softflow_noise_scale: float):
    """
    Return noise and noise magnitude for softflow. See https://arxiv.org/abs/2006.04604

    Return:
        c (torch.Tensor): An (batch_sz x 1) array storing the magnitude of the sampled gausian `eps` along each row
        eps (torch.Tensor): a (batch_sz x dim_x) array, where v[i] is a sample drawn from a zero mean gaussian with variance = c[i]**2
    """
    dim_x = x.shape[1]

    # (batch_size x 1) from uniform(0, 1)
    c = torch.rand_like(x[:, 0]).unsqueeze(1)

    # (batch_size x dim_y) *  (batch_size x dim_y)  | element wise multiplication
    eps = torch.randn_like(x) * torch.cat(dim_x * [c], dim=1) * softflow_noise_scale
    return c, eps


def log_distribution_plots(
    model_wrapper: ModelWrapper, ik_sols: List[np.array], batch_nb: int = -1, epoch: int = -1, show_fig=False
):

    print("\nlog_distribution_plots()")
    for pose_idx, (target_pose, ik_sols) in enumerate(
        zip(model_wrapper.robot_model.end_poses_to_plot_solution_dists_for, ik_sols)
    ):
        print(f"  sampling solutions for {target_pose}")
        n_samples = ik_sols.shape[0]
        outfile = f"/tmp/solution_dist_y={target_pose.tolist()}.png"
        fig_title = f"Target pose: {target_pose.tolist()}, n_samples:{n_samples}, batch_nb:{batch_nb}, robot:'{model_wrapper.robot_model.name}'"
        model_wrapper.plot_distribution(
            target_pose, n_samples, ik_sols, show_fig=show_fig, output_file=outfile, fig_title=fig_title
        )
        wandb.log(
            {
                "epoch": epoch,
                "batch_nb": batch_nb,
                f"samples_ik_vs_generated_pose_{pose_idx}": wandb.Image(outfile),
                "target_pose": str(target_pose.tolist()),
            }
        )


def save_model(
    model_wrapper: ModelWrapper,
    dataset: Dataset,
    save_settings: SaveSettings,
    wandb_run_name: str,
    n_target_poses: int = 250,
):
    """
    Evaluate the nn_model and save it to disk if it is deemed worthy
    """
    if not save_settings.save_model:
        print("save_model(): save_settings.save_model is False, returning without saving")
        return

    results = test_model_accuracy(model_wrapper, dataset, n_target_poses, 25)
    max_allowable_l2_err = save_settings.save_if_mean_l2_error_below
    max_allowable_ang_err = save_settings.save_if_mean_angular_error_below

    if results.ave_l2err != results.ave_l2err:
        print(f"save_model()\tl2 err is NaN, returning")
        return

    if results.ave_l2err > max_allowable_l2_err or results.ave_angular_err > max_allowable_ang_err:
        print(
            f"save_model()\tl2_err: {round(results.ave_l2err, 4)} > {max_allowable_l2_err} and ang_err: {round(results.ave_angular_err, 4)} > {max_allowable_ang_err}, returning"
        )
        return

    # Note: Wandb. descriptions will be displayed as markdown
    description = f""": 
        | Key                 | Value       |
        | ------------------- | ----------- |
        | robot               | {model_wrapper.robot_model.name} |
        | model_type          | {model_wrapper.nn_model_type} |
        | ave_l2_error        | {results.ave_l2err} |
        | ave_angular_err     | {results.ave_angular_err}
    """
    model_artifact = wandb.Artifact(wandb_run_name, "model", description=description)

    # Save locall
    model_state_dict_filepath = f"/tmp/model.pkl"
    with open(model_state_dict_filepath, "wb") as f:
        if model_wrapper.nn_model_type == "multi_cinn":
            for model in model_wrapper.nn_model:
                pickle.dump(model.state_dict(), f)
        else:
            pickle.dump(model_wrapper.nn_model.state_dict(), f)

    # Upload to wandb
    model_artifact.add_file(model_state_dict_filepath)
    wandb.log_artifact(model_artifact)
    print("save_model()\tWaiting for artifact to finish uploading")
    model_artifact.wait()


def save_checkpoint(
    model_wrapper: ModelWrapper,
    optimizer,
    training_config: TrainingConfig,
    hparams,
    dataset_tag: str,
    epoch: int,
    batch_nb: int,
    wandb_run_name: str,
):
    # Save state
    filepath = f"/tmp/training_checkpoint__epoch:{epoch}__batch:{batch_nb}.pkl"
    state = {
        "nn_model_type": model_wrapper.nn_model_type,
        "model": model_wrapper.nn_model.state_dict(),
        "robot_model": model_wrapper.robot_model.name,
        "optimizer": optimizer.state_dict(),
        "training_config": non_private_dict(training_config.__dict__),
        "hparams": non_private_dict(hparams.__dict__),
        "dataset_tag": dataset_tag,
        "epoch": epoch,
        "batch_nb": batch_nb,
    }
    torch.save(state, filepath)

    # Upload to wandb
    model_artifact = wandb.Artifact(wandb_run_name, type="model", description="checkpoint")
    model_artifact.add_file(filepath)
    wandb.log_artifact(model_artifact)
    model_artifact.wait()
