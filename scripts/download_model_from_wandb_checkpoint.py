from typing import Dict
import argparse
import os
from time import time
import pickle

import wandb
import torch

from ikflow.utils import get_wandb_project


def format_state_dict(state_dict: Dict) -> Dict:
    """The `state_dict` saved in checkpoints will have keys with the form:
        "nn_model.module_list.0.M", "nn_model.module_list.0.M_inv", ...

    This function updates them to the expected format below. (note the 'nn_model' prefix is removed)
        "module_list.0.M", "module_list.0.M_inv", ...
    """
    bad_prefix = "nn_model."
    len_prefix = len(bad_prefix)
    updated = {}
    for k, v in state_dict.items():
        # Check that the original state dict is malformatted first
        assert k[0:len_prefix] == bad_prefix
        k_new = k[len_prefix:]
        updated[k_new] = v
    return updated


"""
_____________
Example usage

uv run python scripts/download_model_from_wandb_checkpoint.py --wandb_run_id=2uidt835
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Download ikflow model from Weights and Biases")

    # Note: WandB saves artifacts by the run ID (i.e. '34c2gimi') not the run name ('dashing-forest-33'). This is
    # slightly annoying because you need to click on a run to get its ID.
    parser.add_argument("--wandb_run_id", type=str, help="The run ID of the wandb run to load. Example: '34c2gimi'")
    parser.add_argument("--disable_progress_bar", action="store_true")
    args = parser.parse_args()

    wandb_entity, wandb_project = get_wandb_project()
    t0 = time()
    api = wandb.Api()
    artifact = api.artifact(f"{wandb_entity}/{wandb_project}/model-{args.wandb_run_id}:best_k")
    download_dir = artifact.download()
    print(f"Downloaded artifact in {round(time() - t0, 2)}s")

    t0 = time()
    run = api.run(f"/{wandb_entity}/{wandb_project}/runs/{args.wandb_run_id}")
    print(f"Downloaded run data in {round(time() - t0, 2)}s")

    run_name = run.name
    robot_name = run.config["robot"]
    ckpt_filepath = os.path.join(download_dir, "model.ckpt")
    checkpoint = torch.load(ckpt_filepath, map_location=lambda storage, loc: storage)
    state_dict = format_state_dict(checkpoint["state_dict"])
    global_step = str(checkpoint["global_step"] / 1e6) + "M"

    # Save model's state_dict
    # TODO(@jstmn): Save the global_step aswell
    model_state_dict_filepath = os.path.join(f"{robot_name}__{run_name}__global_step={global_step}.pkl")
    with open(model_state_dict_filepath, "wb") as f:
        pickle.dump(state_dict, f)
