import argparse
import os
from time import time
import sys
import pickle
from typing import Dict

sys.path.append(os.getcwd())

import wandb
import torch


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
        print(k_new)
    return updated


"""
_____________
Example usage
 
python tools/download_model_from_wandb_checkpoint.py --wandb_run_id=1zkh9ofo
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Resume training from a checkpoint")

    # Note: WandB saves artifacts by the run ID (i.e. '34c2gimi') not the run name ('dashing-forest-33'). This is
    # slightly annoying because you need to click on a run to get its ID.
    parser.add_argument("--wandb_run_id", type=str, help="The run ID of the wandb run to load. Example: '34c2gimi'")
    parser.add_argument("--disable_progress_bar", action="store_true")
    args = parser.parse_args()

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    assert wandb_project is not None, (
        "The 'WANDB_PROJECT' environment variable is not set. Either set it with the appropriate wandb project name"
        " (`export WANDB_PROJECT=<your wandb project name>`), or add '--disable_wandb'"
    )
    assert wandb_entity is not None, (
        "The 'WANDB_ENTITY' environment variable is not set. Either set it with the appropriate wandb entity"
        " (`export WANDB_ENTITY=<your wandb username>`), or add '--disable_wandb'"
    )

    t0 = time()
    api = wandb.Api()
    artifact = api.artifact(f"{wandb_entity}/{wandb_project}/model-{args.wandb_run_id}:best_k")
    download_dir = artifact.download()
    print(f"Downloaded artifact in {round(time()- t0, 2)}s")

    t0 = time()
    run = api.run(f"/{wandb_entity}/{wandb_project}/runs/{args.wandb_run_id}")
    print(f"Downloaded run data in {round(time()- t0, 2)}s")

    run_name = run.name
    robot_name = run.config["robot"]
    ckpt_filepath = os.path.join(download_dir, "model.ckpt")
    checkpoint = torch.load(ckpt_filepath, map_location=lambda storage, loc: storage)
    state_dict = format_state_dict(checkpoint["state_dict"])

    # Save model's state_dict
    # TODO(@jstmn): Save the global_step aswell
    model_state_dict_filepath = os.path.join(f"trained_models/{robot_name}-{run_name}.pkl")
    with open(model_state_dict_filepath, "wb") as f:
        pickle.dump(state_dict, f)
