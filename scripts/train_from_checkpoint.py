import argparse
import os
from ikflow import config
from time import time


from ikflow.supporting_types import IkflowModelParameters
from ikflow.ikflow_solver import IkflowSolver
from jkinpylib.robots import get_robot
from ikflow.training.lt_model import IkfLitModel, checkpoint_dir
from ikflow.training.lt_data import IkfLitDataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
from pytorch_lightning import Trainer, seed_everything

import wandb
import torch

DEFAULT_MAX_EPOCHS = 5000
SEED = 0
seed_everything(SEED, workers=True)


"""
_____________
Example usage

python scripts/train_from_checkpoint.py --wandb_run_id=2pcxohrk
python scripts/train_from_checkpoint.py --wandb_run_id=3901tguh # crap run
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

    wandb_run = wandb.init(entity=wandb_entity, project=wandb_project, id=args.wandb_run_id, resume="must")

    # Note: The checkpointing call back we use (`ModelCheckpoint`) saves the k most recent checkpoints and chooses
    # which to save by ranking with `global_step`. This means for whatever reason that the 'latest' alias isn't
    # available on the wandb side. They store 'best_k' instead which is what we want (I hope <*_*>)
    artifact = wandb_run.use_artifact(f"model-{args.wandb_run_id}:best_k")
    t0 = time()
    artifact_dir = artifact.download()
    print(f"Downloaded artifact '{artifact.name}' in {round(1000*(time() - t0),2)} ms")
    ckpt_filepath = os.path.join(artifact_dir, "model.ckpt")
    checkpoint = torch.load(ckpt_filepath, map_location=lambda storage, loc: storage)

    #  Get the original runs WandB config. Example config:
    #     {
    #         ...
    #         "robot_name":"panda_arm",
    #         "val_set_size":25,
    #         "optimizer_name":"ranger",
    #         ...
    #     }
    api = wandb.Api()
    prev_run = api.run(f"/{wandb_entity}/{wandb_project}/runs/{args.wandb_run_id}")
    run_cfg = prev_run.config
    robot_name = run_cfg["robot"]

    # Create training objects
    ikflow_hparams: IkflowModelParameters = checkpoint["hyper_parameters"]["base_hparams"]
    robot = get_robot(robot_name)
    wandb_logger = WandbLogger(log_model="all", save_dir=config.WANDB_CACHE_DIR)
    ik_solver = IkflowSolver(ikflow_hparams, robot)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir(robot_name),
        every_n_train_steps=run_cfg["checkpoint_every"],
        save_on_train_epoch_end=False,
        # Save the last 3 checkpoints. Checkpoint files are logged as wandb artifacts so it doesn't really matter anyway
        save_top_k=3,
        monitor="global_step",
        mode="max",
        filename="ikflow-checkpoint-{step}",
    )

    # Train
    model = IkfLitModel.load_from_checkpoint(ckpt_filepath, ik_solver=ik_solver)
    data_module = IkfLitDataset(robot_name, run_cfg["batch_size"], val_set_size=run_cfg["val_set_size"])

    trainer = Trainer(
        # resume_from_checkpoint=ckpt_filepath, # 'Please pass `Trainer.fit(ckpt_path=)` directly instead'
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=run_cfg["eval_every"],
        # # gpus=1, # deprecated
        accelerator="gpu",
        devices=1,
        log_every_n_steps=run_cfg["log_every"],
        max_epochs=DEFAULT_MAX_EPOCHS,
        enable_progress_bar=False if (os.getenv("IS_SLURM") is not None) or args.disable_progress_bar else True,
    )
    trainer.fit(model, data_module, ckpt_path=ckpt_filepath)
