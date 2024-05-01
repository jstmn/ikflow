import argparse
import os

from jrl.robots import get_robot, Robot
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
from pytorch_lightning import Trainer, seed_everything
import wandb
import torch

from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING
from ikflow import config
from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
from ikflow.training.lt_data import IkfLitDataset
from ikflow.training.training_utils import get_checkpoint_dir
from ikflow.utils import boolean_string, non_private_dict, get_wandb_project


DEFAULT_MAX_EPOCHS = 5000
SEED = 0
seed_everything(SEED, workers=True)

# Model parameters
DEFAULT_COUPLING_LAYER = "glow"
DEFAULT_RNVP_CLAMP = 2.5
DEFAULT_SOFTFLOW_NOISE_SCALE = 0.001
DEFAULT_SOFTFLOW_ENABLED = True
DEFAULT_N_NODES = 6
DEFAULT_DIM_LATENT_SPACE = 8
DEFAULT_COEFF_FN_INTERNAL_SIZE = 1024
DEFAULT_COEFF_FN_CONFIG = 3
DEFAULT_Y_NOISE_SCALE = 1e-7
DEFAULT_ZEROS_NOISE_SCALE = 1e-3
DEFAULT_SIGMOID_ON_OUTPUT = False

# Training parameters
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_N_EPOCHS = 100
DEFAULT_GAMMA = 0.9794578299341784
DEFAULT_STEP_LR_EVERY = int(int(2.5 * 1e6) / 64)
DEFAULT_GRADIENT_CLIP_VAL = 1


# Logging stats
DEFAULT_EVAL_EVERY = 20000
DEFAULT_VAL_SET_SIZE = 500
DEFAULT_LOG_EVERY = 20000
DEFAULT_CHECKPOINT_EVERY = 250000


"""
_____________
Example usage

# Smoke test - wandb enabled
python scripts/train.py \
    --robot_name=panda \
    --nb_nodes=6 \
    --batch_size=128 \
    --learning_rate=0.0005 \
    --log_every=250 \
    --eval_every=100 \
    --val_set_size=250 \
    --checkpoint_every=500


# Smoke test - wandb enabled, with sigmoid clamping
python scripts/train.py \
    --robot_name=panda \
    --nb_nodes=6 \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --log_every=250 \
    --eval_every=250 \
    --val_set_size=250 \
    --softflow_enabled=False \
    --sigmoid_on_output=True \
    --dim_latent_space=10 \
    --checkpoint_every=50000


# Smoke test - wandb disabled
python scripts/train.py \
    --robot_name=fetch \
    --log_every=25 \
    --batch_size=64 \
    --eval_every=100 \
    --val_set_size=100 \
    --checkpoint_every=500 \
    --disable_wandb

# Test the learning rate scheduler
python scripts/train.py \
    --robot_name=panda \
    --learning_rate=1.0 \
    --gamma=0.5 \
    --step_lr_every=10 \
    --disable_wandb
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")

    # Note: WandB saves artifacts by the run ID (i.e. '34c2gimi') not the run name ('dashing-forest-33'). This is
    # slightly annoying because you need to click on a run to get its ID.
    parser.add_argument("--wandb_run_id_to_load_checkpoint", type=str, help="Example: '34c2gimi'")
    parser.add_argument("--robot_name", type=str)

    # Model parameters
    parser.add_argument("--coupling_layer", type=str, default=DEFAULT_COUPLING_LAYER)
    parser.add_argument("--rnvp_clamp", type=float, default=DEFAULT_RNVP_CLAMP)
    parser.add_argument("--softflow_noise_scale", type=float, default=DEFAULT_SOFTFLOW_NOISE_SCALE)
    # NOTE: NEVER use 'bool' type with argparse. It will cause you pain and suffering.
    parser.add_argument("--softflow_enabled", type=str, default=DEFAULT_SOFTFLOW_ENABLED)
    parser.add_argument("--nb_nodes", type=int, default=DEFAULT_N_NODES)
    parser.add_argument("--dim_latent_space", type=int, default=DEFAULT_DIM_LATENT_SPACE)
    parser.add_argument("--coeff_fn_config", type=int, default=DEFAULT_COEFF_FN_CONFIG)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=DEFAULT_COEFF_FN_INTERNAL_SIZE)
    parser.add_argument("--y_noise_scale", type=float, default=DEFAULT_Y_NOISE_SCALE)
    parser.add_argument("--zeros_noise_scale", type=float, default=DEFAULT_ZEROS_NOISE_SCALE)
    # See note above about pain and suffering.
    parser.add_argument("--sigmoid_on_output", type=str, default=DEFAULT_SIGMOID_ON_OUTPUT)

    # Training parameters
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--step_lr_every", type=int, default=DEFAULT_STEP_LR_EVERY)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--lambd", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1.8e-05)

    # Logging options
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--val_set_size", type=int, default=DEFAULT_VAL_SET_SIZE)
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--dataset_tags", nargs="+", type=str, default=[DATASET_TAG_NON_SELF_COLLIDING])
    parser.add_argument("--run_description", type=str)
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()
    print("\nArgparse arguments:")
    for k, v in vars(args).items():
        print(f"  {k}={v}")
    print()

    if args.dataset_tags is None:
        args.dataset_tags = []

    assert (
        DATASET_TAG_NON_SELF_COLLIDING in args.dataset_tags
    ), "The 'non-self-colliding' dataset should be specified (for now)"
    assert args.optimizer in ["ranger", "adadelta", "adamw"]
    assert 0 <= args.lambd and args.lambd <= 1

    # Load model
    robot = get_robot(args.robot_name)
    base_hparams = IkflowModelParameters()
    base_hparams.run_description = args.run_description
    base_hparams.coupling_layer = args.coupling_layer
    base_hparams.nb_nodes = args.nb_nodes
    base_hparams.dim_latent_space = args.dim_latent_space
    base_hparams.coeff_fn_config = args.coeff_fn_config
    base_hparams.coeff_fn_internal_size = args.coeff_fn_internal_size
    base_hparams.rnvp_clamp = args.rnvp_clamp
    base_hparams.softflow_noise_scale = args.softflow_noise_scale
    base_hparams.y_noise_scale = args.y_noise_scale
    base_hparams.zeros_noise_scale = args.zeros_noise_scale
    base_hparams.softflow_enabled = boolean_string(args.softflow_enabled)
    base_hparams.sigmoid_on_output = boolean_string(args.sigmoid_on_output)
    print()
    print(base_hparams)

    torch.autograd.set_detect_anomaly(False)
    data_module = IkfLitDataset(robot.name, args.batch_size, args.val_set_size, args.dataset_tags)

    # Setup wandb logging
    wandb_logger = None
    if not args.disable_wandb:
        wandb_entity, wandb_project = get_wandb_project()
        # Call `wandb.init` before creating a `WandbLogger` object so that runs have randomized names. Without this
        # call, the run names are all set to the project name. See this article for further information: https://lightrun.com/answers/lightning-ai-lightning-wandblogger--use-random-name-instead-of-project-as-default-name
        cfg = {"robot": args.robot_name}
        cfg.update(non_private_dict(args.__dict__))
        data_module.add_dataset_hashes_to_cfg(cfg)

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            notes=args.run_description,
            config=cfg,
        )
        wandb_logger = WandbLogger(log_model="all", save_dir=config.WANDB_CACHE_DIR)
        run = wandb.run
        run.tags = run.tags + tuple(args.dataset_tags)

    ik_solver = IKFlowSolver(base_hparams, robot)
    model = IkfLitModel(
        ik_solver=ik_solver,
        base_hparams=base_hparams,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        gradient_clip=args.gradient_clip_val,
        lambd=args.lambd,
        gamma=args.gamma,
        step_lr_every=args.step_lr_every,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        sigmoid_on_output=boolean_string(args.sigmoid_on_output),
    )

    # Checkpoint callback
    checkpoint_directory = get_checkpoint_dir(args.robot_name)
    if not args.disable_wandb:
        wandb.config.update({"checkpoint_directory": checkpoint_directory})

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_directory,
        every_n_train_steps=args.checkpoint_every,
        save_on_train_epoch_end=False,
        # Save the last 3 checkpoints. Checkpoint files are logged as wandb artifacts so it doesn't really matter anyway
        save_top_k=3,
        monitor="global_step",
        mode="max",
        filename="ikflow-checkpoint-{step}",
    )

    # Train
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=args.eval_every,
        # gpus=1, # deprecated
        devices="auto",
        accelerator="gpu",
        log_every_n_steps=args.log_every,
        max_epochs=DEFAULT_MAX_EPOCHS,
        enable_progress_bar=False if (os.getenv("IS_SLURM") is not None) or args.disable_progress_bar else True,
    )
    trainer.fit(model, data_module)
