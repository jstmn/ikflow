import argparse

import os
import config
from src.training_parameters import IkflowModelParameters
from src.ik_solvers import IkflowSolver
from src.robots import get_robot, KlamptRobotModel
from src.lt_model import IkfLitModel
from src.lt_data import IkfLitDataset
from src.utils import boolean_string

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
from pytorch_lightning import Trainer, seed_everything

import wandb
import torch

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
DEFAULT_COEFF_FN_INTERNAL_SIZE = 256
DEFAULT_COEFF_FN_CONFIG = 3
DEFAULT_Y_NOISE_SCALE = 1e-7
DEFAULT_ZEROS_NOISE_SCALE = 1e-3

# Training parameters
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_EPOCHS = 100
DEFAULT_GAMMA = 0.9794578299341784
DEFAULT_STEP_LR_EVERY = int(int(2.5 * 1e6) / 64)
DEFAULT_GRADIENT_CLIP_VAL = 1


# Logging stats
DEFAULT_EVAL_EVERY = 15000
DEFAULT_VAL_SET_SIZE = 500
DEFAULT_LOG_EVERY = 5000
DEFAULT_CHECKPOINT_EVERY = 250000


"""
_____________
Example usage

# Real
python3 train.py \
    --robot_name=panda_arm \
    --nb_nodes=12 \
    --coeff_fn_internal_size=1024 \
    --coeff_fn_config=3 \
    --batch_size=64 \
    --learning_rate=0.00025 \
    --log_every=1000 \
    --eval_every=5000 \
    --val_set_size=500 \


# Smoke testing - w/ wandb
python train.py \
    --robot_name=panda_arm \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --log_every=50 \
    --eval_every=100 \
    --val_set_size=250 \
    --checkpoint_every=500


# Smoke testing
python train.py \
    --robot_name=panda_arm \
    --batch_size=5 \
    --learning_rate=0.0005 \
    --log_every=5 \
    --eval_every=100 \
    --val_set_size=100 \
    --checkpoint_every=100 \
    --disable_wandb
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")
    parser.add_argument("--robot_name", type=str, required=True)

    # Model parameters
    parser.add_argument("--coupling_layer", type=str, default=DEFAULT_COUPLING_LAYER)
    parser.add_argument("--rnvp_clamp", type=float, default=DEFAULT_RNVP_CLAMP)
    parser.add_argument("--softflow_noise_scale", type=float, default=DEFAULT_SOFTFLOW_NOISE_SCALE)
    parser.add_argument("--softflow_enabled", type=bool, default=DEFAULT_SOFTFLOW_ENABLED)
    parser.add_argument("--nb_nodes", type=int, default=DEFAULT_N_NODES)
    parser.add_argument("--dim_latent_space", type=int, default=DEFAULT_DIM_LATENT_SPACE)
    parser.add_argument("--coeff_fn_config", type=int, default=DEFAULT_COEFF_FN_CONFIG)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=DEFAULT_COEFF_FN_INTERNAL_SIZE)
    parser.add_argument("--y_noise_scale", type=float, default=DEFAULT_Y_NOISE_SCALE)
    parser.add_argument("--zeros_noise_scale", type=float, default=DEFAULT_ZEROS_NOISE_SCALE)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--step_lr_every", type=int, default=DEFAULT_STEP_LR_EVERY)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--lambd", type=float, default=1)

    # Logging options
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--val_set_size", type=int, default=DEFAULT_VAL_SET_SIZE)
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()

    assert 0 <= args.lambd and args.lambd <= 1

    # Load model
    robot = get_robot(args.robot_name)
    base_hparams = IkflowModelParameters()
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

    ik_solver = IkflowSolver(base_hparams, robot)

    assert isinstance(robot, KlamptRobotModel), "Only 3d robots are supported for training currently"
    robot.assert_batch_fk_equal()

    torch.autograd.set_detect_anomaly(True)

    # Setup wandb logging
    if args.disable_wandb:
        wandb_logger = None
    else:
        assert os.getenv("WANDB_PROJECT") is not None, (
            "The 'WANDB_PROJECT' environment variable is not set. Either set it with the appropriate wandb project name"
            " (`export WANDB_PROJECT=<your wandb project name>`), or add '--disable_wandb'"
        )
        assert os.getenv("WANDB_ENTITY") is not None, (
            "The 'WANDB_ENTITY' environment variable is not set. Either set it with the appropriate wandb entity"
            " (`export WANDB_ENTITY=<your wandb username>`), or add '--disable_wandb'"
        )

        # Call `wandb.init` before creating a `WandbLogger` object so that runs have randomized names. Without this
        # call, the run names are all set to the project name. See this article for further information: https://lightrun.com/answers/lightning-ai-lightning-wandblogger--use-random-name-instead-of-project-as-default-name
        wandb.init(project=os.getenv("WANDB_PROJECT"))

        wandb_logger = WandbLogger(
            save_dir=config.WANDB_CACHE_DIR, project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"]
        )
        cfg = {"robot": robot.name}
        cfg.update(args.__dict__)
        cfg.update(base_hparams.__dict__)
        wandb_logger.experiment.config.update(cfg)

    data_module = IkfLitDataset(robot.name, args.batch_size, val_set_size=args.val_set_size)
    model = IkfLitModel(
        ik_solver=ik_solver,
        base_hparams=base_hparams,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        gradient_clip=args.gradient_clip_val,
        lambd=args.lambd,
    )
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[],
        val_check_interval=args.eval_every,
        gpus=1,
        log_every_n_steps=args.log_every,
        max_epochs=DEFAULT_MAX_EPOCHS,
    )
    trainer.fit(model, data_module)
