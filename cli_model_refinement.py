import argparse

from src import config
from src.training_parameters import cINN_SoftflowParameters
from src.model import ModelWrapper_cINN_Softflow
from src.robot_models import get_robot
from src.model_refinement.lt_model import IkfLitModel
from src.model_refinement.lt_data import IkfLitDataset
from src.utils import get_model_wrapper
from tpms import TPMS

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
from pytorch_lightning import Trainer, callbacks, seed_everything

import torch


SEED = 0
seed_everything(SEED, workers=True)

DEFAULT_DATASET_TAG = "LARGE50"
# L2_MULTI_DATASET_TAG = "includes_l2_loss_pts"
L2_MULTI_DATASET_TAG = "includes_l2_loss_pts2pt"


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

# TODO(@jeremysm): Use larger dataset (50mil) so can increase # batches before eval (currently limited to 2500000 / 256 = 9700)

"""
_____________
Example usage

# real
python3 cli_model_refinement.py \
    --dataset_tag="LARGE50" \
    --use_uninitialized_model \
    --loss_fn="l2" \
    --batch_size=512 \
    --learning_rate=0.0005 \
    --log_every=1000 \
    --eval_every=5000 \
    --val_set_size=500

# Dev
python3 cli_model_refinement.py \
    --use_uninitialized_model \
    --loss_fn="l2-multi" \
    --batch_size=5 \
    --learning_rate=0.0005 \
    --log_every=5 \
    --eval_every=100 \
    --val_set_size=100 \
    --disable_wandb



"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")
    parser.add_argument("--tpm_name", type=str, help="Name of the tpm to lookup in tpms.py. Ex: 'panda_tpm'")
    parser.add_argument("--use_uninitialized_model", action="store_true")

    # Training parameters
    parser.add_argument("--dataset_tag", type=str, default=DEFAULT_DATASET_TAG)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--step_lr_every", type=int, default=DEFAULT_STEP_LR_EVERY)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--lambd", type=float, default=1)
    parser.add_argument("--loss_fn", type=str, required=True)

    # Logging options
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--val_set_size", type=int, default=DEFAULT_VAL_SET_SIZE)
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    assert 0 <= args.lambd and args.lambd <= 1

    # Load model
    if args.use_uninitialized_model:
        assert args.tpm_name is None
        robot = get_robot("robosimian")
        base_hparams = cINN_SoftflowParameters()
        base_hparams.nb_nodes = 16
        base_hparams.coeff_fn_config = 3
        base_hparams.coeff_fn_internal_size = 1024
        model_wrapper = ModelWrapper_cINN_Softflow(base_hparams, robot)
        wandb_run_name = "n/a"
    else:
        wandb_run_name = TPMS[args.tpm_name]["wandb_run_name"]
        model_wrapper, base_hparams = get_model_wrapper(wandb_run_name)
        robot = model_wrapper.robot

    robot.assert_batch_fk_equal()
    if args.loss_fn == "l2-multi":
        robot.assert_l2_loss_pts_equal()
        args.dataset_tag = L2_MULTI_DATASET_TAG

    torch.autograd.set_detect_anomaly(True)

    if args.smoke_test:
        data_module = IkfLitDataset(
            robot.name,
            args.batch_size,
            use_small_dataset=True,
            val_set_size=args.val_set_size,
            tag=args.dataset_tag,
        )
        model = IkfLitModel(
            model_wrapper,
            base_hparams,
            args.loss_fn,
            args.learning_rate,
            args.checkpoint_every,
            log_every=args.log_every,
            gradient_clip=args.gradient_clip_val,
            lambd=args.lambd,
        )
        trainer = Trainer(
            callbacks=[],
            val_check_interval=args.eval_every,
            gpus=1,
            log_every_n_steps=args.log_every,
            #             gradient_clip_val=args.gradient_clip_val,
            #             gradient_clip_algorithm="value",
        )
        trainer.fit(model, data_module)
    else:
        # Setup wandb logging
        if args.disable_wandb:
            wandb_logger = None
        else:
            wandb_logger = WandbLogger(project=PROJECT, save_dir=config.WANDB_CACHE_DIR)
            cfg = {
                "base_wandb_run_name": wandb_run_name,
                "base_tpm_name": args.tpm_name,
                "is_cluster": config.is_cluster,
                "is_ubuntu_dev_machine": config.is_ubuntu_dev_machine,
                "gradient_clip_val": args.gradient_clip_val,
                "dataset_tag": args.dataset_tag,
                "robot": robot.name,
            }
            cfg.update(args.__dict__)
            cfg.update(base_hparams.__dict__)
            wandb_logger.experiment.config.update(cfg)

        data_module = IkfLitDataset(
            robot.name, args.batch_size, use_small_dataset=False, val_set_size=args.val_set_size, tag=args.dataset_tag
        )
        model = IkfLitModel(
            model_wrapper=model_wrapper,
            base_hparams=base_hparams,
            loss_fn=args.loss_fn,
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
            #             gradient_clip_val=args.gradient_clip_val,
            #             gradient_clip_algorithm="value",
        )
        trainer.fit(model, data_module)
