from math import inf
import os
import argparse
import time

# Set seed to ensure reproducibility
import torch
import random
import numpy as np


def set_seed():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(0)


set_seed()

from src.dataset_tags import DATASET_TAGS
from src.robot_models import get_robot
from src.utils import boolean_string
from src import config, train, robot_models
from src.model import ModelWrapper_MultiCINN, IkflowModel
from src.training_parameters import MultiCINN_Parameters, IkflowModelParameters, TrainingConfig
from src.supporting_types import EvaluationSettings, LoggingSettings, SaveSettings

import wandb

SMOKE_TEST = False
CHECKPOINT_EVERY = 100000

# Shared Hps
SHARED_DEFAULT_COUPLING_LAYER = "glow"
SHARED_DEFAULT_RNVP_CLAMP = 2.5
SHARED_DEFAULT_SOFTFLOW_NOISE_SCALE = 0.001
SHARED_DEFAULT_SOFTFLOW_ENABLED = True


# Standard cINN Hps
MULTICINN_DEFAULT_SUBNET_NB_NODES = 5
MULTICINN_DEFAULT_NETWORK_WIDTH = 6
MULTICINN_DEFAULT_COEFF_FN_DEPTH = 2
MULTICINN_DEFAULT_COEFF_FN_WIDTH = 768
MULTICINN_DEFAULT_NON_JOINT_SAMPLE_NOISE = 0.01

# Multi-cINN Hps
CINN_DEFAULT_N_NODES = 6
CINN_DEFAULT_DIM_LATENT_SPACE = 8
CINN_DEFAULT_COEFF_FN_INTERNAL_SIZE = 256
CINN_DEFAULT_COEFF_FN_CONFIG = 3
CINN_DEFAULT_Y_NOISE_SCALE = 1e-7
CINN_DEFAULT_ZEROS_NOISE_SCALE = 1e-3

# Training parameters
DEFAULT_LR = 2.5e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_EPOCHS = 100
DEFAULT_GRADIENT_CLAMP = 5.0
DEFAULT_GAMMA = 0.9794578299341784
DEFAULT_STEP_LR_EVERY_K = int(int(2.5 * 1e6) / 64)
DEFAULT_ROBOT = "panda_arm"

# Logging stats
DEFAULT_SAVE_MODEL_IF_BELOW_L2_ERR = 0.1
DEFAULT_SAVE_MODEL_IF_BELOW_ANG_ERR = 3.141592
DEFAULT_eval_every_k = 25000
DEFAULT_LOG_TR_STATS_EVERY = 20000  # batches
DEFAULT_LOG_DIST_PLOT_EVERY = int(int(2.5 * 1e6) / 64)


"""
# Example usage: 

python3 cli_train.py --model_type=ikflow --robot=valkyrie --dataset_tag=plus_1pi_on_32_MD_SIZE --smoke_test
python3 cli_train.py --model_type=ikflow --robot=panda_arm --smoke_test --log_dist_plots
python3 cli_train.py \
    --model_type=ikflow \
    --robot=panda_arm \
    --nb_nodes=12 \
    --dim_latent_space=9 \
    --coeff_fn_config=3 \
    --coeff_fn_internal_size=1024 \
    --smoke_test 
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")

    # Run parameters
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT)
    parser.add_argument("--test_description", type=str, default="not-set")
    parser.add_argument(
        "--model_type", required=True, type=str, help='One of ["ikflow", "mdn", "iresnet", "multi_cinn"]'
    )

    # Shared hyperparameters
    parser.add_argument("--coupling_layer", type=str, default=SHARED_DEFAULT_COUPLING_LAYER)
    parser.add_argument("--rnvp_clamp", type=float, default=SHARED_DEFAULT_RNVP_CLAMP)
    parser.add_argument("--softflow_noise_scale", type=float, default=SHARED_DEFAULT_SOFTFLOW_NOISE_SCALE)
    parser.add_argument("--softflow_enabled", type=bool, default=SHARED_DEFAULT_SOFTFLOW_ENABLED)

    # Standard cINN Hps
    parser.add_argument("--nb_nodes", type=int, default=CINN_DEFAULT_N_NODES)
    parser.add_argument("--dim_latent_space", type=int, default=CINN_DEFAULT_DIM_LATENT_SPACE)
    parser.add_argument("--coeff_fn_config", type=int, default=CINN_DEFAULT_COEFF_FN_CONFIG)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=CINN_DEFAULT_COEFF_FN_INTERNAL_SIZE)
    parser.add_argument("--y_noise_scale", type=float, default=CINN_DEFAULT_Y_NOISE_SCALE)
    parser.add_argument("--zeros_noise_scale", type=float, default=CINN_DEFAULT_ZEROS_NOISE_SCALE)

    # MultiINN Hyperparameters
    parser.add_argument("--subnet_nb_nodes", type=int, default=MULTICINN_DEFAULT_SUBNET_NB_NODES)
    parser.add_argument("--subnet_network_width", type=int, default=MULTICINN_DEFAULT_NETWORK_WIDTH)
    parser.add_argument("--subnet_coeff_fn_depth", type=int, default=MULTICINN_DEFAULT_COEFF_FN_DEPTH)
    parser.add_argument("--subnet_coeff_fn_width", type=int, default=MULTICINN_DEFAULT_COEFF_FN_WIDTH)
    parser.add_argument("--non_joint_sample_noise", type=float, default=MULTICINN_DEFAULT_NON_JOINT_SAMPLE_NOISE)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--gradient_clamp", type=float, default=DEFAULT_GRADIENT_CLAMP)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--step_lr_every_k", type=int, default=DEFAULT_STEP_LR_EVERY_K)

    # Logging options
    parser.add_argument("--save_if_mean_l2_error_below", type=float, default=DEFAULT_SAVE_MODEL_IF_BELOW_L2_ERR)
    parser.add_argument("--save_if_mean_angular_error_below", type=float, default=DEFAULT_SAVE_MODEL_IF_BELOW_ANG_ERR)
    parser.add_argument("--eval_every_k", type=int, default=DEFAULT_eval_every_k)
    parser.add_argument("--log_training_stats_every", type=int, default=DEFAULT_LOG_TR_STATS_EVERY)
    parser.add_argument("--log_dist_plot_every", type=int, default=DEFAULT_LOG_DIST_PLOT_EVERY)
    parser.add_argument(
        "--log_dist_plots", action="store_true", help="Generate and upload plots of gt vs. returned joint distributions"
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run as a smoke test")

    # Joint range testing
    parser.add_argument("--dataset_tag", type=str, help="optional dataset tag")

    args = parser.parse_args()

    assert args.model_type in ["ikflow", "mdn", "iresnet", "multi_cinn"]
    if args.dataset_tag:
        assert args.dataset_tag in DATASET_TAGS

    SMOKE_TEST = SMOKE_TEST or args.smoke_test
    verbosity = 1

    if SMOKE_TEST:
        logging_settings = LoggingSettings(
            wandb_enabled=True, log_loss_every_k=50, eval_every_k=250, log_dist_plot_every=100
        )
        eval_settings = EvaluationSettings(eval_enabled=True, n_poses_per_evaluation=50, n_samples_per_pose=25)
        save_settings = SaveSettings(save_model=False)
    else:
        eval_settings = EvaluationSettings(eval_enabled=True, n_poses_per_evaluation=750, n_samples_per_pose=250)
        logging_settings = LoggingSettings(
            wandb_enabled=True,
            log_loss_every_k=args.log_training_stats_every,
            eval_every_k=args.eval_every_k,
            log_dist_plot_every=args.log_dist_plot_every,
        )
        save_settings = SaveSettings(
            save_model=True,
            save_if_mean_l2_error_below=args.save_if_mean_l2_error_below,
            save_if_mean_angular_error_below=args.save_if_mean_angular_error_below,
        )

    # Load robot nn_model
    robot_model = get_robot(args.robot)

    training_config = TrainingConfig()
    training_config.test_description = args.test_description
    training_config.n_epochs = args.n_epochs
    training_config.batch_size = args.batch_size
    training_config.use_small_dataset = SMOKE_TEST
    training_config.optimizer = "ranger"
    training_config.learning_rate = args.learning_rate
    training_config.gamma = args.gamma
    training_config.dataset_tag = args.dataset_tag
    training_config.step_lr_every_k = args.step_lr_every_k

    if args.model_type == IkflowModel.nn_model_type:

        hyper_parameters = IkflowModelParameters()
        hyper_parameters.coupling_layer = args.coupling_layer
        hyper_parameters.nb_nodes = args.nb_nodes
        hyper_parameters.dim_latent_space = args.dim_latent_space
        hyper_parameters.coeff_fn_config = args.coeff_fn_config
        hyper_parameters.coeff_fn_internal_size = args.coeff_fn_internal_size
        hyper_parameters.rnvp_clamp = args.rnvp_clamp
        hyper_parameters.softflow_noise_scale = args.softflow_noise_scale
        hyper_parameters.y_noise_scale = args.y_noise_scale
        hyper_parameters.zeros_noise_scale = args.zeros_noise_scale
        hyper_parameters.softflow_enabled = boolean_string(args.softflow_enabled)
        model_wrapper = IkflowModel(hyper_parameters, robot_model)

    elif args.model_type == ModelWrapper_MultiCINN.nn_model_type:

        # Set dimension variables
        hyper_parameters = MultiCINN_Parameters()
        hyper_parameters.coupling_layer = args.coupling_layer
        hyper_parameters.subnet_nb_nodes = args.subnet_nb_nodes
        hyper_parameters.subnet_network_width = args.subnet_network_width
        hyper_parameters.subnet_coeff_fn_depth = args.subnet_coeff_fn_depth
        hyper_parameters.subnet_coeff_fn_width = args.subnet_coeff_fn_width
        hyper_parameters.non_joint_sample_noise = args.non_joint_sample_noise
        # hyper_parameters.init_scale = args.init_scale
        hyper_parameters.rnvp_clamp = args.rnvp_clamp
        hyper_parameters.softflow_noise_scale = args.softflow_noise_scale
        hyper_parameters.softflow_enabled = args.softflow_enabled

        model_wrapper = ModelWrapper_MultiCINN(hyper_parameters, robot_model)

    # SLurm
    try:
        job_nb = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        time.sleep(5 * job_nb)  # sleep so that theres a delay between dataset downloads
    except TypeError:  # Err thrown when running on mac
        print("Could not retrieve SLURM_ARRAY_TASK_ID")

    print(hyper_parameters)
    print(model_wrapper.n_parameters)

    # Train
    exit()
    train.train(
        model_wrapper,
        training_config,
        hyper_parameters,
        robot_model,
        eval_settings,
        logging_settings,
        save_settings,
        CHECKPOINT_EVERY,
        verbosity=verbosity,
        dataset_tag=args.dataset_tag,
        log_dist_plots=args.log_dist_plots,
    )

    # Windows needs `wandb.finish()` for some reason (this command throws an error on mac/ubuntu)
    if not os.name == "posix":
        wandb.finish()

    print("done")
