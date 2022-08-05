from time import time
import traceback
from typing import Callable, Dict, Union
from typing import Optional

from src.train_utils import save_model, save_checkpoint, get_softflow_noise, log_distribution_plots
from src.utils import grad_stats, get_learning_rate
from src.robot_models import RobotModel
from src.dataset import Dataset
from src.model import ModelWrapper, ModelWrapper_MDN, ModelWrapper_MultiCINN
from src.evaluation import test_model_accuracy
from src.training_parameters import MDNParameters, MultiCINN_Parameters, IkflowModelParameters, TrainingConfig
from src.supporting_types import EvaluationSettings, LoggingSettings, SaveSettings, TrainingResult, TrainingStepResult
from src.ranger import RangerVA  # from ranger913A.py
import src.config

import torch.optim
import torch
import wandb

torch.set_printoptions(profile="full")

DEVICE = src.config.device


def training_step_cinn_sflow(
    model_wrapper: ModelWrapper,
    hyper_parameters: IkflowModelParameters,
    training_config: TrainingConfig,
    optimizer,
    params_trainable,
    x: torch.Tensor,
    y: torch.Tensor,
    throw_kb_interupt: bool = False,
) -> TrainingStepResult:
    """
    Perform one training step. Run the nn_model, collect the gradient with l_forw.backward() and update the models'
    weights with optimizer.step()
    """

    ts_output = TrainingStepResult()
    t0 = time()
    dim_x, dim_tot = model_wrapper.dim_x, model_wrapper.dim_tot
    batch_size = x.shape[0]

    optimizer.zero_grad()

    def noise_batch(ndim: int):
        """Return a (batch_size x ndim) array filled with values drawn from a normal distribution"""
        return torch.randn(batch_size, ndim).to(DEVICE)

    if hyper_parameters.softflow_enabled:
        c, v = get_softflow_noise(x, hyper_parameters.softflow_noise_scale)
        x = x + v

        if dim_tot > dim_x:
            pad_x = hyper_parameters.zeros_noise_scale * noise_batch(dim_tot - dim_x)
            x = torch.cat([x, pad_x], dim=1)

        cond = torch.cat([y, c], dim=1)
    else:
        if dim_tot > dim_x:
            pad_x = hyper_parameters.zeros_noise_scale * noise_batch(dim_tot - dim_x)
            x = torch.cat([x, pad_x], dim=1)
        cond = y

    # Shape checking
    assert x.shape[1] == model_wrapper.dim_tot
    assert cond.shape[0] == batch_size
    assert cond.shape[1] == model_wrapper.dim_cond

    output, jac = model_wrapper.nn_model.forward(x, c=cond, jac=True)
    zz = torch.sum(output ** 2, dim=1)
    neg_log_likeli = 0.5 * zz - jac
    loss = hyper_parameters.lambd_predict * torch.mean(neg_log_likeli)

    # Backprop the loss
    loss.backward()

    # Check for NaN's in the loss
    if not loss.item() == loss.item():
        ts_output.nan_in_loss = True

    for p in params_trainable:
        p.grad.data.clamp_(-training_config.gradient_clamp, training_config.gradient_clamp)

    optimizer.step()

    # Purely for unit testing
    if throw_kb_interupt:
        raise KeyboardInterrupt

    # Log training metrics
    ts_output.ave_grad, _, ts_output.max_grad = grad_stats(params_trainable)
    ts_output.lr = get_learning_rate(optimizer)
    ts_output.l_fit = loss.item()
    ts_output.runtime = time() - t0
    ts_output.f_output = output
    return ts_output


def training_step_multi_cinn_sflow(
    model_wrapper: ModelWrapper_MultiCINN,
    hyper_parameters: MultiCINN_Parameters,
    training_config: TrainingConfig,
    optimizer,
    params_trainable,
    x: torch.Tensor,
    y: torch.Tensor,
    throw_kb_interupt: bool = False,
) -> TrainingStepResult:
    """
    Perform one training step. Run the nn_model, collect the gradient with l_forw.backward() and update the models'
    weights with optimizer.step()
    """

    ts_output = TrainingStepResult()
    t0 = time()
    batch_size = x.shape[0]

    optimizer.zero_grad()

    def noise_batch(ndim: int):
        """Return a (batch_size x ndim) array filled with values drawn from a normal distribution"""
        return torch.randn(batch_size, ndim).to(DEVICE)

    # Format x, y and the condition tensor
    if hyper_parameters.softflow_enabled:
        c, v = get_softflow_noise(x, hyper_parameters.softflow_noise_scale)
        x = x + v
        cond = torch.cat([y, c], dim=1)
    else:
        cond = y

    # Shape checking
    assert cond.shape[0] == batch_size
    assert cond.shape[1] == model_wrapper.base_dim_cond

    # Calculate loss
    loss = 0

    for i, model in enumerate(model_wrapper.nn_model):
        output = None
        jac = None

        # Get nn inputs
        noise_pad = hyper_parameters.non_joint_sample_noise * noise_batch(hyper_parameters.subnet_network_width - 1)

        joint_i_column = x[:, i].unsqueeze(-1)

        nn_input = torch.cat([joint_i_column, noise_pad], dim=1)
        c = cond
        if i > 0:
            c = torch.cat([cond, x[:, 0:i]], dim=1)

        # Update loss
        output, jac = model.forward(nn_input, c=c, jac=True)
        zz = torch.sum(output ** 2, dim=1)
        neg_log_likeli = 0.5 * zz - jac
        loss = loss + hyper_parameters.lambd_predict * torch.mean(neg_log_likeli)

    # Backprop the loss
    loss.backward()

    # Check for NaN's in the loss
    if not loss.item() == loss.item():
        ts_output.nan_in_loss = True

    # Update optimizer
    for p in params_trainable:
        p.grad.data.clamp_(-training_config.gradient_clamp, training_config.gradient_clamp)
    optimizer.step()

    # For unit testing
    if throw_kb_interupt:
        raise KeyboardInterrupt

    # Log training metrics
    ts_output.ave_grad, _, ts_output.max_grad = grad_stats(params_trainable)
    ts_output.lr = get_learning_rate(optimizer)
    ts_output.l_fit = loss.item()
    ts_output.runtime = time() - t0
    ts_output.f_output = output
    return ts_output


def training_step_mdn(
    model_wrapper: ModelWrapper,
    hyper_parameters: ModelWrapper_MDN,
    training_config: TrainingConfig,
    optimizer,
    params_trainable,
    x: torch.Tensor,
    y: torch.Tensor,
    throw_kb_interupt: bool = False,
) -> TrainingStepResult:
    """
    Perform one training step. Run the MDN model, collect the gradient with loss.backward() and update the models' weights with optimizer.step()
    """
    # Purely for unit testing
    if throw_kb_interupt:
        raise KeyboardInterrupt

    ts_output = TrainingStepResult()
    t0 = time()

    optimizer.zero_grad()

    # Note: here `y` is the conditional variable and `x` is the target variable. Bishop and tonyduan/mdn have the reverse - `x` is the conditional variable and `y` is the target variable.
    loss = model_wrapper.nn_model.loss(y, x).mean()
    loss.backward()

    # Check for NaN's in the loss
    if not loss.item() == loss.item():
        print("NaN in forw. loss")
        ts_output.nan_in_loss = True

    # Save the gradient
    ts_output.ave_grad, _, ts_output.max_grad = grad_stats(params_trainable)
    optimizer.step()

    # Log training metrics
    ts_output.lr = get_learning_rate(optimizer)
    ts_output.l_fit = loss.item()
    ts_output.runtime = time() - t0
    ts_output.f_output = model_wrapper.nn_model.sample(y)
    return ts_output


# Check parameters are valid
def check_mdn_parameters(params: MDNParameters):
    """Asserts that mdn parameters are valid"""
    assert params.n_components > 0


def check_cinn_sflow_parameters(params: IkflowModelParameters):
    """Asserts that ikflow parameters are valid"""
    assert params.coeff_fn_config in [1, 2, 3, 4, 5, 6, 7]


def check_multi_cinn_parameters(params: MultiCINN_Parameters):
    """Asserts that parameters for a MultiINN are valid"""
    assert params.softflow_noise_scale > 0
    assert params.subnet_network_width >= 2
    assert params.subnet_coeff_fn_depth in [1, 2, 3, 4, 5, 6, 7]


# Initialize models
def initialize_glow_model(params_trainable, hyper_parameters: IkflowModelParameters):
    if hyper_parameters.init_scale > 0:
        for p in params_trainable:
            p.data = hyper_parameters.init_scale * torch.randn(p.data.shape).to(DEVICE)


def initialize_multi_cinn_model(params_trainable, hyper_parameters: MultiCINN_Parameters):
    """Initialize a MultiCINN model. Currently a no-op"""
    if hyper_parameters.init_scale > 0:
        for p in params_trainable:
            p.data = hyper_parameters.init_scale * torch.randn(p.data.shape).to(DEVICE)


def initialize_mdn_model(params_trainable, hyper_parameters: MDNParameters):
    pass


training_functions: Dict[str, Callable[[], TrainingStepResult]] = {
    "ikflow": training_step_cinn_sflow,
    "multi_cinn": training_step_multi_cinn_sflow,
    "mdn": training_step_mdn,
}

parameter_checking_functions = {
    "ikflow": check_cinn_sflow_parameters,
    "multi_cinn": check_multi_cinn_parameters,
    "mdn": check_mdn_parameters,
}

initialization_functions = {
    "ikflow": initialize_glow_model,
    "multi_cinn": initialize_multi_cinn_model,
    "mdn": initialize_mdn_model,
}


def train(
    model_wrapper: ModelWrapper,
    training_config: TrainingConfig,
    hyper_parameters: Union[IkflowModelParameters, MDNParameters],
    robot_model: RobotModel,
    eval_settings: EvaluationSettings,
    logging_settings: LoggingSettings,
    save_settings: SaveSettings,
    checkpoint_every: int,
    verbosity: int = 1,
    throw_keyboard_interupt_on_batch: int = -1,
    dataset_tag: Optional[str] = None,
    log_dist_plots=False,
) -> TrainingResult:
    """Train a model with given settings. Hyper-parameters are set in src.training_parameters.

    Args:
        model_wrapper (str):
        training_config (TrainingConfig,
        hyper_parameters (Union[IkflowModelParameters, MDNParameters],
        robot_model (RobotModel):
        eval_settings (EvaluationSettings):
        logging_settings (LoggingSettings):
        save_settings (SaveSettings):
        verbosity (int):
        throw_keyboard_interupt_on_batch:

    Returns:
        (TrainingResult): Training result
    """
    # Check parameters
    assert training_config.optimizer.upper() == "RANGER"
    assert 0 < training_config.gamma and training_config.gamma < 1
    if log_dist_plots:
        assert (
            len(robot_model.end_poses_to_plot_solution_dists_for) > 0
        ), "Need to have poses set to `end_poses_to_plot_solution_dists_for` to use `log_dist_plots`"
    parameter_checking_functions[model_wrapper.nn_model_type](hyper_parameters)

    # Logging
    if verbosity > 0:
        print("Model:", model_wrapper)
        print("Training parameters:", training_config)
        print("Hyper parameters:", hyper_parameters)

    n_samples = 2500
    ik_sols_for_distribution_plots = []
    for target_pose in robot_model.end_poses_to_plot_solution_dists_for:
        ik_samples = robot_model.inverse_kinematics(target_pose, n_samples)
        ik_sols_for_distribution_plots.append(ik_samples)

    # Initialize weights and biases logging
    if logging_settings.wandb_enabled:
        test_config = {
            "robot": model_wrapper.robot_model.name,
            "model_type": model_wrapper.nn_model_type,
            "git_branch": src.config.git_branch,
            "n_trainable_parameters": model_wrapper.n_parameters,
        }
        test_config.update(training_config.__dict__)
        test_config.update(hyper_parameters.__dict__)
        tags = [model_wrapper.robot_model.name, model_wrapper.nn_model_type, "REPRODUCIBLE"]
        if model_wrapper.cuda_out_of_memory:
            tags.append("CUDA_OUT_OF_MEMORY_ERROR")
        wandb_run = wandb.init(
            project="nnik",
            notes=training_config.test_description,
            tags=tags,
            config=test_config,
        )
        # watch() doesn't provide useful info.
        # wandb.watch(model_wrapper.nn_model, log_freq=100, log="gradients")
    else:
        wandb_run = wandb.init(project="nnik", notes="from train.py - logging disabled", tags=["SAFE_TO_DELETE"])

    wandb_run_name = wandb.run.name

    # Report to wandb if there is a cuda error
    if model_wrapper.cuda_out_of_memory:
        msg = {"epoch": 0, "batch_nb": 0, "CUDA_OUT_OF_MEMORY": True}
        wandb.log(msg)
        return

    # Initialize Optimizer
    params_trainable = None
    if model_wrapper.nn_model_type == "multi_cinn":
        for i, model in enumerate(model_wrapper.nn_model):
            if i == 0:
                params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
            else:
                params_trainable = params_trainable + list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        params_trainable = list(filter(lambda p: p.requires_grad, model_wrapper.nn_model.parameters()))
    initialization_functions[model_wrapper.nn_model_type](params_trainable, hyper_parameters)

    optimizer = RangerVA(
        params_trainable,
        lr=training_config.learning_rate,
        betas=(0.95, 0.999),
        eps=1e-04,
        weight_decay=training_config.l2_reg,
    )
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=training_config.gamma)

    # Get dataset
    dataset = Dataset(
        wandb_run,
        robot_model.name,
        training_config.batch_size,
        use_small_dataset=training_config.use_small_dataset,
        tag=dataset_tag,
    )
    dataset.log_dataset_sizes()

    t_start = time()
    batch_nb = 0
    epoch = 0

    print("\nEpoch:BatchNb \tS/batch  \tlr   \t\tl_fit   \tmax(output_rev)\tl_total    ")
    while epoch < training_config.n_epochs:

        for x, y in dataset.train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Plot and upload gt vs. returned distributions
            if log_dist_plots and batch_nb % logging_settings.log_dist_plot_every == 0:
                log_distribution_plots(model_wrapper, ik_sols_for_distribution_plots, batch_nb=batch_nb, epoch=epoch)

            # Checkpoint
            if batch_nb % checkpoint_every == 0:
                save_checkpoint(
                    model_wrapper,
                    optimizer,
                    training_config,
                    hyper_parameters,
                    dataset_tag,
                    epoch,
                    batch_nb,
                    wandb_run_name,
                )

            # Training step
            batch_t0 = time()
            try:
                throw_kb_interupt = batch_nb == throw_keyboard_interupt_on_batch

                # Set model to train mode
                if model_wrapper.nn_model_type == "multi_cinn":
                    for model in model_wrapper.nn_model:
                        model.train()
                else:
                    model_wrapper.nn_model.train()

                tr_step_results = training_functions[model_wrapper.nn_model_type](
                    model_wrapper,
                    hyper_parameters,
                    training_config,
                    optimizer,
                    params_trainable,
                    x,
                    y,
                    throw_kb_interupt=throw_kb_interupt,
                )

                if tr_step_results.nan_in_loss:
                    print("ERROR: nan_in_loss, exiting()")
                    results_dict = {"epoch": epoch, "batch_nb": batch_nb, "nan_in_loss": True}
                    wandb.log(results_dict)
                    save_model(model_wrapper, dataset, save_settings, wandb_run_name)
                    return

                # Log Training results
                if batch_nb > 0 and batch_nb % logging_settings.log_loss_every_k == 0:
                    tr_step_results.print(epoch, batch_nb, batch_t0)
                    if logging_settings.wandb_enabled:
                        # 'If you name your metrics "prefix/metric-name", we auto-create prefix sections.'
                        results_dict = {"epoch": epoch, "batch_nb": batch_nb}
                        results_dict.update(tr_step_results.get_dict())
                        wandb.log(results_dict)

            except KeyboardInterrupt:
                print(f"Caught keyboard interrupt, breaking")
                save_model(model_wrapper, dataset, save_settings, wandb_run_name)
                return

            except Exception as e:
                print(f"Error caught: '{e}'\n {''.join(traceback.format_tb(e.__traceback__))}")
                if "CUDA out of memory" in str(e):
                    results_dict = {"epoch": epoch, "batch_nb": batch_nb, "cuda_out_of_memory_error": True}
                    wandb.log(results_dict)
                save_model(model_wrapper, dataset, save_settings, wandb_run_name)
                return

            # Evaluate every eval_every_k batches
            if eval_settings.eval_enabled and batch_nb > 0 and batch_nb % logging_settings.eval_every_k == 0:
                results = test_model_accuracy(
                    model_wrapper,
                    dataset,
                    eval_settings.n_poses_per_evaluation,
                    eval_settings.n_samples_per_pose,
                )
                print(results)
                if logging_settings.wandb_enabled:
                    results_dict = {"epoch": epoch, "batch_nb": batch_nb}
                    results_dict.update(results.get_dict())
                    wandb.log(results_dict)

            # Step learning rate every k
            if batch_nb > 0 and batch_nb % training_config.step_lr_every_k == 0:
                weight_scheduler.step()

            batch_nb += 1
        # Step the weight scheduler at the end of every epoch
        epoch += 1

    print(f"\n\nTraining took {round((time() - t_start) / 60.0, 2)} minutes\n")
    save_model(model_wrapper, dataset, save_settings, wandb_run_name)
