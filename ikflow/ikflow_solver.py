from typing import Tuple, Optional, Union, Callable, Dict
import pickle
from time import time
import warnings

from jrl.math_utils import geodesic_distance_between_quaternions
from jrl.utils import mm_to_m, make_text_green_or_red
from jrl.robots import Robot
import torch

from ikflow.config import DEVICE, DEFAULT_TORCH_DTYPE
from ikflow.model import IkflowModelParameters, glow_cNF_model
from ikflow.evaluation_utils import evaluate_solutions, SOLUTION_EVALUATION_RESULT_TYPE


def draw_latent(
    latent_distribution: str,
    latent_scale: float,
    shape: Tuple[int, int],
    device: str,
):
    """Draw a sample from the latent noise distribution for running inference"""
    assert latent_distribution in ["gaussian", "uniform"]
    assert latent_scale > 0
    assert len(shape) == 2
    if latent_distribution == "gaussian":
        return latent_scale * torch.randn(shape, device=device)
    if latent_distribution == "uniform":
        return 2 * latent_scale * torch.rand(shape, device=device) - latent_scale


class IKFlowSolver:
    def __init__(self, hyper_parameters: IkflowModelParameters, robot: Robot, compile_model: Optional[Dict] = None):
        """Initialize an IKFlowSolver."""
        assert isinstance(
            hyper_parameters, IkflowModelParameters
        ), f"hyper_parameters should be a IkflowModelParameters type, is {type(hyper_parameters)}"
        assert isinstance(robot, Robot), f"robot should be a Robot type, is {type(robot)}"
        assert isinstance(compile_model, (type(None), Dict))

        # Add 'sigmoid_on_output' if IkflowModelParameters instance doesn't have it. This will be the case when loading
        # from training runs from before this parameter was added
        if not hasattr(hyper_parameters, "sigmoid_on_output"):
            hyper_parameters.sigmoid_on_output = False

        if hyper_parameters.softflow_enabled:
            assert not hyper_parameters.sigmoid_on_output, (
                "sigmoid_on_output and softflow are incompatible, disable one or the other"
            )
        self._robot = robot
        self.dim_cond = 7
        if hyper_parameters.softflow_enabled:
            self.dim_cond = 8  # [x, ... q3, softflow_scale]   (softflow_scale should be 0 for inference)
        self._network_width = hyper_parameters.dim_latent_space

        self._do_compile_model = compile_model is not None
        self._model_weights_loaded = False
        self.nn_model = glow_cNF_model(hyper_parameters, self._robot, self.dim_cond, self._network_width)
        if self._do_compile_model:
            warnings.warn(
                "Compiling the ikflow model is not recommended. There are very minor inference speed reduction gains"
                " (~1 ms), at the cost of a ~15s delay on the first model inference."
            )
            self.nn_model = torch.compile(
                self.nn_model, **compile_model
            )  # modes: 'default', 'reduce-overhead' (2 more)

        self.ndof = self.robot.ndof

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def network_width(self) -> int:
        return self._network_width

    @property
    def conditional_size(self) -> int:
        """Dimensionality of the conditional vector. Without softflow it's 7: [x, y, z, q0, q1, q2, q3]. With softflow
        its 8: [x, ... q3, softflow_scale]  (the softflow value should be 0 for inference)
        """
        return self.dim_cond

    def _run_inference(
        self,
        latent: torch.Tensor,
        conditional: torch.Tensor,
        t0: float,
        clamp_to_joint_limits: bool,
        return_detailed: bool,
    ):
        """Run the network."""
        assert latent.shape[0] == conditional.shape[0], f"{len(latent)} != {len(conditional)}"

        # Run model, format and return output
        t0 = time()
        output_rev, _ = self.nn_model(latent, c=conditional, rev=True)
        solutions = output_rev[:, 0 : self.ndof]

        if clamp_to_joint_limits:
            solutions = self.robot.clamp_to_joint_limits(solutions)

        # Format and return output
        if return_detailed:
            pos_errors, rot_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
                self.robot, conditional[:, 0:7], solutions
            )
            return solutions, pos_errors, rot_errors, joint_limits_exceeded, self_collisions, time() - t0
        return solutions

    def _calculate_pose_error(self, qs: torch.Tensor, target_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # check error
        pose_realized = self.robot.forward_kinematics(qs)
        pos_errors = torch.norm(pose_realized[:, 0:3] - target_poses[:, 0:3], dim=1)
        rot_errors = geodesic_distance_between_quaternions(target_poses[:, 3:], pose_realized[:, 3:])
        return pos_errors, rot_errors

    def _generate_exact_ik_solutions(
        self,
        target_poses: torch.Tensor,
        repeat_count: int,
        n_opt_steps_max: int,
        pos_error_threshold: float,
        rot_error_threshold: float,
        printc: Callable[[str], None],
        run_lma_on_cpu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Note: it's faster to run LMA on the cpu, then move the solutions back to the gpu

        prev refers to _generate_exact_ik_solutions_slow()

        Timing (cpu):

            --> n_solutions = 500

                this: "Failed to converge after 3 iterations (410/500 valid) (0.2568323612213135 seconds)"
                prev: "Failed to converge after 3 iterations (410/500 valid) (0.27968668937683105 seconds)" # 8.89854%

            --> n_solutions = 1000

                this: "Failed to converge after 3 iterations (838/1000 valid) (0.37751317024230957 seconds)"
                prev: "Failed to converge after 3 iterations (838/1000 valid) (0.5498697757720947 seconds)" # 45.6558%

            --> n_solutions = 5000

                this: "Failed to converge after 3 iterations (4283/5000 valid) (1.6686820983886719 seconds)"
                prev: "Failed to converge after 3 iterations (4280/5000 valid) (2.366809606552124 seconds)" # 41.8371%

        Timing (cuda):

            --> n_solutions = 1000

                this: 0.23626351356506348 seconds
                    t_lma:   0.13701748847961426
                    t_ikf:   0.010305643081665039
                    t_other: 0.08887577056884766

                prev: 0.5188534259796143 seconds
                    t_lma:   0.23550701141357422
                    t_ikf:   0.009996891021728516
                    t_other: 0.335827112197876
        """

        latent_distribution = "gaussian"
        latent_scale = 1.0

        t0 = time()
        n = target_poses.shape[0]
        n_tiled = n * repeat_count
        device = target_poses.device
        device_0 = device

        do_run_entirely_on_cpu = run_lma_on_cpu and n < 750

        t_lma = 0
        t_ikf = 0

        with torch.inference_mode():
            # Get seeds
            t01 = time()
            conditional = torch.cat(
                [target_poses, torch.zeros((n, 1), dtype=DEFAULT_TORCH_DTYPE, device=device)], dim=1
            )
            conditional_tiled = conditional.repeat((repeat_count, 1))
            target_poses_tiled = conditional_tiled[:, 0:7]
            latent = draw_latent(latent_distribution, latent_scale, (n_tiled, self._network_width), device)
            q = self._run_inference(latent, conditional_tiled, t0, True, False)
            t_ikf += time() - t01

            if run_lma_on_cpu and do_run_entirely_on_cpu:
                q = q.cpu()
                target_poses_tiled = target_poses_tiled.cpu()
                device = "cpu"

            t02 = time()
            final_solutions = torch.zeros(n, self.ndof, dtype=torch.float32, device=device)
            final_valids = torch.zeros(n, dtype=torch.bool, device=device)
            n_invalid = n

            for i in range(n_opt_steps_max):
                assert len(q) == n_invalid * repeat_count
                t02 = time()
                if run_lma_on_cpu and not do_run_entirely_on_cpu:
                    q = self.robot.inverse_kinematics_step_levenburg_marquardt(target_poses_tiled.cpu(), q.cpu())
                    q = q.to(device)
                else:
                    q = self.robot.inverse_kinematics_step_levenburg_marquardt(target_poses_tiled, q)
                t_lma += time() - t02
                pos_errors, rot_errors = self._calculate_pose_error(q, target_poses_tiled)
                valids_i_tiled = torch.logical_and(pos_errors < pos_error_threshold, rot_errors < rot_error_threshold)

                # TODO: use torch.mod()
                valids_i = torch.zeros(n_invalid, dtype=torch.bool, device=device)
                sols_i = torch.zeros((n_invalid, self.ndof), dtype=torch.float32, device=device)

                valid_idxs = torch.nonzero(valids_i_tiled)
                for j in range(valid_idxs.shape[0]):
                    idx = valid_idxs[j, 0]
                    sol_idx = idx % n_invalid
                    sols_i[sol_idx, :] = q[idx, :]
                    valids_i[sol_idx] = True

                final_solutions[torch.logical_not(final_valids)] = sols_i
                final_valids[torch.logical_not(final_valids)] = valids_i

                if final_valids.all():
                    printc(make_text_green_or_red(f"Converged after {i + 1} iterations ({time() - t0} seconds)", True))
                    return final_solutions.to(device_0), final_valids.to(device_0)

                q = q[torch.logical_not(valids_i).repeat((repeat_count)), :]
                target_poses_tiled = target_poses_tiled[torch.logical_not(valids_i).repeat((repeat_count)), :]
                n_invalid = n - final_valids.sum().item()

            # Didn't converge
            t_other = time() - t0 - t_lma - t_ikf
            printc("  t_lma:  ", t_lma)
            printc("  t_ikf:  ", t_ikf)
            printc("  t_other:", t_other)
            printc(
                make_text_green_or_red(
                    f"\nFailed to converge after {i + 1} iterations"
                    f" ({final_valids.sum().item()}/{final_valids.numel()} valid) ({time() - t0} seconds)",
                    False,
                )
            )
            return final_solutions.to(device_0), final_valids.to(device_0)

    # ------------------------------------------------------------------------------------------------------------------
    # --- Public methods
    #

    # TODO: refactor to generate_approximate_ik_solutions_single_pose() and generate_approximate_ik_solutions_multi_pose()
    def generate_ik_solutions(
        self,
        y: torch.Tensor,
        n: Optional[int] = None,
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1.0,
        clamp_to_joint_limits: bool = True,
        refine_solutions: bool = False,
        return_detailed: bool = False,
        allow_uninitialized: bool = False,
    ) -> Union[torch.Tensor, SOLUTION_EVALUATION_RESULT_TYPE]:
        """Run the network in reverse to generate samples conditioned on a pose y

        Example usages, single target pose:
            solutions = ikflow_solver.generate_ik_solutions(target_pose, number_of_solutions, return_detailed=False)
            solutions, pos_errors, rot_errors, joint_limits_exceeded, self_colliding, runtime = ikflow_solver.generate_ik_solutions(
                target_pose, number_of_solutions, return_detailed=True
            )
        Example usages, batched target poses:
            solutions = ikflow_solver.generate_ik_solutions(target_poses, None, return_detailed=False)
            solutions, pos_errors, rot_errors, joint_limits_exceeded, self_colliding, runtime = ikflow_solver.generate_ik_solutions(
                target_pose, number_of_solutions, return_detailed=True
            )

        Args:
            y (List[float]): A torch.Tensor that is either a single target pose with dimension [7], with the form
                                [x, y, z, q0, q1, q2, q3], or a batch of target poses with dimensions [batch x 7]
            n (int): The number of solutions to return if y is a single pose
            latent Optional[torch.Tensor]: A [ n x network_width ] tensor that contains the n latent vectors to pass
                                            through the network. If None, the latent vectors are drawn from the
                                            distribution 'latent_distribution' with scale 'latent_scale'.
            latent_distribution (str): One of ["gaussian", "uniform"]
            latent_scale (float, optional): The scaling factor for the latent. Latent vectors are multiplied by this
                                            value.
            clamp_to_joint_limits (bool): Clamp the generated IK solutions to within the joint limits of the robot. May
                                            increase the error of the solutions.
            refine_solutions (bool): Optimize the solutions using a traditional gradient descent based IK solver.
            return_detailed (bool): Flag to return stats about the solutions. See returns below for more details.

        Returns:
            If return_detailed is False:
                - torch.Tensor: A [n x n_dofs] batch of IK solutions
            Else:
                - torch.Tensor: [n x n_dofs] tensor of IK solutions.
                - torch.Tensor: [n] tensor of positional errors of the IK solutions. The error is the L2 norm of the
                                    realized poses of the solutions and the target pose.
                - torch.Tensor: [n] tensor of angular errors of the IK solutions. The error is the geodesic distance
                                    between the realized orientation of the robot's end effector from the IK solutions
                                    and the targe orientation.
                - torch.Tensor: [n] tensor of bools indicating whether each IK solutions has exceeded the robots joint
                                    limits.
                - torch.Tensor: [n] tensor of bools indicating whether each IK solutions is self colliding.
                - float: The runtime of the operation
        """
        t0 = time()
        if not allow_uninitialized:
            assert self._model_weights_loaded, "Model weights have not been loaded. Call load_state_dict(...)"
        assert isinstance(y, torch.Tensor), f"y must be a torch.Tensor (got {type(y)})."
        if y.numel() == 7:
            assert isinstance(n, int)
            assert n > 0
        else:
            assert y.shape[1] == 7, f"y must be of shape [7] or [n x 7], got {y.shape}"

        assert isinstance(latent_distribution, str)
        assert isinstance(latent_scale, float)
        assert isinstance(latent, torch.Tensor) or (
            latent is None
        ), f"latent must either be a torch.Tensor or None (got {type(latent)})."
        assert not refine_solutions, "refine_solutions is deprecated, use generate_exact_ik_solutions() instead"
        if "cuda" in str(DEVICE):
            assert "cpu" not in str(y.device), f"Cuda is available ('{DEVICE}'), but target_poses are on {y.device}"

        n = y.shape[0] if n is None else n
        device = y.device

        with torch.inference_mode():
            # Get conditional
            if y.numel() == 7:
                conditional = torch.cat(
                    [y.expand((n, 7)), torch.zeros((n, 1), dtype=DEFAULT_TORCH_DTYPE, device=device)], dim=1
                )
            else:
                conditional = torch.cat([y, torch.zeros((n, 1), dtype=DEFAULT_TORCH_DTYPE, device=device)], dim=1)

            # Get latent
            if latent is None:
                latent = draw_latent(latent_distribution, latent_scale, (n, self._network_width), device)
            return self._run_inference(latent, conditional, t0, clamp_to_joint_limits, return_detailed)

    def generate_exact_ik_solutions(
        self,
        target_poses: torch.Tensor,
        repeat_counts: Tuple[int] = (1, 3, 10),
        pos_error_threshold: float = mm_to_m(1),
        rot_error_threshold: float = 0.1,
        verbosity: int = 0,
        run_lma_on_cpu: bool = True,
        return_detailed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as generate_ik_solutions() but refines solutions using Levenberg-Marquardt after they're generated.

        NOTE: returned solutions may be self colliding
        """
        assert target_poses.shape[1] == 7, f"target_poses must be of shape [n x 7], got {target_poses.shape}"
        assert isinstance(repeat_counts, tuple), f"repeat_counts must be a tuple, got {type(repeat_counts)}"
        assert not return_detailed, "return_detailed is not currently supported for generate_exact_ik_solutions()"
        assert self._model_weights_loaded, "Model weights have not been loaded. Call load_state_dict(...)"
        t0 = time()
        n_opt_steps_max = 3  # repeat_count = 3 ->  s, repeat_count = 2 ->  s
        n_retries = len(repeat_counts)

        def printc(s, *args, **kwargs):
            if verbosity > 0:
                print(s, *args, **kwargs)

        with torch.inference_mode():
            repeat_count = repeat_counts[0]
            solutions, valids = self._generate_exact_ik_solutions(
                target_poses,
                repeat_count,
                n_opt_steps_max,
                pos_error_threshold,
                rot_error_threshold,
                printc,
                run_lma_on_cpu,
            )

            if valids.all():
                printc("All solutions converged, returning")
                return solutions, valids

            for i in range(1, n_retries):
                retry_repeat_count = repeat_counts[i]
                missing_target_poses = target_poses[torch.logical_not(valids), :]
                new_solutions, new_solution_valids = self._generate_exact_ik_solutions(
                    missing_target_poses,
                    retry_repeat_count,
                    n_opt_steps_max,
                    pos_error_threshold,
                    rot_error_threshold,
                    printc,
                    run_lma_on_cpu,
                )
                solutions[torch.logical_not(valids), :] = new_solutions
                valids[torch.logical_not(valids)] = new_solution_valids  # update the valid idxs

                if new_solutions.all():
                    printc(
                        make_text_green_or_red(
                            f"All missing target poses solutions converged, returning ({time() - t0} sec)", True
                        )
                    )
                    return solutions, valids

            printc(make_text_green_or_red(f"Missing target poses not found, returning ({time() - t0} sec)", False))
            return solutions, valids

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""

        with open(state_dict_filename, "rb") as f:
            try:
                state_dict = pickle.load(f)

                if self._do_compile_model:
                    # the keys in compiled models are prepended with '_orig_mod.' for some reason i'm not aware of
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        assert not k.startswith("_orig_mod.")
                        new_state_dict[f"_orig_mod.{k}"] = v
                    state_dict = new_state_dict

                self.nn_model.load_state_dict(state_dict)
                self._model_weights_loaded = True

                # Generate solutions so that torch.compile can do its thing
                if self._do_compile_model:
                    y = torch.randn(500, 7, device=DEVICE, dtype=DEFAULT_TORCH_DTYPE)
                    for i in [10, 20, 50, 100, 75, 150, 499]:
                        t0 = time()
                        self.generate_ik_solutions(y[0:i])
                        print(f"Ran generate_ik_solutions(y[0:{i}], 7) in {time()-t0:5f}")

            except pickle.UnpicklingError as e:
                print(f"Error loading state dict from {state_dict_filename}: {e}")
                raise e
