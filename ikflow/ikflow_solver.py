from typing import Tuple, Optional, Union
import pickle
from time import time

from jrl.math_utils import geodesic_distance_between_quaternions
from jrl.utils import mm_to_m, make_text_green_or_red
from jrl.robots import Robot
import torch

from ikflow import config
from ikflow.config import DEFAULT_TORCH_DTYPE
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
    def __init__(self, hyper_parameters: IkflowModelParameters, robot: Robot):
        """Initialize an IKFlowSolver."""
        assert isinstance(
            hyper_parameters, IkflowModelParameters
        ), f"Error - hyper_parameters should be IkflowModelParameters type, is {type(hyper_parameters)}"
        assert isinstance(robot, Robot), f"Error - robot should be Robot type, is {type(robot)}"

        # Add 'sigmoid_on_output' if IkflowModelParameters instance doesn't have it. This will be the case when loading
        # from training runs from before this parameter was added
        if not hasattr(hyper_parameters, "sigmoid_on_output"):
            hyper_parameters.sigmoid_on_output = False

        if hyper_parameters.softflow_enabled:
            assert (
                not hyper_parameters.sigmoid_on_output
            ), f"sigmoid_on_output and softflow are incompatible, disable one or the other"
        self._robot = robot
        self.dim_cond = 7
        if hyper_parameters.softflow_enabled:
            self.dim_cond = 8  # [x, ... q3, softflow_scale]   (softflow_scale should be 0 for inference)
        self._network_width = hyper_parameters.dim_latent_space

        # Note: Changing `nn_model` to `_nn_model` may break the logic in 'download_model_from_wandb_checkpoint.py'
        self.nn_model = glow_cNF_model(hyper_parameters, self._robot, self.dim_cond, self._network_width)
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

        """Refine a batch of IK solutions using the klampt IK solver
        Args:
            ikflow_solutions (torch.Tensor): A batch of IK solutions of the form [batch x n_dofs]
            target_pose (Union[List[float], np.ndarray]): The target endpose(s). Must either be of the form
                                                            [x, y, z, q0, q1, q2, q3] or be a [batch x 7] numpy array
        Returns:
            torch.Tensor: A batch of IK refined solutions [batch x n_dofs]
        """

    def _run_inference(
        self,
        latent: torch.Tensor,
        conditional: torch.Tensor,
        t0: float,
        clamp_to_joint_limits: bool,
        return_detailed: bool,
        verbsosity: int = 0,
    ):
        """Run the network."""
        assert latent.shape[0] == conditional.shape[0], f"{len(latent)} != {len(conditional)}"

        # Run model, format and return output
        t0 = time()
        output_rev, _ = self.nn_model(latent, c=conditional, rev=True)
        solutions = output_rev[:, 0 : self.ndof]
        if verbsosity > 0:
            print(f"generated {len(solutions)} solutions in {time() - t0:.5f} seconds")

        if clamp_to_joint_limits:
            solutions = self.robot.clamp_to_joint_limits(solutions)

        # Format and return output
        if return_detailed:
            pos_errors, rot_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
                self.robot, conditional[:, 0:7], solutions
            )
            return solutions, pos_errors, rot_errors, joint_limits_exceeded, self_collisions, time() - t0
        return solutions

    # ------------------------------------------------------------------------------------------------------------------
    # --- Public methods
    #

    def generate_ik_solutions(
        self,
        y: torch.Tensor,
        n: Optional[int],
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1.0,
        clamp_to_joint_limits: bool = True,
        refine_solutions: bool = False,
        return_detailed: bool = False,
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
        assert isinstance(y, torch.Tensor), f"y must be a torch.Tensor (got {type(y)})."
        if y.numel() == 7:
            print("numel == 7")
            assert isinstance(n, int)
            assert n > 0
        else:
            assert y.shape[1] == 7, f"y must be of shape [7] or [n x 7], got {y.shape}"

        assert isinstance(latent_distribution, str)
        assert isinstance(latent_scale, float)
        assert isinstance(latent, torch.Tensor) or (
            latent is None
        ), f"latent must either be a torch.Tensor or None (got {type(latent)})."
        assert not refine_solutions, f"refine_solutions is deprecated, use generate_exact_ik_solutions() instead"
        if "cuda" in str(config.device):
            assert "cpu" not in str(
                y.device
            ), "Use cuda if available. Note: you need to run self.nn_model.to('cpu') if you really want to use the cpu"

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

    def _check_converged(self, qs, target_poses, pos_error_threshold, rot_error_threshold):
        # check error
        pose_realized = self.robot.forward_kinematics_batch(qs)
        pos_errors = torch.norm(pose_realized[:, 0:3] - target_poses[:, 0:3], dim=1)
        rot_errors = geodesic_distance_between_quaternions(target_poses[:, 3:], pose_realized[:, 3:])
        converged = pos_errors.max() < pos_error_threshold and rot_errors.max() < rot_error_threshold
        return converged, pos_errors, rot_errors

    def _generate_exact_ik_solutions(
        self,
        target_poses: torch.Tensor,
        repeat_count: int,
        n_opt_steps_max: int,
        pos_error_threshold: float,
        rot_error_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: it may be faster to run LMA on the cpu, then move the solutions back to the gpu"""

        latent_distribution = "gaussian"
        latent_scale = 1.0

        t0 = time()
        n = target_poses.shape[0]
        n_tiled = n * repeat_count
        device = target_poses.device

        with torch.inference_mode():

            # Get conditional
            conditional = torch.cat(
                [target_poses, torch.zeros((n, 1), dtype=DEFAULT_TORCH_DTYPE, device=device)], dim=1
            )
            conditional_tiled = conditional.repeat((repeat_count, 1))
            target_poses_tiled = conditional_tiled[:, 0:7]

            # Get latent
            latent = draw_latent(latent_distribution, latent_scale, (n_tiled, self._network_width), device)

            # Get seeds
            solutions_0 = self._run_inference(latent, conditional_tiled, t0, True, False, verbsosity=1)
            q = solutions_0.clone()
            # print("solutions_0:", solutions_0)

            # assert False, "todo: remove converged idxs from q after each iteration"
            converged = False
            for i in range(n_opt_steps_max):
                # print("\ni:", i)

                # qprev = q.clone()
                q = self.robot.inverse_kinematics_single_step_levenburg_marquardt(target_poses_tiled, q)
                # print("diff:", q - qprev)

                _, pos_errors, rot_errors = self._check_converged(
                    q, target_poses_tiled, pos_error_threshold, rot_error_threshold
                )
                valids = torch.logical_and(pos_errors < pos_error_threshold, rot_errors < rot_error_threshold)
                # print(" ", valids)
                # print(" ", pos_errors)
                # print(" ", rot_errors)
                # print("rot_error_threshold:", rot_error_threshold)
                # print()

                converged_notes = torch.zeros(n, dtype=torch.bool, device=device)
                for j in range(n):
                    idxs = j + torch.arange(0, n_tiled, n, device=device)
                    # print(j, idxs)
                    assert idxs.numel() == repeat_count
                    valids_j = valids[idxs]
                    converged_notes[j] = valids_j.any().item()

                # print("converged_notes:", converged_notes.sum(), "/", converged_notes.numel())
                converged = converged_notes.all().item()
                if converged:
                    break

            # Didn't converge
            if not converged:
                print(
                    make_text_green_or_red(
                        f"\nFailed to converge after {i+1} iterations"
                        f" ({converged_notes.sum().item()}/{converged_notes.numel()} valid) ({time() - t0} seconds)",
                        False,
                    )
                )
                # return q[0:n, :]
            else:
                print(make_text_green_or_red(f"Converged after {i+1} iterations ({time() - t0} seconds)", True))

            valid_idxs = torch.nonzero(valids)  # torch.Size([6, 1])
            sols = torch.zeros(n, self.ndof, dtype=torch.float32, device=device)

            for i in range(valid_idxs.shape[0]):
                idx = valid_idxs[i, 0]
                sol_idx = idx % n
                # print(i, idx, sol_idx)
                sols[sol_idx, :] = q[idx, :]
            return sols, converged_notes

    def generate_exact_ik_solutions(
        self,
        target_poses: torch.Tensor,
        repeat_counts: Tuple[int] = (1, 3, 15),
        pos_error_threshold: float = mm_to_m(1),
        rot_error_threshold: float = 0.1,
        return_detailed: bool = False,
    ) -> Union[torch.Tensor, SOLUTION_EVALUATION_RESULT_TYPE]:
        """Same as generate_ik_solutions() but optimizes solutions using Levenberg-Marquardt after they're generated.

        Note: Currently y must be [batch x 7]
        """
        assert target_poses.shape[1] == 7, f"target_poses must be of shape [n x 7], got {target_poses.shape}"
        assert not return_detailed, f"return_detailed is not supported for generate_exact_ik_solutions()"

        t0 = time()
        n_opt_steps_max = 3  # repeat_count = 3 ->  s, repeat_count = 2 ->  s

        n_retries = len(repeat_counts)

        with torch.inference_mode():

            repeat_count = repeat_counts[0]
            solutions, valids = self._generate_exact_ik_solutions(
                target_poses, repeat_count, n_opt_steps_max, pos_error_threshold, rot_error_threshold
            )
            if valids.all():
                print("All solutions converged, returning")
                return solutions

            for i in range(1, n_retries):
                retry_repeat_count = repeat_counts[i]
                print("Some solutions did not converge, trying again with new repeat_count")
                print(f"retry_repeat_count: {retry_repeat_count}")

                missing_target_poses = target_poses[torch.logical_not(valids), :]
                print("missing_target_poses:", missing_target_poses.shape)

                new_solutions, new_solution_valids = self._generate_exact_ik_solutions(
                    missing_target_poses, retry_repeat_count, n_opt_steps_max, pos_error_threshold, rot_error_threshold
                )
                solutions[torch.logical_not(valids), :] = new_solutions
                valids[torch.logical_not(valids)] = new_solution_valids  # update the valid idxs

                if new_solutions.all():
                    print(
                        make_text_green_or_red(
                            f"All missing target poses solutions converged, returning ({time() - t0} sec)", True
                        )
                    )
                    return solutions

            print(make_text_green_or_red(f"Missing target poses not found, returning ({time() - t0} sec)", False))
            assert False

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""
        with open(state_dict_filename, "rb") as f:
            try:
                self.nn_model.load_state_dict(pickle.load(f))
            except pickle.UnpicklingError as e:
                print(f"Error loading state dict from {state_dict_filename}: {e}")
                raise e
