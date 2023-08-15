from typing import List, Tuple, Optional, Union
import pickle
from time import time

from jrl.robots import Robot
import numpy as np
import torch

from ikflow import config
from ikflow.config import DEFAULT_TORCH_DTYPE
from ikflow.model import IkflowModelParameters, glow_cNF_model
from ikflow.evaluation_utils import evaluate_solutions, SOLUTION_EVALUATION_RESULT_TYPE

def draw_latent(
    user_specified_latent: torch.Tensor, latent_distribution: str, latent_scale: float, shape: Tuple[int, int]
):
    """Draw a sample from the latent noise distribution for running inference"""
    assert latent_distribution in ["gaussian", "uniform"]
    assert latent_scale > 0
    assert len(shape) == 2
    if user_specified_latent is not None:
        return user_specified_latent
    if latent_distribution == "gaussian":
        return latent_scale * torch.randn(shape).to(config.device)
    if latent_distribution == "uniform":
        return 2 * latent_scale * torch.rand(shape).to(config.device) - latent_scale


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

    # TODO(@jstmn): Unit test this function
    def refine_solutions(
        self,
        ikflow_solutions: torch.Tensor,
        target_pose: Union[List[float], np.ndarray],
        positional_tolerance: float = 1e-3,
        out_device=config.device,
    ) -> Tuple[torch.Tensor, float]:
        """Refine a batch of IK solutions using the klampt IK solver
        Args:
            ikflow_solutions (torch.Tensor): A batch of IK solutions of the form [batch x n_dofs]
            target_pose (Union[List[float], np.ndarray]): The target endpose(s). Must either be of the form
                                                            [x, y, z, q0, q1, q2, q3] or be a [batch x 7] numpy array
        Returns:
            torch.Tensor: A batch of IK refined solutions [batch x n_dofs]
        """
        t0 = time()
        b = ikflow_solutions.shape[0]
        if isinstance(target_pose, list):
            target_pose = np.array(target_pose)
        if isinstance(target_pose, np.ndarray) and len(target_pose.shape) == 2:
            assert target_pose.shape[0] == b, f"target_pose.shape ({target_pose.shape[0]}) != [{b} x {self.ndof}]"

        ikflow_solutions_np = ikflow_solutions.detach().cpu().numpy()
        refined = ikflow_solutions_np.copy()
        is_single_pose = (len(target_pose.shape) == 1) or (target_pose.shape[0] == 1)
        pose = target_pose

        for i in range(b):
            if not is_single_pose:
                pose = target_pose[i]
            ik_sol = self._robot.inverse_kinematics_klampt(
                pose, seed=ikflow_solutions_np[i], positional_tolerance=positional_tolerance
            )
            if ik_sol is not None:
                refined[i] = ik_sol

        return torch.tensor(refined, device=out_device, dtype=DEFAULT_TORCH_DTYPE)

    def _solve_return_helper(
        self,
        conditional: torch.Tensor,
        latent: torch.Tensor,
        clamp_to_joint_limits: bool,
        refine_solutions: bool,
        return_detailed: bool,
        t0: float,
    ):
        """Internal function to call IKFlow, and format and return the output"""
        assert latent.shape[0] == conditional.shape[0]
        assert latent.shape[1] == self._network_width
        assert conditional.shape[1] == self.dim_cond

        # Run model
        output_rev, _ = self.nn_model(latent, c=conditional, rev=True)
        solutions = output_rev[:, 0 : self.ndof]

        if clamp_to_joint_limits:
            solutions = self.robot.clamp_to_joint_limits(solutions)

        # Refine if requested
        if refine_solutions:
            target_pose_s = conditional.detach().cpu().numpy()[:, 0:7]
            solutions = self.refine_solutions(solutions, target_pose_s)

        # Format and return output
        if return_detailed:
            l2_errors, angular_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
                self.robot, conditional[:, 0:7], solutions
            )
            return solutions, l2_errors, angular_errors, joint_limits_exceeded, self_collisions, time() - t0
        return solutions

    def _validate_solve_input(
        self,
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1.0,
        refine_solutions: bool = False,
        return_detailed: bool = False,
    ):
        assert isinstance(latent_distribution, str)
        assert isinstance(latent_scale, float)
        assert isinstance(latent, torch.Tensor) or (
            latent is None
        ), f"latent must either be a torch.Tensor or None (got {type(latent)})."
        assert isinstance(refine_solutions, bool)
        assert isinstance(return_detailed, bool)

    # ------------------------------------------------------------------------------------------------------------------
    # --- Public methods
    #

    def solve(
        self,
        y: List[float],
        n: int,
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1.0,
        clamp_to_joint_limits: bool = True,
        refine_solutions: bool = False,
        return_detailed: bool = False,
        device: str = config.device,
    ) -> Union[torch.Tensor, SOLUTION_EVALUATION_RESULT_TYPE]:
        """Run the network in reverse to generate samples conditioned on a pose y

        Example usage(s):
            solutions = ikflow_solver.solve(
                target_pose, number_of_solutions, refine_solutions=True, return_detailed=False
            )
            solutions, l2_errors, angular_errors, joint_limits_exceeded, self_colliding, runtime = ikflow_solver.solve(
                target_pose, number_of_solutions, refine_solutions=False, return_detailed=True
            )

        Args:
            y (List[float]): Target endpose of the form [x, y, z, q0, q1, q2, q3]
            n (int): The number of solutions to return
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
        assert len(y) == 7, (
            f"IKFlowSolver.solve() expects a 1d list or numpy array of length 7 (got {y}). This means you probably want"
            " to call solve_n_poses() instead"
        )
        self._validate_solve_input(latent, latent_distribution, latent_scale, refine_solutions, return_detailed)

        # TODO: Figure out how to use this as a decorator. Currently getting the error 'TypeError: __call__() takes 2
        # positional arguments but 6 were given'
        with torch.inference_mode():
            # Get latent
            # (:, 0:3) is x, y, z, (:, 3:7) is quat
            conditional = torch.zeros(n, self.dim_cond, device=device, dtype=DEFAULT_TORCH_DTYPE)
            conditional[:, 0:3] = torch.tensor(y[:3])
            conditional[:, 3 : 3 + 4] = torch.tensor(y[3:])

            # Get latent
            latent = draw_latent(latent, latent_distribution, latent_scale, (n, self._network_width))

            # Run model, format and return output
            return self._solve_return_helper(
                conditional, latent, clamp_to_joint_limits, refine_solutions, return_detailed, t0
            )

    def solve_n_poses(
        self,
        ys: np.array,
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1.0,
        clamp_to_joint_limits: bool = True,
        refine_solutions: bool = False,
        return_detailed: bool = False,
        device: str = config.device,
    ) -> Union[torch.Tensor, SOLUTION_EVALUATION_RESULT_TYPE]:
        """Same as solve(), but with a batch of target poses.
        ys: [batch x 7]
        """
        t0 = time()
        assert ys.shape[1] == 7
        self._validate_solve_input(latent, latent_distribution, latent_scale, refine_solutions, return_detailed)
        n = ys.shape[0]

        # TODO: Figure out how to use this as a decorator. Currently getting the error 'TypeError: __call__() takes 2
        # positional arguments but 6 were given'
        with torch.inference_mode():
            # Get conditional
            # Note: No code change required here to handle using/not using softflow.
            conditional = torch.zeros(n, self.dim_cond, dtype=torch.float32, device=device)
            if isinstance(ys, torch.Tensor):
                conditional[:, 0:7] = ys
            else:
                conditional[:, 0:7] = torch.tensor(ys, dtype=torch.float32)

            # Get latent
            latent = draw_latent(latent, latent_distribution, latent_scale, (n, self._network_width))

            # Run model, format and return output
            return self._solve_return_helper(
                conditional, latent, clamp_to_joint_limits, refine_solutions, return_detailed, t0
            )

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""
        with open(state_dict_filename, "rb") as f:
            try:
                self.nn_model.load_state_dict(pickle.load(f))
            except pickle.UnpicklingError as e:
                print(f"Error loading state dict from {state_dict_filename}: {e}")
                raise e
