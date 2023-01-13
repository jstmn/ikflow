from typing import List, Tuple, Optional, Union
import pickle
from time import time

from ikflow import config
from jkinpylib.robots import Robot
from ikflow.model import IkflowModelParameters
from ikflow.model import glow_cNF_model

import numpy as np
import torch

VERBOSE = False


def draw_latent(user_specified_latent, latent_distribution, latent_scale, shape):
    """Draw a sample from the latent noise distribution for running inference"""
    assert latent_distribution in ["gaussian", "uniform"]
    assert latent_scale > 0
    assert len(shape) == 2
    if user_specified_latent is not None:
        return user_specified_latent
    if latent_distribution == "gaussian":
        return latent_scale * torch.randn(shape).to(config.device)
    elif latent_distribution == "uniform":
        return 2 * latent_scale * torch.rand(shape).to(config.device) - latent_scale


class IKFlowSolver:
    def __init__(self, hyper_parameters: IkflowModelParameters, robot: Robot):
        """Initialize an IKFlowSolver."""
        assert isinstance(hyper_parameters, IkflowModelParameters)
        assert isinstance(robot, Robot)
        self._robot = robot
        self.dim_cond = 7
        if hyper_parameters.softflow_enabled:
            self.dim_cond = 8  # [x, ... q3, softflow_scale]   (softflow_scale should be 0 for inference)
        self._network_width = hyper_parameters.dim_latent_space

        # Note: Changing `nn_model` to `_nn_model` may break the logic in 'download_model_from_wandb_checkpoint.py'
        self.nn_model = glow_cNF_model(hyper_parameters, self._robot, self.dim_cond, self._network_width)
        self.n_dofs = self.robot.n_dofs

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def network_width(self) -> int:
        return self._network_width

    @property
    def conditional_size(self) -> int:
        """Dimensionality of the conditional vector. Without softflow it's 7: [x, y, z, q0, q1, q2, q3]. With softflow
        its 8: [x, ... q3, softflow_scale]  (softflow_scale should be 0 for inference)
        """
        return self.dim_cond

    # TODO(@jstmn): Unit test this function
    def refine_solutions(
        self,
        ikflow_solutions: torch.Tensor,
        target_pose: Union[List[float], np.ndarray],
        positional_tolerance: float = 1e-3,
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
            assert target_pose.shape[0] == b, f"target_pose.shape ({target_pose.shape[0]}) != [{b} x {self.n_dofs}]"

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

        return torch.from_numpy(refined).to(config.device), time() - t0

    def solve(
        self,
        y: List[float],
        n: int,
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1,
        refine_solutions: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Run the network in reverse to generate samples conditioned on a pose y
        Args:
            y (List[float]): Target endpose of the form [x, y, z, q0, q1, q2, q3]
            n (int): The number of solutions to return
            latent Optional[torch.Tensor]: A [ n x network_width ] tensor that contains the n latent vectors to pass
                                            through the network. If None, the latent vectors are drawn from the
                                            distribution 'latent_distribution' with scale 'latent_scale'.
            latent_distribution (str): One of ["gaussian", "uniform"]
            latent_scale (float, optional): The scaling factor for the latent. Latent vectors are multiplied by this value.

        Returns:
            Tuple[torch.Tensor, float]:
                - A [n x n_dofs] batch of IK solutions
                - The runtime of the operation
        """
        t0 = time()
        assert len(y) == 7
        assert latent_distribution in ["gaussian", "uniform"]

        with torch.inference_mode():
            # (:, 0:3) is x, y, z, (:, 3:7) is quat
            conditional = torch.zeros(n, self.dim_cond)
            conditional[:, 0:3] = torch.tensor(y[:3])
            conditional[:, 3 : 3 + 4] = torch.tensor(np.array([y[3:]]))
            # conditional[:, 0:3] = torch.FloatTensor(y[:3])
            # conditional[:, 3 : 3 + 4] = torch.FloatTensor(np.array([y[3:]]))
            conditional = conditional.to(config.device)

            latent = draw_latent(latent, latent_distribution, latent_scale, (n, self._network_width))
            assert latent.shape[0] == n
            assert latent.shape[1] == self._network_width
            output_rev, _ = self.nn_model(latent, c=conditional, rev=True)
            solutions = output_rev[:, 0 : self.n_dofs]
            runtime = time() - t0
            if not refine_solutions:
                return solutions, runtime
        refined, refinement_runtime = self.refine_solutions(solutions, y)
        return refined, runtime + refinement_runtime

    def solve_n_poses(
        self,
        ys: np.array,
        latent: Optional[torch.Tensor] = None,
        latent_distribution: str = "gaussian",
        latent_scale: float = 1,
        refine_solutions: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Same as solve, but for multiple ys.
        ys: [batch x 7]
        """
        t0 = time()
        assert ys.shape[1] == 7
        with torch.inference_mode():
            n = ys.shape[0]

            # Note: No code change required here to handle using/not using softflow.
            conditional = torch.zeros(n, self.dim_cond)
            if isinstance(ys, torch.Tensor):
                # conditional[:, 0:7] = torch.tensor(ys.clone().detach())
                conditional[:, 0:7] = ys
            else:
                conditional[:, 0:7] = torch.tensor(ys)
                # conditional[:, 0:7] = torch.FloatTensor(ys)

            conditional = conditional.to(config.device)

            latent = draw_latent(latent, latent_distribution, latent_scale, (n, self._network_width))
            assert latent.shape[0] == n
            assert latent.shape[1] == self._network_width
            output_rev, _ = self.nn_model(latent, c=conditional, rev=True)
            solutions = output_rev[:, 0 : self.n_dofs]
            runtime = time() - t0
            if not refine_solutions:
                return solutions, runtime
        refined, refinement_runtime = self.refine_solutions(solutions, ys)
        return refined, runtime + refinement_runtime

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""
        with open(state_dict_filename, "rb") as f:
            self.nn_model.load_state_dict(pickle.load(f))
