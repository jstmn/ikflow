from typing import List, Tuple, Optional
import os
import pickle
from time import time

import config
from src.robots import RobotModel
from src.supporting_types import IkflowModelParameters
from src.model import glow_cNF_model

import numpy as np
import torch

VERBOSE = False


def draw_latent_noise(user_specified_latent_noise, latent_noise_distribution, latent_noise_scale, shape):
    """Draw a sample from the latent noise distribution for running inference"""
    assert latent_noise_distribution in ["gaussian", "uniform"]
    assert latent_noise_scale > 0
    assert len(shape) == 2
    if user_specified_latent_noise is not None:
        return user_specified_latent_noise
    if latent_noise_distribution == "gaussian":
        return latent_noise_scale * torch.randn(shape).to(config.device)
    elif latent_noise_distribution == "uniform":
        return 2 * latent_noise_scale * torch.rand(shape).to(config.device) - latent_noise_scale


class IkflowSolver:
    def __init__(self, hyper_parameters: IkflowModelParameters, robot_model: RobotModel):
        """Initialize an IkflowSolver."""
        assert isinstance(hyper_parameters, IkflowModelParameters)

        self.dim_cond = 7  # [x, y, z, q0, q1, q2, q3]
        if hyper_parameters.softflow_enabled:
            self.dim_cond = 8  # [x, ... q3, softflow_scale]   (softflow_scale should be 0 for inference)
        self.network_width = hyper_parameters.dim_latent_space
        self.nn_model = glow_cNF_model(hyper_parameters, robot_model, self.dim_cond, self.network_width)
        self.ndofs = robot_model.ndofs
        self._robot_model = robot_model

    @property
    def robot(self) -> RobotModel:
        return self._robot_model

    def make_samples(
        self,
        y: List[float],
        m: int,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        """Run the network in reverse to generate samples conditioned on a pose y

        Args:
            y (List[float]): Target endpose of the form [x, y, z, q0, q1, q2, q3]
            m (int): The number of samples to draw
            latent_noise (Optional[torch.Tensor], optional): A batch of [batch x network_widthal] latent noise vectors.
                                                                Note that `y` and `m` will be ignored if this variable
                                                                is passed. Defaults to None.
            latent_noise_distribution (str): One of ["gaussian", "uniform"]
            latent_noise_scale (float, optional): The scaling factor for the latent noise samples. Samples from the
                                                    gaussian latent space are multiplied by this value.
        Returns:
            Tuple[torch.Tensor, float]:
                - A [batch x ndofs] batch of IK solutions
                - The runtime of the operation
        """
        assert len(y) == 7
        assert latent_noise_distribution in ["gaussian", "uniform"]

        # (:, 0:3) is x, y, z, (:, 3:7) is quat
        conditional = torch.zeros(m, self.dim_cond)
        conditional[:, 0:3] = torch.FloatTensor(y[:3])
        conditional[:, 3 : 3 + 4] = torch.FloatTensor(np.array([y[3:]]))
        conditional = conditional.to(config.device)

        latent_noise = draw_latent_noise(
            latent_noise, latent_noise_distribution, latent_noise_scale, (m, self.network_width)
        )
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.network_width
        t0 = time()
        output_rev, jac = self.nn_model(latent_noise, c=conditional, rev=True)
        return output_rev[:, 0 : self.ndofs], time() - t0

    def make_samples_mult_y(
        self,
        ys: np.array,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        """Same as make_samples, but for multiple ys.

        ys: [batch x 7]
        """
        assert ys.shape[1] == 7

        ys = torch.FloatTensor(ys)
        m = ys.shape[0]

        # Note: No code change required here to handle using/not using softflow.
        conditional = torch.zeros(m, self.dim_cond)
        conditional[:, 0:7] = ys
        conditional = conditional.to(config.device)
        shape = (m, self.network_width)
        latent_noise = draw_latent_noise(latent_noise, latent_noise_scale, shape)
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.network_width
        t0 = time()
        output_rev, jac = self.nn_model(latent_noise, c=conditional, rev=True)
        return output_rev[:, 0 : self.ndofs], time() - t0

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""
        with open(state_dict_filename, "rb") as f:
            self.nn_model.load_state_dict(pickle.load(f))
