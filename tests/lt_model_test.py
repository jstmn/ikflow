import unittest
import os
import sys

sys.path.append(os.getcwd())

import config
from src.ik_solvers import IkflowSolver
from src.robots import get_robot
from src.lt_model import IkfLitModel
from src.training_parameters import IkflowModelParameters

import torch

torch.manual_seed(0)

ROBOT_MODEL = get_robot("panda_arm")


class LitModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hps = IkflowModelParameters()
        self.ik_solver = IkflowSolver(self.hps, ROBOT_MODEL)

        self.model = IkfLitModel(self.ik_solver, self.hps, learning_rate=1.0, checkpoint_every=-1)

    def test_gradient_calculated(self):
        batch_size = 10
        batch = (
            torch.randn((batch_size, self.ik_solver.dim_x)).to(config.device),
            torch.randn((batch_size, self.ik_solver.dim_y)).to(config.device),
        )
        fixed_latent_noise = torch.randn((5, self.ik_solver.dim_tot)).to(config.device)
        zero_response0 = self.ik_solver.make_samples([0] * 7, m=5, latent_noise=fixed_latent_noise)[0]
        optimizer = torch.optim.Adadelta(self.model.nn_model.parameters(), lr=self.model.hparams.learning_rate)

        self.model.zero_grad()
        loss = self.model.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        zero_response_f = self.ik_solver.make_samples([0] * 7, m=5, latent_noise=fixed_latent_noise)[0]
        self.assertFalse(torch.allclose(zero_response0, zero_response_f))

    def test_lit_model_has_params(self):
        n_parameters = len(list(self.model.parameters()))
        self.assertGreater(n_parameters, 0)


if __name__ == "__main__":
    unittest.main()
