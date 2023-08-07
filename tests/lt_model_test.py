import unittest


from ikflow import config
from ikflow.ikflow_solver import IKFlowSolver
from jrl.robots import get_robot
from ikflow.training.lt_model import IkfLitModel
from ikflow.model import IkflowModelParameters

import torch

torch.manual_seed(0)

ROBOT_MODEL = get_robot("panda")


class LitModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hps = IkflowModelParameters()
        self.ik_solver = IKFlowSolver(self.hps, ROBOT_MODEL)
        self.model = IkfLitModel(
            self.ik_solver, self.hps, learning_rate=1.0, checkpoint_every=-1, optimizer_name="adamw"
        )

    def test_gradient_calculated(self):
        batch_size = 10
        batch = (
            torch.randn((batch_size, self.ik_solver.robot.ndof)).to(config.device),
            torch.randn((batch_size, 7)).to(config.device),
        )
        fixed_latent = torch.randn((5, self.ik_solver.network_width)).to(config.device)
        zero_response0 = self.ik_solver.solve([0] * 7, n=5, latent=fixed_latent)
        optimizer = torch.optim.Adadelta(self.model.nn_model.parameters(), lr=self.model.hparams.learning_rate)

        self.model.zero_grad()
        loss = self.model.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        zero_response_f = self.ik_solver.solve([0] * 7, n=5, latent=fixed_latent)
        self.assertFalse(torch.allclose(zero_response0, zero_response_f))

    def test_lit_model_has_params(self):
        n_parameters = len(list(self.model.parameters()))
        self.assertGreater(n_parameters, 0)


if __name__ == "__main__":
    unittest.main()
