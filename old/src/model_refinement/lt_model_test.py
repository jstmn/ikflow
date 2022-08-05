import unittest
import os
import sys

sys.path.append(os.getcwd())

from src import config
from src.model import IkflowModel
from src.robot_models import get_robot
from src.model_refinement.lt_model import IkfLitModel
from src.training_parameters import IkflowModelParameters

import torch

torch.manual_seed(0)

ROBOT_MODEL = get_robot("panda_arm")
print("\n___\n")


class LitModelTest(unittest.TestCase):
    def setUp(self) -> None:

        self.hps = IkflowModelParameters()
        self.model_wrapper = IkflowModel(self.hps, ROBOT_MODEL)

        self.model = IkfLitModel(
            self.model_wrapper,
            self.model_wrapper.nn_model,
            self.hps,
            loss_fn="l2",
            learning_rate=1,
            checkpoint_every=-1,
            log_every=5e10,
        )

    def test_gradient_calculated(self):
        batch_size = 16
        batch = (
            torch.randn((batch_size, self.model_wrapper.dim_x)).to(config.device),
            torch.randn((batch_size, self.model_wrapper.dim_y)).to(config.device),
        )
        fixed_latent_noise = torch.randn((5, self.model_wrapper.dim_tot)).to(config.device)
        zero_response0 = self.model_wrapper.make_samples([0] * 7, m=5, latent_noise=fixed_latent_noise)
        optimizer = torch.optim.Adadelta(self.model.nn_model.parameters(), lr=self.model.hparams.learning_rate)

        self.model.zero_grad()
        loss = self.model.training_step(batch, 0)
        loss.backward()
        for name, p in self.model.named_parameters():
            print(name, p.grad is None)

        optimizer.step()
        zero_response_f = self.model_wrapper.make_samples([0] * 7, m=5, latent_noise=fixed_latent_noise)

        print("zero_response0")
        print(zero_response0)

        print("\nzero_response_f")
        print(zero_response_f)

    def test_lit_model_has_params(self):
        n_parameters = len(list(self.model.parameters()))
        # print("self.model n_parameters:                 ", n_parameters)
        self.assertGreater(len(list(self.model.parameters())), 0)


if __name__ == "__main__":

    unittest.main()
