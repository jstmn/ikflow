import unittest

import torch

from ikflow.model import glow_cNF_model, IkflowModelParameters
from jkinpylib.robots import get_robot

torch.manual_seed(0)


class ModelTest(unittest.TestCase):
    def test_glow_cNF_model(self):
        """Smoke test - checks that glow_cNF_model() returns"""
        params = IkflowModelParameters()
        robot = get_robot("panda")
        model = glow_cNF_model(params, robot, dim_cond=7, ndim_tot=9)

    def test_accuracy(self):
        """Download a trained model, check that the expected accuracy is achieved when it is evaluated on a test set"""
        # TODO: Implement this test
        pass


if __name__ == "__main__":
    unittest.main()
