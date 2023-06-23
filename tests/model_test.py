import unittest
from typing import List, Tuple

import torch
from jrl.robots import Panda
import FrEIA.framework as Ff

from ikflow.utils import set_seed
from ikflow.model import glow_cNF_model, IkflowModelParameters, get_pre_sigmoid_scaling_node
from ikflow.ikflow_solver import IKFlowSolver

set_seed()

_PANDA = Panda()


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        # panda_joint_lims: (-> midpoint) (-> range)
        # (-2.8973, 2.8973)  0.0     5.7946
        # (-1.7628, 1.7628)  0.0     3.5256
        # (-2.8973, 2.8973)  0.0     5.7946
        # (-3.0718, -0.0698) -1.5708 3.0020000000000002
        # (-2.8973, 2.8973)  0.0     5.7946
        # (-0.0175, 3.7525)  1.8675  3.77
        # (-2.8973, 2.8973)  0.0     5.7946

        self.panda_theta_upper = torch.tensor(
            [[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]], device="cuda"
        )

        # sanity check:
        self.assertEqual(self.panda_theta_upper.shape, (1, 7))
        for i, (_, upper) in enumerate(_PANDA.actuated_joints_limits):
            self.assertAlmostEqual(self.panda_theta_upper[0, i].item(), upper, places=5)

        self.panda_theta_lower = torch.tensor(
            [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]], device="cuda"
        )
        # sanity check self.panda_theta_lower:
        self.assertEqual(self.panda_theta_lower.shape, (1, 7))
        for i, (lower, _) in enumerate(_PANDA.actuated_joints_limits):
            self.assertAlmostEqual(self.panda_theta_lower[0, i].item(), lower, places=5)

        self.panda_mid = torch.tensor([[0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]], device="cuda")

    # ==================================================================================================================
    # Tests
    #

    def test_IkFlowFixedLinearTransform_forward(self):
        """Test that get_pre_sigmoid_scaling_node is returning a correctly configured IkFlowFixedLinearTransform node
        during the forward pass
        """
        ndim_tot = 7
        input_node = Ff.InputNode(ndim_tot, name="input")
        nodes = [input_node]
        _, scaling_node = get_pre_sigmoid_scaling_node(ndim_tot, _PANDA, nodes)

        # Test 1: theta is the upper limit joint limit
        expected = 1.0 * torch.ones((1, 7), device="cuda")
        output, _ = scaling_node.forward([self.panda_theta_upper - 1e-8], rev=False)
        output = output[0]
        torch.testing.assert_close(output, expected)

        # Test 2: theta is the lower joint limit
        expected = torch.zeros((1, 7), device="cuda")
        output, _ = scaling_node.forward([self.panda_theta_lower + 1e-8], rev=False)
        output = output[0]
        torch.testing.assert_close(output, expected)

        # Test 3: theta is halfway between joint limits, output should be ~0.5
        theta = self.panda_mid
        expected = 0.5 * torch.ones((1, 7), device="cuda")
        output, _ = scaling_node.forward([theta], rev=False)
        output = output[0]
        torch.testing.assert_close(output, expected)

    def test_IkFlowFixedLinearTransform_reverse(self):
        """Test that get_pre_sigmoid_scaling_node is returning a correctly configured IkFlowFixedLinearTransform node
        during the reverse pass
        """
        ndim_tot = 7
        input_node = Ff.InputNode(ndim_tot, name="input")
        nodes = [input_node]
        _, scaling_node = get_pre_sigmoid_scaling_node(ndim_tot, _PANDA, nodes)

        # Test 1: theta is the upper limit joint limit
        input = 1.0 * torch.ones((1, 7), device="cuda")
        expected = self.panda_theta_upper
        output, _ = scaling_node.forward([input], rev=True)
        output = output[0]
        torch.testing.assert_close(output, expected)

        # Test 2: theta is the lower joint limit
        input = torch.zeros((1, 7), device="cuda")
        expected = self.panda_theta_lower
        output, _ = scaling_node.forward([input], rev=True)
        output = output[0]
        torch.testing.assert_close(output, expected)

        # Test 3: theta is halfway between joint limits, output should be ~0.5
        input = 0.5 * torch.ones((1, 7), device="cuda")
        expected = self.panda_mid
        output, _ = scaling_node.forward([input], rev=True)
        output = output[0]
        torch.testing.assert_close(output, expected)

    def test_sigmoid_on_output(self):
        """Test that a working model is returned with sigmoid_on_output"""
        params = IkflowModelParameters()
        params.sigmoid_on_output = True
        model = glow_cNF_model(params, _PANDA, dim_cond=8, ndim_tot=7)
        conditional = torch.randn((5, 8), device="cuda")

        # Test 1: Reverse pass, gaussian noise
        latent = torch.randn((5, 7), device="cuda")
        output_rev, _ = model(latent, c=conditional, rev=True)
        assert_joint_angle_tensor_in_joint_limits(output_rev, _PANDA.actuated_joints_limits, eps=1e-5)

        # Test 2: Reverse pass, really large gaussian noise
        latent = 1e8 * torch.randn((5, 7), device="cuda")
        output_rev, _ = model(latent, c=conditional, rev=True)
        assert_joint_angle_tensor_in_joint_limits(output_rev, _PANDA.actuated_joints_limits, eps=1e-5)

    def test_glow_cNF_model(self):
        """Smoke test - checks that glow_cNF_model() returns"""

        # Setup model
        params = IkflowModelParameters()
        params.softflow_enabled = False
        params.dim_latent_space = 9
        model = glow_cNF_model(params, _PANDA, dim_cond=7, ndim_tot=9)

        # Run model
        conditional = torch.randn((5, 7), device="cuda")
        latent = torch.randn((5, 9), device="cuda")
        output_rev, _ = model(latent, c=conditional, rev=True)
        loss = torch.sum(output_rev)
        loss.backward()

    def test_IKFlowSolver_smoketest(self):
        """Smoke test - checks that IKFlowSolver is happy"""

        # Test 1: works with softflow disabled
        params = IkflowModelParameters()
        params.softflow_enabled = False
        solver = IKFlowSolver(params, _PANDA)

        conditional = torch.randn((5, 7), device="cuda")
        latent = torch.randn((5, 9), device="cuda")
        output_rev, _ = solver.nn_model(latent, c=conditional, rev=True)
        loss = torch.sum(output_rev)
        loss.backward()

        # Test 1: works with softflow
        params = IkflowModelParameters()
        params.softflow_enabled = True
        solver = IKFlowSolver(params, _PANDA)

        conditional = torch.randn((5, 8), device="cuda")
        latent = torch.randn((5, 9), device="cuda")
        output_rev, _ = solver.nn_model(latent, c=conditional, rev=True)
        loss = torch.sum(output_rev)
        loss.backward()

    def test_glow_cNF_model_w_softflow(self):
        """Smoke test - checks that glow_cNF_model() returns"""
        # Setup model
        params = IkflowModelParameters()
        params.softflow_enabled = True
        params.dim_latent_space = 9
        model = glow_cNF_model(params, _PANDA, dim_cond=8, ndim_tot=9)

        # Run model
        conditional = torch.randn((5, 8), device="cuda")
        latent = torch.randn((5, 9), device="cuda")
        output_rev, _ = model(latent, c=conditional, rev=True)
        loss = torch.sum(output_rev)
        loss.backward()

    # TODO: Implement this test
    # def test_accuracy(self):
    #     """Download a trained model, check that the measured accuracy matches the expected accuracy"""


if __name__ == "__main__":
    unittest.main()
