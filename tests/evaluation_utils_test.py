import unittest

import torch
import numpy as np
from jrl.robots import Panda
from jrl.utils import set_seed

from ikflow.evaluation_utils import solution_pose_errors, calculate_joint_limits_exceeded

set_seed()


class EvaluationUtilsTest(unittest.TestCase):
    def test_solution_pose_errors(self):
        """Check that solution_pose_errors() matches the expected value"""
        robot = Panda()

        target_pose = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        solutions = torch.zeros((1, 7), dtype=torch.float32, device="cpu")
        realized_pose = robot.forward_kinematics(solutions.numpy())[0]
        realized_pose_gt = np.array([0.088, 0.0, 0.926, 0.0, 0.92387953, 0.38268343, 0.0])
        np.testing.assert_allclose(realized_pose, realized_pose_gt, atol=1e-5)  # sanity check values

        # np.sqrt((1 - 0.088 )**2 +  (1 - 0.0 )**2 + (1 - 0.926 )**2 ) = 1.355440887681938
        l2_error_expected = 1.355440887681938
        angular_error_expected = 3.1415927  # From https://www.andre-gaschler.com/rotationconverter/

        l2_error_returned, angular_error_returned = solution_pose_errors(robot, solutions, target_pose)
        self.assertAlmostEqual(l2_error_returned[0].item(), l2_error_expected)
        self.assertAlmostEqual(angular_error_returned[0].item(), angular_error_expected, delta=5e-4)

    def test_calculate_joint_limits_exceeded(self):
        """Test that calculate_joint_limits_exceeded() is working as expected"""

        configs = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [-2, 0, 0],
                [0, -1.999, 0],
                [0, 2.0001, 0],
            ]
        )
        joint_limits = [
            (-1, 1),
            (-2, 2),
            (-3, 3),
        ]
        expected = torch.tensor([False, False, True, False, True], dtype=torch.bool)
        returned = calculate_joint_limits_exceeded(configs, joint_limits)
        self.assertEqual(returned.dtype, torch.bool)
        self.assertEqual(returned.shape, (5,))
        torch.testing.assert_close(returned, expected)


if __name__ == "__main__":
    unittest.main()
