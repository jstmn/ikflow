import unittest

import torch
import numpy as np
from jkinpylib.robots import Panda

from ikflow.utils import set_seed
from ikflow.evaluation_utils import get_solution_errors

set_seed()
np.set_printoptions(suppress=True, precision=8)


class EvaluationUtilsTest(unittest.TestCase):
    def test_get_solution_errors(self):
        """Check that get_solution_errors() matches the expected value"""
        robot = Panda()

        target_pose = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        solutions = torch.zeros((1, 7), dtype=torch.float32, device="cpu")
        realized_pose = robot.forward_kinematics(solutions.numpy())[0]
        realized_pose_gt = np.array([0.088, 0.0, 0.926, 0.0, 0.92387953, 0.38268343, 0.0])
        np.testing.assert_allclose(realized_pose, realized_pose_gt, atol=1e-5)  # sanity check values

        # np.sqrt((1 - 0.088 )**2 +  (1 - 0.0 )**2 + (1 - 0.926 )**2 ) = 1.355440887681938
        l2_error_expected = 1.355440887681938

        # from pyquaternion import Quaternion
        # q1 = Quaternion([1.0, 0.0, 0.0, 0.0]); q2 = Quaternion([0.0, 0.92387953, 0.38268343, 0.0])
        # print(Quaternion.absolute_distance(q1, q2 ))
        angular_error_expected = 1.4142135600925292

        l2_error_returned, angular_error_returned = get_solution_errors(robot, solutions, target_pose)
        self.assertAlmostEqual(l2_error_returned[0].item(), l2_error_expected)
        self.assertAlmostEqual(angular_error_returned[0].item(), angular_error_expected)


if __name__ == "__main__":
    unittest.main()
