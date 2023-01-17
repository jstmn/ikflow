import unittest

from jkinpylib.conversions import xyzw_to_wxyz, wxyz_to_xyzw, rotation_matrix_from_quaternion, geodesic_distance
from ikflow.utils import set_seed

import torch
import numpy as np

set_seed()

# torch.set_printoptions(precision=)


class ModelTest(unittest.TestCase):
    def test_quaternion_formatting(self):
        """Check that xyzw_to_wxyz and wxyz_to_xyzw are inverses"""
        q = np.random.rand(4)
        q_reformatted = wxyz_to_xyzw(xyzw_to_wxyz(q))
        self.assertTrue(np.allclose(q, q_reformatted))

        q = np.random.rand(4)
        q_reformatted = xyzw_to_wxyz(wxyz_to_xyzw(q))
        self.assertTrue(np.allclose(q, q_reformatted))

    def test_geodesic_distance(self):
        """Test geodesic_distance()"""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True, device="cpu")
        q2 = torch.tensor(
            [[1.0, -0.000209831749089, -0.000002384310619, 0.000092415713879]], device="cpu", requires_grad=True
        )
        m1 = rotation_matrix_from_quaternion(q1)
        m2 = rotation_matrix_from_quaternion(q2)

        # Test #1: Returns 0 for closeby quaternions
        distance = geodesic_distance(m1, m2)

        self.assertAlmostEqual(distance[0].item(), 0.0, delta=5e-4)

        # Test #2: Passes a gradient when for closeby quaternions
        loss = distance.mean()
        loss.backward()
        self.assertFalse(torch.isnan(q1.grad).any())
        self.assertFalse(torch.isnan(q2.grad).any())


if __name__ == "__main__":
    unittest.main()
