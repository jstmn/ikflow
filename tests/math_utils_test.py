import unittest


from ikflow.math_utils import xyzw_to_wxyz, wxyz_to_xyzw
from ikflow.utils import set_seed

import numpy as np

set_seed()


class ModelTest(unittest.TestCase):
    def test_quaternion_formatting(self):
        """Check that xyzw_to_wxyz and wxyz_to_xyzw are inverses"""
        q = np.random.rand(4)
        q_reformatted = wxyz_to_xyzw(xyzw_to_wxyz(q))
        self.assertTrue(np.allclose(q, q_reformatted))

        q = np.random.rand(4)
        q_reformatted = xyzw_to_wxyz(wxyz_to_xyzw(q))
        self.assertTrue(np.allclose(q, q_reformatted))


if __name__ == "__main__":
    unittest.main()
