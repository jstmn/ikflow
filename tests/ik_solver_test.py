import unittest
import os
import sys

sys.path.append(os.getcwd())

import config
from src.ik_solvers import get_split_len

import torch

torch.manual_seed(0)


class IkSolverTest(unittest.TestCase):
    def test_get_split_len(self):
        """Test the output of get_split_len()"""
        for dim_total in range(3, 20):
            split_dim = get_split_len(dim_total)
            self.assertGreaterEqual(split_dim, 0)
            self.assertLess(split_dim, dim_total)


if __name__ == "__main__":
    unittest.main()
