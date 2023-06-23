import unittest

import torch
from jrl.robots import Panda

from ikflow import config
from ikflow.utils import set_seed
from ikflow.model import TINY_MODEL_PARAMS
from ikflow.ikflow_solver import IKFlowSolver

set_seed()


def _assert_different(t1: torch.Tensor, t2: torch.Tensor):
    assert t1.ndim == 2
    assert t2.ndim == 2
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            v_t1 = t1[i, j]
            n_matching = ((t2 - v_t1).abs() < 1e-8).sum().item()
            assert n_matching == 0


class IkflowSolverTest(unittest.TestCase):
    def test_solve_n_poses(self):
        robot = Panda()
        ikflow_solver = IKFlowSolver(TINY_MODEL_PARAMS, robot)

        # Test 1: equal inputs should have equal outputs
        ys = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], device=config.device, dtype=torch.float32)
        latent = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=config.device,
            dtype=torch.float32,
        )
        ikf_sols = ikflow_solver.solve_n_poses(ys, latent=latent, refine_solutions=False)
        torch.testing.assert_close(ikf_sols[0], ikf_sols[1])

        # Test 2: different inputs should have equal outputs
        ys = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], device=config.device, dtype=torch.float32)
        latent = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=config.device,
            dtype=torch.float32,
        )
        ikf_sols = ikflow_solver.solve_n_poses(ys, latent=latent, refine_solutions=False)
        _assert_different(ikf_sols[0][None, :], ikf_sols[1][None, :])


if __name__ == "__main__":
    unittest.main()
