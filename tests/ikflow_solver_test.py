import unittest

import torch
from jrl.robots import Panda
from jrl.math_utils import geodesic_distance_between_quaternions

from ikflow import config
from ikflow.model_loading import get_ik_solver
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

    # pytest tests/ikflow_solver_test.py -k 'test_generate_exact_ik_solutions'
    # python tests/ikflow_solver_test.py IkflowSolverTest.test_generate_exact_ik_solutions
    def test_generate_exact_ik_solutions(self):
        model_name = "panda__full__lp191_5.25m"
        POS_ERROR_THRESHOLD = 0.001
        ROT_ERROR_THRESHOLD = 0.01
        # device = "cpu"
        device = config.device
        n_solutions = 5000

        ikflow_solver, _ = get_ik_solver(model_name)
        robot = ikflow_solver.robot

        set_seed()
        torch.set_printoptions(linewidth=300, precision=7, sci_mode=False)
        _, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(
            n_solutions, only_non_self_colliding=True, tqdm_enabled=False
        )
        target_poses = torch.tensor(target_poses, device=device, dtype=torch.float32)
        solutions = ikflow_solver.generate_exact_ik_solutions(
            target_poses, pos_error_threshold=POS_ERROR_THRESHOLD, rot_error_threshold=ROT_ERROR_THRESHOLD
        ).clone()

        # evaluate solutions
        pose_realized = robot.forward_kinematics_batch(solutions)
        torch.testing.assert_close(pose_realized[:, 0:3], target_poses[:, 0:3], atol=POS_ERROR_THRESHOLD, rtol=1e-2)
        rot_errors = geodesic_distance_between_quaternions(target_poses[:, 3:], pose_realized[:, 3:])
        self.assertLess(rot_errors.max().item(), ROT_ERROR_THRESHOLD)
        torch.testing.assert_close(solutions, robot.clamp_to_joint_limits(solutions))

    def test_solve_multiple_poses(self):

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
        ikf_sols = ikflow_solver.generate_ik_solutions(ys, None, latent=latent, refine_solutions=False)
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
        ikf_sols = ikflow_solver.generate_ik_solutions(ys, None, latent=latent, refine_solutions=False)
        _assert_different(ikf_sols[0][None, :], ikf_sols[1][None, :])


if __name__ == "__main__":
    unittest.main()
