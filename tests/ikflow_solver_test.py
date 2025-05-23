import unittest
from time import time

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
    # uv run python tests/ikflow_solver_test.py IkflowSolverTest.test___calculate_pose_error
    def test___calculate_pose_error(self):
        model_name = "panda__full__lp191_5.25m"
        n_solutions = 1000

        ikflow_solver, _ = get_ik_solver(model_name)
        # ikflow_solver._calculate_pose_error(qs, target_poses)

        print("Number of solutions, CPU, CUDA")
        for n_solutions in [1, 10, 100, 1000, 5000]:
            qs, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(n_solutions)
            qs_cpu = torch.tensor(qs, device="cpu", dtype=torch.float32)
            qs_cu = torch.tensor(qs, device="cuda:0", dtype=torch.float32)
            target_poses_cpu = torch.tensor(target_poses, device="cpu", dtype=torch.float32)
            target_poses_cu = torch.tensor(target_poses, device="cuda:0", dtype=torch.float32)

            t0 = time()
            ikflow_solver._calculate_pose_error(qs_cpu, target_poses_cpu)
            t_cpu = time() - t0

            t0 = time()
            ikflow_solver._calculate_pose_error(qs_cu, target_poses_cu)
            t_cuda = time() - t0

            print(n_solutions, "\t", round(1000 * t_cpu, 6), "\t", round(1000 * t_cuda, 6))

    # pytest tests/ikflow_solver_test.py -k 'test_generate_exact_ik_solutions'
    # uv run python tests/ikflow_solver_test.py IkflowSolverTest.test_generate_exact_ik_solutions
    def test_generate_exact_ik_solutions(self):
        model_name = "panda__full__lp191_5.25m"
        POS_ERROR_THRESHOLD = 0.001
        ROT_ERROR_THRESHOLD = 0.01
        # device = "cpu"
        device = DEVICE
        n_solutions = 1000
        repeat_counts = (1, 3, 10)

        ikflow_solver, _ = get_ik_solver(model_name)
        robot = ikflow_solver.robot

        set_seed(3)
        torch.set_printoptions(linewidth=300, precision=7, sci_mode=False)
        _, target_poses = ikflow_solver.robot.sample_joint_angles_and_poses(
            n_solutions, only_non_self_colliding=True, tqdm_enabled=False
        )
        target_poses = torch.tensor(target_poses, device=device, dtype=torch.float32)
        solutions, valid_solutions = ikflow_solver.generate_exact_ik_solutions(
            target_poses,
            pos_error_threshold=POS_ERROR_THRESHOLD,
            rot_error_threshold=ROT_ERROR_THRESHOLD,
            repeat_counts=repeat_counts,
        )

        # evaluate solutions
        pose_realized = robot.forward_kinematics(solutions)
        torch.testing.assert_close(pose_realized[:, 0:3], target_poses[:, 0:3], atol=POS_ERROR_THRESHOLD, rtol=1e-2)
        rot_errors = geodesic_distance_between_quaternions(target_poses[:, 3:], pose_realized[:, 3:])
        self.assertLess(rot_errors.max().item(), ROT_ERROR_THRESHOLD)
        torch.testing.assert_close(solutions, robot.clamp_to_joint_limits(solutions.clone()))
        assert valid_solutions.sum().item() == n_solutions

    def test_solve_multiple_poses(self):
        robot = Panda()
        ikflow_solver = IKFlowSolver(TINY_MODEL_PARAMS, robot)

        # Test 1: equal inputs should have equal outputs
        ys = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], device=DEVICE, dtype=torch.float32)
        latent = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=DEVICE,
            dtype=torch.float32,
        )
        ikf_sols = ikflow_solver.generate_ik_solutions(ys, None, latent=latent, refine_solutions=False)
        torch.testing.assert_close(ikf_sols[0], ikf_sols[1])

        # Test 2: different inputs should have equal outputs
        ys = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], device=DEVICE, dtype=torch.float32)
        latent = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=DEVICE,
            dtype=torch.float32,
        )
        ikf_sols = ikflow_solver.generate_ik_solutions(ys, None, latent=latent, refine_solutions=False)
        _assert_different(ikf_sols[0][None, :], ikf_sols[1][None, :])


if __name__ == "__main__":
    unittest.main()
