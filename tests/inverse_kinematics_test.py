from typing import Tuple, Callable

import unittest
import os
import sys

sys.path.append(os.getcwd())

from src.robots import RobotModel, PandaArm
from src.math_utils import geodesic_distance_between_quaternions
from src.utils import set_seed

from tqdm import tqdm
import torch
import numpy as np

# Set seed to ensure reproducibility
set_seed()


MAX_ALLOWABLE_L2_ERR = 2.5e-4
MAX_ALLOWABLE_ANG_ERR = 0.01  # degrees


class TestInverseKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.robots = [PandaArm()]

    # Helper functions
    def assert_pose_position_almost_equal(
        self, endpoints1: np.array, endpoints2: np.array, max_allowable_l2_err=MAX_ALLOWABLE_L2_ERR
    ):
        """Check that the position of each pose is nearly the same"""
        l2_errors = np.linalg.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], axis=1)
        self.assertLess(np.max(l2_errors), max_allowable_l2_err)
        for i in range(l2_errors.shape[0]):
            self.assertLess(l2_errors[i], max_allowable_l2_err)

    def assert_pose_rotation_almost_equal(
        self, endpoints1: np.array, endpoints2: np.array, max_allowable_rotational_err=MAX_ALLOWABLE_ANG_ERR
    ):
        """Check that the rotation of each pose is nearly the same"""
        rotational_errors = geodesic_distance_between_quaternions(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
        for i in range(rotational_errors.shape[0]):
            self.assertLess(rotational_errors[i], max_allowable_rotational_err)

    def get_fk_poses(
        self, robot: RobotModel, samples: np.array
    ) -> Tuple[np.array, np.array, Tuple[np.array, np.array]]:
        """Return fk solutions calculated by kinpy, klampt, and batch_fk"""
        kinpy_fk = robot.forward_kinematics_kinpy(samples)
        klampt_fk = robot.forward_kinematics_klampt(samples)
        batch_fk_t, batch_fk_R = robot.forward_kinematics_batch(torch.from_numpy(samples).float())

        assert batch_fk_t.shape[0] == kinpy_fk.shape[0]
        assert batch_fk_R.shape[0] == kinpy_fk.shape[0]
        # TODO(@jeremysm): Get batch_fk_R to quaternion and return (n x 7) array
        return kinpy_fk, klampt_fk, (batch_fk_t.cpu().data.numpy(), batch_fk_R.cpu().data.numpy())

    # --- Tests
    def ik_with_random_seed_test(self, robot: RobotModel, ik_func: Callable, positional_tol: float):
        """
        Test that fk(ik(fk(sample))) = fk(sample) using a random seed in ik(...)
        """
        n_samples = 100
        samples = robot.sample(n_samples)
        poses_gt = robot.forward_kinematics_klampt(samples)

        for i, pose_gt in enumerate(poses_gt):
            samples_ik = ik_func(pose_gt, seed=None, positional_tolerance=positional_tol, verbosity=2)
            self.assertEqual(samples_ik.shape, (1, robot.ndofs))
            poses_ik = robot.forward_kinematics_klampt(samples_ik)
            self.assertEqual(poses_ik.shape, (1, 7))
            pose_gt = pose_gt.reshape(1, 7)
            self.assert_pose_position_almost_equal(pose_gt, poses_ik, max_allowable_l2_err=2.0 * positional_tol)
            self.assert_pose_rotation_almost_equal(pose_gt, poses_ik)

    def ik_with_seed_test(self, robot: RobotModel, ik_func: Callable, positional_tol: float):
        """
        Test that fk(ik(fk(sample))) = fk(sample) using a seed in ik(...)

        This test is to ensure that the seed is working properly. Example code to print print out the realized end pose
        error of the perturbed joint angles:
            samples_noisy_poses = robot.forward_kinematics_klampt(samples_noisy)
            samples_noisy_l2_errors = np.linalg.norm(samples_noisy_poses[:, 0:3] - poses_gt[:, 0:3], axis=1)
            print("Perturbed configs L2 end pose error:", samples_noisy_l2_errors)
        """
        n_samples = 100
        n_successes = 0

        samples = robot.sample(n_samples)
        poses_gt = robot.forward_kinematics_klampt(samples)
        samples_noisy = samples + np.random.normal(0, 0.1, samples.shape)

        for sample_noisy, pose_gt in zip(samples_noisy, poses_gt):
            samples_ik = ik_func(pose_gt, seed=sample_noisy, positional_tolerance=positional_tol, verbosity=2)
            # try:
            #     samples_ik = ik_func(pose_gt, seed=sample_noisy, positional_tolerance=positional_tol, verbosity=2)
            # except Exception as e:
            #     continue
            self.assertEqual(samples_ik.shape, (1, robot.ndofs))
            poses_ik = robot.forward_kinematics_klampt(samples_ik)
            self.assertEqual(poses_ik.shape, (1, 7))
            pose_gt = pose_gt.reshape(1, 7)
            self.assert_pose_position_almost_equal(pose_gt, poses_ik, max_allowable_l2_err=2.0 * positional_tol)
            self.assert_pose_rotation_almost_equal(pose_gt, poses_ik)
            n_successes += 1

        print(f"Success rate: {round(100*(n_successes / n_samples), 2)}% ({n_successes}/{n_samples})")

    def test_klampt_ik_w_seed(self):
        positional_tol = 1e-3
        for robot in self.robots:
            self.ik_with_seed_test(robot, robot.inverse_kinematics_klampt, positional_tol=positional_tol)

    def test_klampt_ik_w_random_seed(self):
        positional_tol = 1e-3
        for robot in self.robots:
            self.ik_with_random_seed_test(robot, robot.inverse_kinematics_klampt, positional_tol=positional_tol)

    # def test_tracik_ik_w_seed(self):
    #     for robot in self.robots:
    #         self.ik_with_seed_test(robot, robot.inverse_kinematics_tracik)

    # def test_tracik_ik_w_random_seed(self):
    #     for robot in self.robots:
    #         self.ik_with_random_seed_test(robot, robot.inverse_kinematics_tracik)

    # def test_ik_seeding(self):
    #     wandb_run_name = tpms.TPMS["atlas_tpm"]["wandb_run_name"]
    #     model_wrapper, hps = get_model_wrapper(wandb_run_name)

    #     def randomize_klampt_robot_q():
    #         # model_wrapper.robot_model._klampt_robot.randomizeConfig()
    #         model_wrapper.robot_model._klampt_robot.setConfig(
    #             model_wrapper.robot_model.x_to_qs(model_wrapper.robot_model.sample(1))[0]
    #         )

    #     # Randomly drawn poses are held fixed for a given batch_size
    #     batch_size = 5
    #     r_samples = model_wrapper.robot_model.sample(batch_size)
    #     end_poses = model_wrapper.robot_model.forward_kinematics(r_samples)

    #     with torch.inference_mode():
    #         seeds, seed_runtime = model_wrapper.make_samples_mult_y(end_poses, latent_noise_scale=0.75)

    #     seeds = seeds.cpu().numpy()
    #     seeds += 0.001 * np.random.randn(seeds.shape[0], seeds.shape[1])
    #     seeds_fk = model_wrapper.robot_model.forward_kinematics(seeds)

    #     for i in range(batch_size):

    #         print(f"\n_________\n{i}")

    #         pose = end_poses[i]
    #         seed = np.expand_dims(seeds[i, :], axis=0)

    #         print("\npose:   ", pose)

    #         # Set seed, if seeding is enabled
    #         print("seed fk:      ", seeds_fk[i])
    #         print("seed:         ", seed)
    #         seed_l2_error = np.linalg.norm(seeds_fk[i, 0:3] - pose[0:3])
    #         print("seed_l2_error:", 1000 * seed_l2_error, "mm")
    #         print()
    #         (
    #             sols,
    #             solution_n_iterations,
    #             pre_solve_configs,
    #             solution_n_retries,
    #         ) = model_wrapper.robot_model.inverse_kinematics(pose, 1, seed=seed, return_stats=True)
    #         sols_fk = model_wrapper.robot_model.forward_kinematics(sols)
    #         l2_error = np.linalg.norm(sols_fk[0, 0:3] - pose[0:3])
    #         print("sols:                 ", sols)
    #         print("solution_n_iterations:", solution_n_iterations)
    #         print("pre_solve_configs:    ", pre_solve_configs)
    #         print("solution_n_retries:   ", solution_n_retries)
    #         print("l2_error:             ", 1000 * l2_error, "mm")

    #         # randomize_klampt_robot_q()

    #         # print("\n Random seed:")
    #         # sols, solution_n_iterations, pre_solve_configs, solution_n_retries = \
    #         #     model_wrapper.robot_model.inverse_kinematics(pose, 1, return_stats=True)
    #         # print("sols:                 ", sols)
    #         # print("solution_n_iterations:", solution_n_iterations)
    #         # print("pre_solve_configs:    ", pre_solve_configs)
    #         # print("solution_n_retries:   ", solution_n_retries)
    #         # l2_error = np.linalg.norm(sols_fk[0, 0:3] - pose[0:3])
    #         # print("l2_error:", 1000* l2_error, "mm")


if __name__ == "__main__":
    unittest.main()
