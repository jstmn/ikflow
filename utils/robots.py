from abc import abstractmethod
import os
import sys
from typing import List, Tuple, Union

sys.path.append(os.getcwd())

import config
from utils.forward_kinematics import BatchFK, klampt_fk, kinpy_fk, klampt_fk_w_l2_loss_pts
from utils.math_utils import geodesic_distance_between_quaternions

import torch
import numpy as np
import kinpy as kp
import klampt
from klampt import IKObjective, IKSolver
from klampt.model import ik
from klampt.math import so3


def format_joint_limits(lower: List[float], upper: List[float]) -> List[Tuple[float, float]]:
    """Format a list of upper and lower joint limits to a list of tuples"""
    actuated_joints_limits = []
    for l, u in zip(lower, upper):
        if u - l > 2 * np.pi:
            print("format_joint_limits()\ttu:", u, "\tl:", l, "\t-> setting lower to -pi/2, upper to pi/2")
            l = -np.pi
            u = np.pi
        actuated_joints_limits.append((l, u))
    return actuated_joints_limits


# ___________________________________________________________________________________________
# Testing

MAX_ALLOWABLE_L2_ERR = 5e-4
MAX_ALLOWABLE_ANG_ERR = 0.0008726646  # .05 degrees


def assert_endpose_position_almost_equal(endpoints1: np.array, endpoints2: np.array):
    """Check that the position of each pose is nearly the same"""
    l2_errors = np.linalg.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], axis=1)
    for i in range(l2_errors.shape[0]):
        assert l2_errors[i] < MAX_ALLOWABLE_L2_ERR


def assert_endpose_rotation_almost_equal(endpoints1: np.array, endpoints2: np.array, threshold=None):
    """Check that the rotation of each pose is nearly the same"""
    if threshold is None:
        threshold = MAX_ALLOWABLE_ANG_ERR
    rotational_errors = geodesic_distance_between_quaternions(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
    for i in range(rotational_errors.shape[0]):
        assert rotational_errors[i] < threshold, f"Rotation error {rotational_errors[i]} > {threshold}"


def get_fk_poses(robot, samples: np.array, verbose=False, batch=True):

    """Return fk solutions calculated by kinpy, klampt, and batch_fk"""
    kinpy_fk = robot.forward_kinematics_kinpy(samples)
    klampt_fk = robot.forward_kinematics_klampt(samples)

    if not batch:
        return kinpy_fk, klampt_fk

    batch_fk_t, batch_fk_R = robot.forward_kinematics_batch(torch.from_numpy(samples).float())
    assert batch_fk_t.shape[0] == kinpy_fk.shape[0]
    assert batch_fk_R.shape[0] == kinpy_fk.shape[0]

    # TODO(@jeremysm): Get batch_fk_R to quaternion and return (n x 7) array
    return kinpy_fk, klampt_fk, (batch_fk_t.cpu().data.numpy(), batch_fk_R.cpu().data.numpy())


# ___________________________________________________________________________________________


class RobotModel:
    """Abstract robot model class. Stores generic functions"""

    def __init__(
        self,
        robot_name: str,
        actuated_joints_limits: List[Tuple[float, float]],
        dim_x: int,
        dim_y: int,
        verbosity: int = 1,
        end_poses_to_plot_solution_dists_for=[],
    ):

        self.name = robot_name
        self.actuated_joints_limits = actuated_joints_limits
        self.dim_x = dim_x
        self.dim_y = dim_y

        # Stores a list of end poses that ground truth, and generated joint angles will be plotted for
        if len(end_poses_to_plot_solution_dists_for) > 0:
            if not "numpy" in str(type(end_poses_to_plot_solution_dists_for[0])):
                end_poses_to_plot_solution_dists_for = [np.array(p) for p in end_poses_to_plot_solution_dists_for]
        self.end_poses_to_plot_solution_dists_for = end_poses_to_plot_solution_dists_for

        # Input validation
        assert self.dim_x == len(self.actuated_joints_limits)
        assert dim_y in [2, 7]
        assert isinstance(end_poses_to_plot_solution_dists_for, list)

        if verbosity > 0:
            print(f"Initialized RobotModel.{self.name}")

    def sample(self, n: int, limit_increase=0, solver=None):
        """
        Returns N, x vectors of length self.dim_x
        """
        angs = np.random.rand(n, self.dim_x)  # between [0, 1)

        joint_limits = [[l, u] for (l, u) in self.actuated_joints_limits]

        # Print joint limits
        # print(f"Current joint limits")
        # for idx, lu in enumerate(joint_limits):
        #     print(idx, lu)

        # TODO(@jeremysm): Create a standalone function for getting updated joint limits
        # Change the joint limits
        if abs(limit_increase) > 1e-3:

            for i in range(self.dim_x):
                joint_limits[i][0] -= limit_increase
                joint_limits[i][1] += limit_increase

                # If the total limit range is greater than 2*pi, set the bounds to [-pi, pi]
                range_ = joint_limits[i][1] - joint_limits[i][0]
                if range_ > 2 * np.pi:
                    joint_limits[i][0] = -np.pi
                    joint_limits[i][1] = np.pi

        # Sample
        for i in range(self.dim_x):
            range_ = joint_limits[i][1] - joint_limits[i][0]

            # Need to have a positive difference between lower and upper limits
            assert range_ > 0
            angs[:, i] *= range_
            angs[:, i] += joint_limits[i][0]
        return angs

    def joint_limit_delta_valid(self, limit_increase: float, debug=False):
        """Return whether the rotation limits are valid for each joint, in a change of `delta` is
        applied to their lower and upper limits. See `DATASET_TAG` in data.py for further notes on this

        Check that 1. The limit delta is positive, and 2. the limit delta is not more than 2*pi
        """
        # TODO(@jeremysmorgan)
        for idx, (l, u) in enumerate(self.actuated_joints_limits):
            l -= limit_increase
            u += limit_increase
            joint_range = u - l

            if debug:
                print(f"joint_{idx}: {(l, u)}\t range: {joint_range}")
            if joint_range < 0:
                if debug:
                    print(f"joint_range < 0, returning False")
                return False
            if joint_range > 2 * np.pi:
                if debug:
                    print(f"joint_range  > 2*np.pi")
        return True

    def visualize(self):
        """Visualize the robot. Function should be implemented by subclass"""
        raise NotImplementedError()

    @abstractmethod
    def forward_kinematics(self, x: np.array) -> np.array:
        """Forward kinematics"""
        raise NotImplementedError()

    @abstractmethod
    def inverse_kinematics(self, y: np.array, n_solutions: int, seed="random", return_stats=True) -> np.array:
        """Inverse kinematics"""
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"RobotModel<{self.name}>"


# Gripe: Why are links actuated in the `Klampt/bin/RobotPose` tool? This is confusing and misleading
class KlamptRobotModel(RobotModel):
    """Extends a RobotModel class with functionality from provided by Klampt"""

    dim_y = 7

    def __init__(
        self,
        robot_name: str,
        urdf_filepath: str,
        joint_chain: List[str],
        actuated_joints: List[str],
        actuated_joints_limits: List[Tuple[float, float]],
        dim_x: int,
        end_effector_link_name: str,
        verbosity=0,
        end_poses_to_plot_solution_dists_for=[],
    ):
        assert len(end_effector_link_name) > 0, f"End effector link name '{end_effector_link_name}' is empty"

        super(KlamptRobotModel, self).__init__(
            robot_name,
            actuated_joints_limits,
            dim_x,
            KlamptRobotModel.dim_y,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=end_poses_to_plot_solution_dists_for,
        )

        self.urdf_filepath = urdf_filepath
        self.joint_chain = joint_chain
        self.actuated_joints = actuated_joints
        self.end_effector_link_name = end_effector_link_name

        # Additional input validation
        assert len(self.actuated_joints) == len(self.actuated_joints_limits)
        assert len(self.actuated_joints) == dim_x
        assert len(self.actuated_joints) <= len(self.joint_chain)
        for joint in self.actuated_joints:
            assert joint in self.joint_chain

        with open(urdf_filepath) as f:
            self.kinpy_fk_chain = kp.build_chain_from_urdf(f.read())

        self.batch_fk_calc = BatchFK(
            self.urdf_filepath,
            self.end_effector_link_name,
            self.joint_chain,
            self.actuated_joints,
            print_init=False,
            l2_loss_pts=[],
        )

        self.world_model = klampt.WorldModel()

        if not self.world_model.loadRobot(self.urdf_filepath):
            raise ValueError(f"Error loading urdf: {self.urdf_filepath}")

        self._klampt_robot: klampt.RobotModel = self.world_model.robot(0)
        self._klampt_config_dim = len(self._klampt_robot.getConfig())
        self._klampt_ee_link = self._klampt_robot.link(self.end_effector_link_name)

        assert (
            self._klampt_robot.numDrivers() == self.dim_x
        ), f"# of active joints in urdf {self._klampt_robot.numDrivers()} doesn't equal `dim_x`: {self.dim_x}"

        self._active_dofs = self.get_active_dofs()
        self.assert_fk_functions_equal()
        # TODO(@jeremysm): Update batch_fk so that it works for baxter and kuka11dof
        # self.assert_batch_fk_equal()

        # Printout relevent information
        if verbosity > 0:
            print(self)
            print("\ndrivers:")
            for i in range(self.dim_x):
                print(f"  driver_{i}:", self._klampt_robot.driver(i).getName())
            # self._klampt_robot.numLinks(), self._klampt_robot.link(i).getName()
            # self._klampt_robot.getConfig(), self._klampt_robot.getJointType(i))

    @property
    def klampt_robot_model(self):
        return self._klampt_robot

    # Tests
    def assert_fk_functions_equal(self):
        """
        Test that kinpy, klampt, and batch_fk all return the same poses
        """
        n_samples = 10
        samples = self.sample(n_samples)
        kinpy_fk, klampt_fk = get_fk_poses(self, samples, batch=False)
        assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)
        print(f"klampt, kinpy fk functions match for {self.name}")

    # Tests
    def assert_batch_fk_equal(self):
        """
        Test that kinpy, klampt, and batch_fk all return the same poses
        """
        n_samples = 10
        samples = self.sample(n_samples)
        kinpy_fk, klampt_fk, (batch_fk_t, batch_fk_R) = get_fk_poses(self, samples)
        assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_position_almost_equal(kinpy_fk, batch_fk_t)
        print("assert_batch_fk_equal(): klampt, kinpy, batch_fk functions match")

    def assert_l2_loss_pts_equal(self):
        """
        Test that kinpy, klampt, and batch_fk all return the same poses
        """
        raise NotImplementedError("l2 loss points are not ready for use yet")
        n_samples = 10
        samples = self.sample(n_samples)
        fk_klampt = self.forward_kinematics_klampt(samples, with_l2_loss_pts=True)

        # (B x 3*(n+1) )
        samples_torch = torch.from_numpy(samples).float().to(config.device)
        fk_bfk = self.forward_kinematics_batch(samples_torch, with_l2_loss_pts=True)

        for i in range(len(self.l2_loss_pts) + 1):
            root_idx_klampt = 7 * i
            t_klampt = fk_klampt[:, root_idx_klampt + 0 : root_idx_klampt + 3]
            root_idx = 3 * i
            t_bfk = fk_bfk[:, root_idx + 0 : root_idx + 3].cpu().numpy()
            assert_endpose_position_almost_equal(t_klampt, t_bfk)

        print("assert_l2_loss_pts_equal(): klampt, batch_fk functions equal")

    def __str__(self) -> str:
        s = f"KlamptRobotModel()\n"
        s += "  robot_model:  " + self.name + "\n"
        s += "  dim_x:        " + str(self.dim_x) + "\n"
        s += "  ee_link:      " + self.end_effector_link_name + "\n"
        s += "  active_dofs:  " + str(self._active_dofs) + "\n"
        s += "  klampt_q_len: " + str(self._klampt_config_dim) + "\n"
        return s

    # TODO(@jeremysm): Why does this function have replicated
    def get_active_dofs(self) -> List[int]:
        active_dofs = []
        self._klampt_robot.setConfig([0] * self._klampt_config_dim)
        idxs = [1000 * (i + 1) for i in range(self.dim_x)]
        q_temp = self._klampt_robot.configFromDrivers(idxs)
        for idx, v in enumerate(q_temp):
            if v in idxs:
                active_dofs.append(idx)
        assert len(active_dofs) == self.dim_x, f"len(active_dofs): {len(active_dofs)} != self.dim_x: {self.dim_x}"
        return active_dofs

    def visualize_kinpy(self):
        """ """
        th = {}
        # for joint in self.actuated_joints:
        # 	th[joint] = np.pi / 2
        ret = self.kinpy_fk_chain.forward_kinematics(th)
        print(ret)
        viz = kp.Visualizer()
        mesh_file_path = "/".join(self.urdf_filepath.split(".urdf")[0].split("/")[0:-1])
        viz.add_robot(ret, self.kinpy_fk_chain.visuals_map(), axes=True, mesh_file_path=mesh_file_path)
        viz.spin()

    def x_to_qs(self, x: np.ndarray) -> List[List[float]]:
        """Return a list of configurations from an array of joint angles

        Args:
            x: (n x dim_x) array of joint angle settings

        Returns:
            A list of configurations representing the robots state in klampt
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.dim_x

        n = x.shape[0]
        qs = []
        for i in range(n):
            qs.append(self._klampt_robot.configFromDrivers(x[i].tolist()))
        return qs

    def qs_to_x(self, qs: List[List[float]]) -> np.array:
        """Calculate joint angle values from klampt configurations"""
        res = np.zeros((len(qs), self.dim_x))
        for idx, q in enumerate(qs):
            drivers = self._klampt_robot.configToDrivers(q)
            res[idx, :] = drivers
        return res

    def forward_kinematics(self, x: torch.tensor, device=None) -> np.array:
        """ """
        return self.forward_kinematics_klampt(x)

    def forward_kinematics_klampt(self, x: np.array, with_l2_loss_pts=False) -> np.array:
        """Forward kinematics using the klampt library"""
        assert with_l2_loss_pts is False, "l2_loss_pts not ready for use"
        if with_l2_loss_pts:
            assert len(self.l2_loss_pts) > 0
            return klampt_fk_w_l2_loss_pts(self._klampt_robot, self._klampt_ee_link, self.x_to_qs(x), self.l2_loss_pts)

        return klampt_fk(self._klampt_robot, self._klampt_ee_link, self.x_to_qs(x))

    def forward_kinematics_kinpy(self, x: torch.tensor, device=None) -> np.array:
        return kinpy_fk(self.kinpy_fk_chain, self.actuated_joints, self.end_effector_link_name, self.dim_x, x)

    def forward_kinematics_batch(
        self, x: torch.tensor, device=config.device, with_l2_loss_pts=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert with_l2_loss_pts is False, "l2_loss_pts not ready for use"
        if with_l2_loss_pts:
            assert len(self.l2_loss_pts) > 0
            return self.batch_fk_calc.batch_fk_w_l2_loss_pts(x, device=device)
        return self.batch_fk_calc.batch_fk(x, device=device)

    def inverse_kinematics(
        self, pose: np.array, n_solutions: int, seed="random", debug=False, return_stats=False
    ) -> np.array:
        """Run klampts inverse kinematics solver on given `poses`

        Per http://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-IK.html#ik-solver:
        'To use the solver properly, you must understand how the solver uses the RobotModel:
            First, the current configuration of the robot is the seed configuration to the solver.
            Second, the robot’s joint limits are used as the defaults.
            Third, the solved configuration is stored in the RobotModel’s current configuration.'
        """

        assert len(pose.shape) == 1
        assert len(pose) == 7
        if seed is None:
            seed = "random"
        assert "numpy" in str(type(seed)) or seed == "random", f"Invalide seed of type {type(seed)}"
        assert self._active_dofs is not None, f"self._active_dofs ({type(self._active_dofs)}): {self._active_dofs}"

        random_seed = True
        if isinstance(seed, np.ndarray):
            assert (
                seed.shape[0] == n_solutions
            ), f"Seed {seed.shape} is not correctly formatted (expecting [{n_solutions} x {self.dim_x}])"
            assert seed.shape[1] == self.dim_x
            seed_qs = self.x_to_qs(seed)
            random_seed = False

        if debug:
            print(f"inverse_kinematics(): Calculating {n_solutions} solutions for pose: {pose}")

        n_tries_per_pose = 250
        max_iterations = 2500
        R = so3.from_quaternion(pose[3 : 3 + 4])
        obj = ik.objective(self._klampt_ee_link, t=pose[0:3].tolist(), R=R)
        solution_qs = []
        solution_n_iterations = []
        solution_n_retries = []
        pre_solve_configs = []

        n_solutions_iterator = range(n_solutions)
        if debug:
            import tqdm

            n_solutions_iterator = tqdm.tqdm(n_solutions_iterator)

        # Get `sols_per_pose` solutions for each pose
        for sol_i in n_solutions_iterator:
            solution_found = False
            for retry_i in range(n_tries_per_pose):

                solver = IKSolver(self._klampt_robot)
                solver.add(obj)
                solver.setActiveDofs(self._active_dofs)
                solver.setMaxIters(max_iterations)
                solver.setTolerance(1e-4)

                # `sampleInitial()` needs to come after `setActiveDofs()`, otherwise x,y,z,r,p,y of the robot will
                # be randomly set aswell <(*<_*)>
                if random_seed:
                    solver.sampleInitial()
                else:
                    solver.setBiasConfig(seed_qs[sol_i])

                if return_stats:
                    pre_solve_config = self._klampt_robot.getConfig()

                res = solver.solve()
                if not res:
                    # print("IK failed after", solver.lastSolveIters(), "iterations. residual", solver.getResidual())
                    continue

                # print("Solved in ", solver.lastSolveIters(), "iterations")
                # print(solver.lastSolveIters(), end="\t")
                solution_qs.append(self._klampt_robot.getConfig())
                solution_found = True
                if return_stats:
                    solution_n_iterations.append(solver.lastSolveIters())
                    x = np.round(self.qs_to_x([pre_solve_config])[0], 8)
                    pre_solve_configs.append(x.tolist())
                    solution_n_retries.append(retry_i)
                break

            if not solution_found:
                print(f"inverse_kinematics() No solution found for {pose} ({n_tries_per_pose} random starts failed)")

        if return_stats:
            return self.qs_to_x(solution_qs), solution_n_iterations, pre_solve_configs, solution_n_retries

        return self.qs_to_x(solution_qs)


# ----------------------------------------------------------------------------------------------------------------------
#
#                                             Planar robots (RobotModel2D subclasses)
#


class RobotModel2d(RobotModel):
    """RobotModel2d represents planar robots whose workspace is R(2)"""

    dim_y = 2

    def __init__(self, robot_name: str, actuated_joints_limits: List[Tuple[float, float]], dim_x: int, verbosity=1):
        super().__init__(robot_name, actuated_joints_limits, dim_x, RobotModel2d.dim_y, verbosity=verbosity)

    def visualize(self):
        """Visualize the robot"""
        # TODO(@jeremysm): Implement visualization
        raise NotImplementedError()

    @staticmethod
    def jacobian(self, thetas: np.array) -> np.array:
        raise NotImplementedError()

    def inverse_kinematics(
        self,
        y_target: np.array,
        n_solutions: int,
        seed="random",
        alpha=0.1,
        max_n_steps=100,
        valid_err_threshold=0.0005,
        debug=False,
    ) -> np.array:
        """Generate samples for the robot to achieve the desired end pose"""

        if debug:
            print("y_target:", y_target)

        y_target = np.reshape(y_target, (2, 1))
        sols = np.zeros((n_solutions, self.dim_x))

        def err(_y):
            assert y_target.shape[0] == 2
            assert y_target.shape[1] == 1
            assert _y.shape[0] == 2
            assert _y.shape[1] == 1
            return np.sqrt((y_target[0, 0] - _y[0, 0]) ** 2 + (y_target[1, 0] - _y[1, 0]) ** 2)

        def fk(_q):
            assert _q.shape[0] == self.dim_x
            assert _q.shape[1] == 1
            # _q should be (dim_x x 1)
            y = self.forward_kinematics(np.reshape(_q, (1, self.dim_x)))
            return np.reshape(y, (2, 1))

        for sol_i in range(n_solutions):

            # Reshape q to (dim_x x 1)
            q = np.reshape(self.sample(1), (self.dim_x, 1))
            sol_found = False

            for j in range(max_n_steps):

                y = fk(q)

                # (2 x 1)
                err_direction = np.reshape(y_target - y, (2, 1))

                # (2 x dim_x)
                J = self.jacobian(q[:, 0])

                # (dim_x x 2)
                J_pinv = np.linalg.pinv(J)

                #    (dim_x x 2) * (2 x 1) ->  (dim_x x 1)
                delta_q = np.matmul(J_pinv, err_direction)
                q = q + alpha * delta_q

                if debug:
                    print(f"(sol. {sol_i})\tq_{j}:", q[:, 0], "\ty:", fk(q)[:, 0])
                if err(fk(q)) < valid_err_threshold:
                    sols[sol_i, :] = q[:, 0]
                    sol_found = True
                    break

            if not sol_found:
                raise RuntimeError(
                    f"Could not generate {sol_i}'th ik solution for y: {y[:, 0]} (final error: {err(fk(q))})"
                )
        return sols


def segment_points(xy, length, angle):
    uv = np.array(xy)[:, :]
    angle = np.array(angle)
    uv[:, 0] += length * np.cos(angle)
    uv[:, 1] += length * np.sin(angle)
    return xy, uv


class Planar2DofArm(RobotModel2d):
    """Planar robot with two revolute joints"""

    name = "planar_2dof"
    arm_lengths = [1, 1]

    def __init__(self, verbosity=0):
        actuated_joints_limits = [(-np.pi, np.pi), (-np.pi, np.pi)]
        dim_x = 2
        RobotModel2d.__init__(self, Planar2DofArm.name, actuated_joints_limits, dim_x, verbosity=verbosity)

    def forward_kinematics(self, samples: np.array):
        """Forward kinematics"""
        assert len(samples.shape) == 2
        assert samples.shape[1] == 2, f"samples must be (n x 2) (Input shape is {samples.shape})"
        starting_pos = np.zeros(samples.shape)
        x0, x1 = segment_points(starting_pos, Planar2DofArm.arm_lengths[0], samples[:, 0])
        x1, x2 = segment_points(x1, Planar2DofArm.arm_lengths[1], samples[:, 0] + samples[:, 1])
        return x2


class Planar3DofArm(RobotModel2d):
    """Planar robot with two revolute joints"""

    name = "planar_3dof"
    arm_lengths = [1, 1, 1]

    def __init__(self, verbosity=0):
        actuated_joints_limits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        dim_x = 3
        RobotModel2d.__init__(self, Planar3DofArm.name, actuated_joints_limits, dim_x, verbosity=verbosity)

    def forward_kinematics(self, samples: np.array):
        """Forward kinematics"""
        assert len(samples.shape) == 2
        assert samples.shape[1] == 3, f"samples must be (n, 3) {samples.shape}"
        starting_pos = np.zeros((samples.shape[0], 2))
        x0, x1 = segment_points(starting_pos, Planar3DofArm.arm_lengths[0], samples[:, 0])
        x1, x2 = segment_points(x1, Planar3DofArm.arm_lengths[1], samples[:, 0] + samples[:, 1])
        x2, x3 = segment_points(x2, Planar3DofArm.arm_lengths[2], samples[:, 0] + samples[:, 1] + samples[:, 2])
        return x3

    def jacobian(self, thetas: np.array):
        """Get the jacobian of the end effector with respect to the joint angles. `theta` must be a (dim_x,) array"""

        assert len(thetas.shape) == 1

        a, b, c = Planar3DofArm.arm_lengths
        x = thetas[0]
        y = thetas[1]
        z = thetas[2]

        J = np.zeros((2, 3))

        def sin(t):
            return np.sin(t)

        def cos(t):
            return np.cos(t)

        J[0, 0] = -a * sin(x) - b * sin(x + y) - c * sin(x + y + z)
        J[0, 1] = -b * sin(x + y) - c * sin(x + y + z)
        J[0, 2] = -c * sin(x + y + z)
        J[1, 0] = a * cos(x) + b * cos(x + y) + c * cos(x + y + z)
        J[1, 1] = b * cos(x + y) + c * cos(x + y + z)
        J[1, 2] = c * cos(x + y + z)
        return J


class RailYChain3(RobotModel2d):
    """Planar robot with one joint along the y axis, 3 rotational"""

    name = "raily_chain3"
    arm_lengths = [0.5, 0.5, 1.0]
    raily_range = (-1, 1)

    def __init__(self, verbosity=0):
        actuated_joints_limits = [RailYChain3.raily_range, (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        dim_x = 4
        RobotModel2d.__init__(self, RailYChain3.name, actuated_joints_limits, dim_x, verbosity=verbosity)

    def forward_kinematics(self, samples: np.array) -> np.array:
        """Forwared kinematics with the
        Args:
            samples (np.array):
        Returns:
            (np.array): A (n x 2) numpy array containing the end points for each sample
        """
        assert len(samples.shape) == 2
        assert samples.shape[1] == self.dim_x, f"samples must be (n, {self.dim_x}). Currently: {samples.shape}"

        samples[:, 0] = np.clip(samples[:, 0], a_min=RailYChain3.raily_range[0], a_max=RailYChain3.raily_range[1])
        starting_pos = np.stack([np.zeros((samples.shape[0])), samples[:, 0]], axis=1)
        x0, x1 = segment_points(starting_pos, RailYChain3.arm_lengths[0], samples[:, 1])
        x1, x2 = segment_points(x1, RailYChain3.arm_lengths[1], samples[:, 1] + samples[:, 2])
        x2, x3 = segment_points(x2, RailYChain3.arm_lengths[2], samples[:, 1] + samples[:, 2] + samples[:, 3])
        return x3

    def jacobian(self, thetas: np.array):
        assert len(thetas.shape) == 1
        a, b, c = RailYChain3.arm_lengths

        w = thetas[0]
        x = thetas[1]
        y = thetas[2]
        z = thetas[3]

        # (dim_y x dim_x)
        J = np.zeros((2, 4))

        def sin(t):
            return np.sin(t)

        def cos(t):
            return np.cos(t)

        """
            [
              0 | -a sin(x) - b sin(x + y) - c sin(x + y + z) | -b sin(x + y) - c sin(x + y + z) | -c sin(x + y + z)
              1 | a cos(x) + b cos(x + y) + c cos(x + y + z)   | b cos(x + y) + c cos(x + y + z)  | c cos(x + y + z)
            ]
        """
        J[0, 0] = 0
        J[0, 1] = -a * sin(x) - b * sin(x + y) - c * sin(x + y + z)
        J[0, 2] = -b * sin(x + y) - c * sin(x + y + z)
        J[0, 3] = -c * sin(x + y + z)

        J[1, 0] = 1
        J[1, 1] = a * cos(x) + b * cos(x + y) + c * cos(x + y + z)
        J[1, 2] = b * cos(x + y) + c * cos(x + y + z)
        J[1, 3] = c * cos(x + y + z)
        return J


# ----------------------------------------------------------------------------------------------------------------------
#
#                                             3d Robots (RobotModel3D subclasses)
#


class Atlas(KlamptRobotModel):
    name = "atlas"
    formal_robot_name = "ATLAS (2013) - Arm and Waist"

    def __init__(self, verbosity=0):
        lower_lims = [-0.610865, -1.2, -0.790809, -1.9635, -1.39626, 0.0, 0.0, -1.571, -0.436]
        upper_lims = [0.610865, 1.28, 0.790809, 1.9635, 1.74533, 3.14159, 2.35619, 1.571, 1.571]
        actuated_joints_limits = format_joint_limits(lower_lims, upper_lims)
        actuated_joints = [
            "back_lbz",
            "back_mby",
            "back_ubx",
            "l_arm_usy",
            "l_arm_shx",
            "l_arm_ely",
            "l_arm_elx",
            "l_arm_uwy",
            "l_arm_mwx",
        ]
        joint_chain = [
            "back_lbz",
            "back_mby",
            "back_ubx",
            "l_arm_usy",
            "l_arm_shx",
            "l_arm_ely",
            "l_arm_elx",
            "l_arm_uwy",
            "l_arm_mwx",
        ]

        dim_x = len(actuated_joints)
        end_effector_link_name = "l_hand"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/atlas/atlas.urdf"

        poses_for_dist_plots = [
            [
                0.24738925037485776,
                0.7320043670997642,
                0.3187327970133442,
                0.4378322934122965,
                0.018850079547699232,
                -0.43738190189404025,
                0.7852672342851993,
            ],
            [
                0.7321550608251243,
                0.14512674067207174,
                0.6955283073124627,
                0.2574253094963592,
                0.6772971669557831,
                -0.4394416650584948,
                -0.5309348177271779,
            ],
            [
                0.30376904839853475,
                0.24613356287897334,
                0.9093512686587919,
                0.45353587035014126,
                0.357854570776559,
                -0.6790776314454338,
                -0.4528784505256618,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=1,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class AtlasArm(KlamptRobotModel):
    name = "atlas_arm"
    formal_robot_name = "ATLAS (2013) - Arm"

    def __init__(self, verbosity=0):
        lower_lims = [-1.9635, -1.39626, 0.0, 0.0, -1.571, -0.436]
        upper_lims = [1.9635, 1.74533, 3.14159, 2.35619, 1.571, 1.571]
        actuated_joints_limits = format_joint_limits(lower_lims, upper_lims)

        actuated_joints = ["l_arm_usy", "l_arm_shx", "l_arm_ely", "l_arm_elx", "l_arm_uwy", "l_arm_mwx"]

        joint_chain = [
            "back_lbz",
            "back_mby",
            "back_ubx",
            "l_arm_usy",
            "l_arm_shx",
            "l_arm_ely",
            "l_arm_elx",
            "l_arm_uwy",
            "l_arm_mwx",
        ]

        dim_x = len(actuated_joints)
        end_effector_link_name = "l_hand"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/atlas/atlas_arm.urdf"

        poses_for_dist_plots = [
            [
                -0.39683691866509685,
                0.5085140587143718,
                0.7856906894808809,
                0.006238466089075263,
                0.3308201451252269,
                -0.9412886571710855,
                0.06704309808797732,
            ],
            [
                0.18237094544860177,
                0.3352804819638863,
                0.07517425931178714,
                0.3836813853539195,
                0.4684448424740588,
                -0.26440077363068143,
                -0.7506265749331344,
            ],
            [
                -0.009321417705612102,
                0.6752258901181196,
                0.39124795098046355,
                0.8154451596866131,
                0.3409078197812957,
                -0.16377533030476976,
                -0.4381879632523024,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class Baxter(KlamptRobotModel):
    name = "baxter"
    formal_robot_name = "Baxter"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [
            (-1.70167993878, 1.70167993878),
            (-2.147, 1.047),
            (-3.05417993878, 3.05417993878),
            (-0.05, 2.618),
            (-3.059, 3.059),
            (-1.57079632679, 2.094),
            (-3.059, 3.059),
        ]
        actuated_joints = [
            "left_s0",
            "left_s1",
            "left_e0",
            "left_e1",
            "left_w0",
            "left_w1",
            "left_w2",
        ]
        joint_chain = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2", "left_hand"]
        dim_x = len(actuated_joints)
        end_effector_link_name = "left_hand"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/baxter/baxter.urdf"

        poses_for_dist_plots = [
            [
                0.24061811215679385,
                0.3736164370305189,
                1.246183045110637,
                0.6344778949522116,
                -0.33068477238244065,
                -0.30141136154120673,
                -0.6302670650329586,
            ]
        ]
        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )

    def forward_kinematics_batch(self, x: torch.tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("batch_fk isn't working for Baxter. Fix this bug before using this function")


class Kuka11dof(KlamptRobotModel):
    name = "kuka_11dof"

    def __init__(self, verbosity=0):
        actuated_joints = [
            "lwr_arm_0A_joint",
            "lwr_arm_1A_joint",
            "lwr_arm_2A_joint",
            "lwr_arm_3A_joint",
            "lwr_arm_0B_joint",
            "lwr_arm_1B_joint",
            "lwr_arm_2B_joint",
            "lwr_arm_3B_joint",
            "lwr_arm_4_joint",
            "lwr_arm_5_joint",
            "lwr_arm_6_joint",
        ]
        actuated_joints_limits = [
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]
        dim_x = len(actuated_joints)

        joint_chain = [
            "arm_world_joint",
            "lwr_arm_0A_joint",
            "lwr_arm_1A_joint",
            "lwr_arm_2A_joint",
            "lwr_arm_3A_joint",
            "lwr_arm_0B_joint",
            "lwr_arm_1B_joint",
            "lwr_arm_2B_joint",
            "lwr_arm_3B_joint",
            "lwr_arm_4_joint",
            "lwr_arm_5_joint",
            "lwr_arm_6_joint",
        ]

        urdf_filepath = f"{config.URDFS_DIRECTORY}/kuka_11dof/urdf/model.urdf"
        end_effector_link_name = "lwr_arm_7_link"

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
        )


class PandaArm(KlamptRobotModel):
    name = "panda_arm"
    formal_robot_name = "Panda"

    def __init__(self, verbosity=0):
        lower_lims = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        upper_lims = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        actuated_joints_limits = format_joint_limits(lower_lims, upper_lims)
        actuated_joints = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        dim_x = len(actuated_joints)
        joint_chain = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_joint8",
            "panda_hand_joint",
        ]

        urdf_filepath = f"{config.URDFS_DIRECTORY}/panda_arm/panda.urdf"
        end_effector_link_name = "panda_hand"

        end_poses_to_plot_solution_dists_for = [
            np.array([0.25, 0.25, 0.65, 1, 0, 0, 0]),
            np.array([0.25, 0.25, 0.25, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0, 0]),
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=end_poses_to_plot_solution_dists_for,
        )


class Pr2(KlamptRobotModel):
    name = "pr2"
    formal_robot_name = "PR2"

    def __init__(self, verbosity=0):

        actuated_joints_limits = [
            (0, 0.33),
            (-0.7146018366025517, 2.2853981633974483),
            (-0.5236, 1.3963),
            (-0.8, 3.9000000000000004),
            (-2.3213, 0.00),
            (-np.pi, np.pi),  # continuous
            (-2.18, 0.0),
            (-np.pi, np.pi),  # Continuous
        ]
        actuated_joints = [
            "torso_lift_joint",  #
            "l_shoulder_pan_joint",  #
            "l_shoulder_lift_joint",  #
            "l_upper_arm_roll_joint",  #
            "l_elbow_flex_joint",  #
            "l_forearm_roll_joint",  #
            "l_wrist_flex_joint",  #
            "l_wrist_roll_joint",  #
        ]

        joint_chain = [
            "torso_lift_joint",  # Prismatic
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_upper_arm_joint",  # Fixed
            "l_elbow_flex_joint",
            "l_forearm_roll_joint",  # Continuous
            "l_forearm_joint",  # Fixed
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",  # Continuous
            "l_gripper_palm_joint",  # Fixed
        ]
        end_effector_link_name = "l_gripper_palm_link"

        dim_x = len(actuated_joints)
        urdf_filepath = f"{config.URDFS_DIRECTORY}/pr2/pr2.urdf"

        end_poses_to_plot_solution_dists_for = [
            np.array([0.5, 0.65, 0.65, 1, 0, 0, 0]),
            np.array([0.35, 0.65, 1, 0.707, 0.707, 0, 0]),
            np.array([0.4, 0.15, 1.25, -0.518, -0.316, 0.318, -0.729]),
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=end_poses_to_plot_solution_dists_for,
        )

    def forward_kinematics_batch(self, x: torch.tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("batch_fk isn't working for PR2. Fix this bug before using this function")


class Robonaut2(KlamptRobotModel):
    name = "robonaut2"
    formal_robot_name = "Robonaut 2 - Arm and Waist"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [
            (-3.14, 3.14),
            (-2.8270, 2.8270),
            (-1.6758, 0.4292),
            (-4.5028, -0.3318),
            (-2.7930, 0.0),
            (-1.5360, 4.6770),
            (-0.8, 0.8),
            (-0.75, 0.15),
        ]

        actuated_joints = [
            "r2/waist/joint0",
            "r2/left_arm/joint0",
            "r2/left_arm/joint1",
            "r2/left_arm/joint2",
            "r2/left_arm/joint3",
            "r2/left_arm/joint4",
            "r2/left_arm/wrist/pitch",
            "r2/left_arm/wrist/yaw",
        ]
        joint_chain = [
            "r2/fixed/world_ref/robot_world",
            "r2/fixed/robot_world/robot_base",
            "r2/waist/joint0",
            "r2/fixed/waist_center/left_shoulder_jr3",
            "r2/left_arm/joint0",
            "r2/left_arm/joint1",
            "r2/left_arm/joint2",
            "r2/left_arm/joint3",
            "r2/left_arm/joint4",
            "r2/fixed/left_lower_arm/left_forearm_jr3",
            "r2/fixed/left_lower_arm/left_forearm",
            "r2/left_arm/wrist/pitch",
            "r2/left_arm/wrist/yaw",
            "r2/fixed/left_wrist_yaw/left_palm",
        ]
        dim_x = len(actuated_joints)
        end_effector_link_name = "r2/left_palm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/robonaut2/r2b.urdf"

        poses_for_dist_plots = [
            [
                0.8258596586068078,
                -0.5858493570793044,
                0.48044202718072987,
                0.3948485110351959,
                -0.7134771359349987,
                0.37392004504799786,
                -0.44184706601106166,
            ],
            [
                0.5882190371419995,
                -0.3922902806566411,
                0.47798707157289294,
                0.11631237811121191,
                -0.7017551541822395,
                0.02135463592281715,
                0.702534777645477,
            ],
            [
                -0.5615855230598156,
                -0.712403577025488,
                0.6812918854701584,
                0.14317741335521192,
                -0.2575469612031873,
                -0.11016765112612166,
                -0.9492275173662361,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class Robonaut2Arm(KlamptRobotModel):
    name = "robonaut2_arm"
    formal_robot_name = "Robonaut 2 - Arm"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [
            (-2.8270, 2.8270),
            (-1.6758, 0.4292),
            (-4.5028, -0.3318),
            (-2.7930, 0.0),
            (-1.5360, 4.6770),
            (-0.8, 0.8),
            (-0.75, 0.15),
        ]

        actuated_joints = [
            "r2/left_arm/joint0",
            "r2/left_arm/joint1",
            "r2/left_arm/joint2",
            "r2/left_arm/joint3",
            "r2/left_arm/joint4",
            "r2/left_arm/wrist/pitch",
            "r2/left_arm/wrist/yaw",
        ]
        joint_chain = [
            "r2/fixed/world_ref/robot_world",
            "r2/fixed/robot_world/robot_base",
            "r2/waist/joint0",
            "r2/fixed/waist_center/left_shoulder_jr3",
            "r2/left_arm/joint0",
            "r2/left_arm/joint1",
            "r2/left_arm/joint2",
            "r2/left_arm/joint3",
            "r2/left_arm/joint4",
            "r2/fixed/left_lower_arm/left_forearm_jr3",
            "r2/fixed/left_lower_arm/left_forearm",
            "r2/left_arm/wrist/pitch",
            "r2/left_arm/wrist/yaw",
            "r2/fixed/left_wrist_yaw/left_palm",
        ]
        dim_x = len(actuated_joints)
        end_effector_link_name = "r2/left_palm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/robonaut2/r2b_arm.urdf"

        poses_for_dist_plots = [
            [
                0.09341010678865844,
                0.9752588060734974,
                0.3799172310449838,
                0.3314524726959266,
                -0.17922138629792758,
                -0.5675826948042454,
                0.7320306261325108,
            ],
            [
                0.40248353124070535,
                0.8354361736002678,
                0.319904898068602,
                0.462757754814557,
                0.7909779141055692,
                0.3994205122555624,
                0.025931720839493636,
            ],
            [
                0.31175605185661515,
                0.7970444515021772,
                0.8552372513291725,
                0.927120140888319,
                -0.018004211949672704,
                -0.31735394696647323,
                0.19852094362080727,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class Robosimian(KlamptRobotModel):
    name = "robosimian"
    formal_robot_name = "Robosimian"

    def __init__(self, verbosity=0):
        actuated_joints = [
            "limb1_joint1",
            "limb1_joint2",
            "limb1_joint3",
            "limb1_joint4",
            "limb1_joint5",
            "limb1_joint6",
            "limb1_joint7",
        ]
        actuated_joints_limits = [
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]
        dim_x = len(actuated_joints)

        joint_chain = [
            "limb1_joint0",
            "limb1_joint1",
            "limb1_joint2",
            "limb1_joint3",
            "limb1_joint4",
            "limb1_joint5",
            "limb1_joint6",
            "limb1_joint7",
        ]

        urdf_filepath = f"{config.URDFS_DIRECTORY}/robosimian/robosimian.urdf"
        end_effector_link_name = "limb1_link7"

        poses_for_dist_plots = [
            [
                0.16863024470701707,
                0.28201492512129694,
                0.09281403370874561,
                0.42241508566252,
                0.2186476500202038,
                -0.8206667485092372,
                -0.31664615651645767,
            ],
            [
                -0.0003772088443808608,
                0.7611122361839995,
                0.5208955051353593,
                0.6015064248600906,
                0.5585453599305961,
                0.08596543991743738,
                0.5646477175122863,
            ],
            [
                0.22080825444213317,
                -0.14147027202572388,
                -0.162022296710575,
                0.6872598384396488,
                -0.6064581201042858,
                0.36353385856255893,
                -0.16651004986516937,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=1,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class Ur5(KlamptRobotModel):
    name = "ur5"
    formal_robot_name = "Ur5"

    def __init__(self, verbosity=0):
        actuated_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        actuated_joints_limits = [
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]
        dim_x = len(actuated_joints)

        joint_chain = [
            "world_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "ee_fixed_joint",
        ]

        end_poses_to_plot_solution_dists_for = [
            np.array([0.25, 0.25, 0.25, 1, 0, 0, 0]),
            np.array(
                [
                    -0.16066744359911017,
                    -0.46044821355921434,
                    0.7014843570585816,
                    0.2680316364469035,
                    -0.7876796236371675,
                    0.4989661066563734,
                    -0.24238951458053215,
                ]
            ),
            np.array(
                [-0.25, -0.5, 0.75, 0.2680316364469035, -0.7876796236371675, 0.4989661066563734, -0.24238951458053215]
            ),
        ]

        urdf_filepath = f"{config.URDFS_DIRECTORY}/ur5/ur5.urdf"
        end_effector_link_name = "ee_link"

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=end_poses_to_plot_solution_dists_for,
        )


class Valkyrie(KlamptRobotModel):
    name = "valkyrie"
    formal_robot_name = "Valkyrie - Whole Arm and Waist"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [
            (-1.329, 1.181),
            (-0.13, 0.666),
            (-0.23, 0.255),
            (-2.85, 2.0),
            (-1.519, 1.266),
            (-3.1, 2.18),
            (-2.174, 0.12),
            (-2.019, 3.14),
            (-0.62, 0.625),
            (-0.36, 0.49),
        ]

        actuated_joints = [
            "torsoYaw",
            "torsoPitch",
            "torsoRoll",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "leftElbowPitch",
            "leftForearmYaw",
            "leftWristRoll",
            "leftWristPitch",
        ]
        joint_chain = [
            "torsoYaw",
            "torsoPitch",
            "torsoRoll",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "leftElbowPitch",
            "leftForearmYaw",
            "leftWristRoll",
            "leftWristPitch",
        ]

        dim_x = len(actuated_joints)
        end_effector_link_name = "leftPalm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/valkyrie/valkyrie.urdf"

        poses_for_dist_plots = [
            [
                0.1631548595586403,
                0.38019979688006955,
                -0.23180358879210688,
                0.26871356304177924,
                0.23683691683157207,
                -0.8281293424230389,
                0.43116480385241546,
            ],
            [
                -0.05765381540052625,
                0.5675829631081436,
                0.3396543296998731,
                0.1589133467463874,
                -0.6823528573626945,
                0.7068772246948352,
                -0.09729190861811998,
            ],
            [
                -0.04632182559569811,
                0.3565035736432792,
                -0.22339373632121245,
                0.39123120212634566,
                -0.3548683414559493,
                -0.38421302654596334,
                0.7572231883318611,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class ValkyrieArm(KlamptRobotModel):
    name = "valkyrie_arm"
    formal_robot_name = "Valkyrie - Lower Arm"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [(-2.174, 0.12), (-2.019, 3.14), (-0.62, 0.625), (-0.36, 0.49)]

        actuated_joints = ["leftElbowPitch", "leftForearmYaw", "leftWristRoll", "leftWristPitch"]
        joint_chain = [
            "torsoYaw",
            "torsoPitch",
            "torsoRoll",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "leftElbowPitch",
            "leftForearmYaw",
            "leftWristRoll",
            "leftWristPitch",
        ]

        dim_x = len(actuated_joints)
        end_effector_link_name = "leftPalm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/valkyrie/valkyrie_arm.urdf"

        poses_for_dist_plots = [
            [
                0.32122548147020047,
                0.5404984792065127,
                0.3187,
                0.4403797693045965,
                0.4360722745819032,
                0.35160399720867336,
                -0.7016275787589018,
            ],
            [
                0.019204849725272997,
                0.8676484144429102,
                0.3187,
                0.8388044524732614,
                0.06633372049525497,
                0.539327880384251,
                0.03365063857523537,
            ],
            [
                0.15087484871783086,
                0.8441132237792031,
                0.3187,
                0.9169363660677733,
                -0.09211120253236206,
                -0.19087505724537315,
                -0.3380975295266799,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


class ValkyrieArmShoulder(KlamptRobotModel):
    name = "valkyrie_arm_shoulder"
    formal_robot_name = "Valkyrie - Whole Arm"

    def __init__(self, verbosity=0):
        actuated_joints_limits = [
            (-2.85, 2.0),
            (-1.519, 1.266),
            (-3.1, 2.18),
            (-2.174, 0.12),
            (-2.019, 3.14),
            (-0.62, 0.625),
            (-0.36, 0.49),
        ]

        actuated_joints = [
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "leftElbowPitch",
            "leftForearmYaw",
            "leftWristRoll",
            "leftWristPitch",
        ]
        joint_chain = [
            "torsoYaw",
            "torsoPitch",
            "torsoRoll",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "leftElbowPitch",
            "leftForearmYaw",
            "leftWristRoll",
            "leftWristPitch",
        ]

        dim_x = len(actuated_joints)
        end_effector_link_name = "leftPalm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/valkyrie/valkyrie_arm_shoulder.urdf"

        poses_for_dist_plots = [
            [
                -0.3911646773998547,
                0.6385187360027016,
                0.35509862894194566,
                0.6685533899874475,
                0.32359108188303354,
                -0.6183914351692027,
                -0.2567434699684373,
            ],
            [
                0.45580416771298704,
                0.49369849979206487,
                0.2005524552777278,
                0.15616410419091192,
                -0.5542839646289336,
                -0.4203879942898,
                -0.7011818547099464,
            ],
            [
                -0.02387703146157259,
                0.8501011803055892,
                0.17151903477736002,
                0.057805033185784886,
                -0.12103849886869333,
                -0.9899837700809652,
                -0.04404991380561811,
            ],
        ]

        KlamptRobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            dim_x,
            end_effector_link_name,
            verbosity=verbosity,
            end_poses_to_plot_solution_dists_for=poses_for_dist_plots,
        )


ATLAS = Atlas.name
ATLAS_ARM = AtlasArm.name
KUKA_11DOF = Kuka11dof.name
ROBONAUT = Robonaut2.name
ROBONAUT_ARM = Robonaut2Arm.name
ROBOSIMIAN = Robosimian.name
UR5 = Ur5.name
PANDA = PandaArm.name
PR2 = Pr2.name
VALKYRIE = Valkyrie.name
VALKYRIE_ARM = ValkyrieArm.name
VALKYRIE_ARM_SHOULDER = ValkyrieArmShoulder.name

PLANAR_3DOF = Planar3DofArm.name
RAILY_CHAIN3 = RailYChain3.name

ROBOT_NAMES = [
    ATLAS,
    ATLAS_ARM,
    KUKA_11DOF,
    ROBONAUT,
    ROBONAUT_ARM,
    ROBOSIMIAN,
    UR5,
    PANDA,
    PR2,
    VALKYRIE,
    VALKYRIE_ARM,
    VALKYRIE_ARM_SHOULDER,
    PLANAR_3DOF,
    RAILY_CHAIN3,
]


def get_all_3d_robots() -> List[KlamptRobotModel]:
    """
    Return all implemented 3d robots
    """
    verbosity = 1
    return [
        # IK isn't working. Viz looks funky
        # Kuka11dof(verbosity=verbosity),
        # Batch_fk isn't working
        # Baxter(verbosity=verbosity),
        # Pr2(verbosity=verbosity),
        Atlas(verbosity=verbosity),
        AtlasArm(verbosity=verbosity),
        PandaArm(verbosity=verbosity),
        Robonaut2(verbosity=verbosity),  # vis works too
        Robonaut2Arm(verbosity=verbosity),
        Robosimian(verbosity=verbosity),
        Ur5(verbosity=verbosity),  # Vis is messed up
        Valkyrie(verbosity=verbosity),
        ValkyrieArm(verbosity=verbosity),
        ValkyrieArmShoulder(verbosity=verbosity),
    ]


def get_all_2d_robots() -> List[RobotModel2d]:
    """
    Return all implemented 2d robot models
    """
    return [Planar2DofArm(), RailYChain3(), Planar3DofArm()]


def get_robot(name: str, verbosity: int = 0) -> Union[KlamptRobotModel, RobotModel2d]:
    """
    Return a robot given its name.
    """
    if name == RailYChain3.name:
        return RailYChain3(verbosity=verbosity)
    if name == Planar3DofArm.name:
        return Planar3DofArm(verbosity=verbosity)
    if name == Atlas.name:
        return Atlas(verbosity=verbosity)
    if name == AtlasArm.name:
        return AtlasArm(verbosity=verbosity)
    if name == Baxter.name:
        return Baxter(verbosity=verbosity)
    if name == Kuka11dof.name:
        return Kuka11dof(verbosity=verbosity)
    if name == PandaArm.name:
        return PandaArm(verbosity=verbosity)
    if name == Pr2.name:
        return Pr2(verbosity=verbosity)
    if name == Robonaut2.name:
        return Robonaut2(verbosity=verbosity)
    if name == Robonaut2Arm.name:
        return Robonaut2Arm(verbosity=verbosity)
    if name == Robosimian.name:
        return Robosimian(verbosity=verbosity)
    if name == Ur5.name:
        return Ur5(verbosity=verbosity)
    if name == Valkyrie.name:
        return Valkyrie(verbosity=verbosity)
    if name == ValkyrieArm.name:
        return ValkyrieArm(verbosity=verbosity)
    if name == ValkyrieArmShoulder.name:
        return ValkyrieArmShoulder(verbosity=verbosity)
    raise ValueError(f"Unrecognized robot '{name}'.\nAvailable robots: '{ROBOT_NAMES}'")


def robot_name_to_fancy_robot_name(name: str) -> str:

    robot_cls = [
        RailYChain3,
        Planar3DofArm,
        Atlas,
        AtlasArm,
        Baxter,
        Kuka11dof,
        PandaArm,
        Pr2,
        Robonaut2,
        Robonaut2Arm,
        Robosimian,
        Ur5,
        Valkyrie,
        ValkyrieArm,
        ValkyrieArmShoulder,
    ]

    for cls in robot_cls:
        if cls.name == name:
            return cls.formal_robot_name


if __name__ == "__main__":

    for robot in get_all_3d_robots():

        print(f"visualizeing robot: {robot.name}")
        robot.visualize()
        try:
            robot.visualize_kinpy()
        except ValueError as e:
            print("error visualizing with kinpy")
            print(e)
