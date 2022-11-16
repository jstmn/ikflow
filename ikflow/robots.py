import os
import sys
from typing import List, Tuple, Union, Optional

sys.path.append(os.getcwd())

import config
from ikflow.forward_kinematics import BatchFK, klampt_fk, kinpy_fk
from ikflow.math_utils import geodesic_distance_between_quaternions, xyzw_to_wxyz, wxyz_to_xyzw

import torch
import numpy as np
import kinpy as kp
import klampt
from klampt import IKSolver
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
    def __init__(
        self,
        robot_name: str,
        urdf_filepath: str,
        joint_chain: List[str],
        actuated_joints: List[str],
        actuated_joints_limits: List[Tuple[float, float]],
        ndofs: int,
        end_effector_link_name: str,
    ):
        assert len(end_effector_link_name) > 0, f"End effector link name '{end_effector_link_name}' is empty"
        assert len(actuated_joints) == len(actuated_joints_limits)
        assert len(actuated_joints) == ndofs
        assert len(actuated_joints) <= len(joint_chain)
        for joint in actuated_joints:
            assert joint in joint_chain

        self._robot_name = robot_name
        self._ndofs = ndofs
        self._end_effector_link_name = end_effector_link_name
        self._actuated_joints = actuated_joints
        self._actuated_joints_limits = actuated_joints_limits

        # Initialize klampt
        # Note: Need to save `_klampt_world_model` as a member variable otherwise you'll be doomed to get a segfault
        self._klampt_world_model = klampt.WorldModel()
        assert self._klampt_world_model.loadRobot(urdf_filepath), f"Error loading urdf '{urdf_filepath}'"
        self._klampt_robot: klampt.robotsim.RobotModel = self._klampt_world_model.robot(0)
        self._klampt_ee_link: klampt.robotsim.RobotModelLink = self._klampt_robot.link(self._end_effector_link_name)
        self._klampt_config_dim = len(self._klampt_robot.getConfig())
        self._klampt_active_dofs = self._get_klampt_active_dofs()

        assert (
            self._klampt_robot.numDrivers() == ndofs
        ), f"# of active joints in urdf {self._klampt_robot.numDrivers()} doesn't equal `ndofs`: {self._ndofs}"

        # Initialize Kinpy.
        # Note: we use both klampt and kinpy to gain confidence that our FK function is correct. This
        # is a temporary measure until we are sure they are correct.
        with open(urdf_filepath) as f:
            self._kinpy_fk_chain = kp.build_chain_from_urdf(f.read().encode("utf-8"))

        # Initialize the batched forward kinematics module
        self.batch_fk_calc = BatchFK(
            urdf_filepath, self._end_effector_link_name, joint_chain, self._actuated_joints, print_init=False
        )

        # Checks
        self.assert_forward_kinematics_functions_equal()
        self.assert_batch_fk_equal()

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def end_effector_link_name(self) -> str:
        return self._end_effector_link_name

    @property
    def ndofs(self) -> int:
        """Returns the number of degrees of freedom of the robot"""
        return self._ndofs

    @property
    def actuated_joints_limits(self) -> List[Tuple[float, float]]:
        return self._actuated_joints_limits

    def _get_klampt_active_dofs(self) -> List[int]:
        active_dofs = []
        self._klampt_robot.setConfig([0] * self._klampt_config_dim)
        idxs = [1000 * (i + 1) for i in range(self.ndofs)]
        q_temp = self._klampt_robot.configFromDrivers(idxs)
        for idx, v in enumerate(q_temp):
            if v in idxs:
                active_dofs.append(idx)
        assert len(active_dofs) == self.ndofs, f"len(active_dofs): {len(active_dofs)} != self.ndofs: {self.ndofs}"
        return active_dofs

    def sample(self, n: int, solver=None) -> np.ndarray:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors

        Args:
            n (int): _description_
            solver (_type_, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        angs = np.random.rand(n, self._ndofs)  # between [0, 1)
        joint_limits = [[l, u] for (l, u) in self._actuated_joints_limits]

        # Sample
        for i in range(self._ndofs):
            range_ = joint_limits[i][1] - joint_limits[i][0]
            assert range_ > 0
            angs[:, i] *= range_
            angs[:, i] += joint_limits[i][0]
        return angs

    def _x_to_qs(self, x: np.ndarray) -> List[List[float]]:
        """Return a list of klampt configurations (qs) from an array of joint angles (x)

        Args:
            x: (n x ndofs) array of joint angle settings

        Returns:
            A list of configurations representing the robots state in klampt
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self._ndofs

        n = x.shape[0]
        qs = []
        for i in range(n):
            qs.append(self._klampt_robot.configFromDrivers(x[i].tolist()))
        return qs

    def _qs_to_x(self, qs: List[List[float]]) -> np.array:
        """Calculate joint angle values (x) from klampt configurations (qs)"""
        res = np.zeros((len(qs), self.ndofs))
        for idx, q in enumerate(qs):
            drivers = self._klampt_robot.configToDrivers(q)
            res[idx, :] = drivers
        return res

    def assert_batch_fk_equal(self, verbosity=0):
        """
        Test that kinpy, klampt, and batch_fk all return the same poses
        """
        n_samples = 10
        samples = self.sample(n_samples)
        kinpy_fk, klampt_fk, (batch_fk_t, batch_fk_R) = get_fk_poses(self, samples)
        assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_position_almost_equal(kinpy_fk, batch_fk_t)
        if verbosity > 0:
            print("assert_batch_fk_equal(): klampt, kinpy, batch_fk functions match")

    def assert_forward_kinematics_functions_equal(self):
        """Test that kinpy and klampt the same poses"""
        n_samples = 10
        samples = self.sample(n_samples)
        kinpy_fk = self.forward_kinematics_kinpy(samples)
        klampt_fk = self.forward_kinematics_klampt(samples)
        assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
        assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)
        print(f"Success - klampt, kinpy forward-kinematics functions agree for '{self.robot_name}'")

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Forward Kinematics                                             ---
    # ---                                                                                                            ---

    def forward_kinematics(self, x: torch.tensor) -> np.array:
        """ """
        return self.forward_kinematics_klampt(x)

    def forward_kinematics_klampt(self, x: np.array) -> np.array:
        """Forward kinematics using the klampt library"""
        return klampt_fk(self._klampt_robot, self._klampt_ee_link, self._x_to_qs(x))

    def forward_kinematics_kinpy(self, x: torch.tensor) -> np.array:
        return kinpy_fk(self._kinpy_fk_chain, self._actuated_joints, self._end_effector_link_name, self.ndofs, x)

    def forward_kinematics_batch(self, x: torch.tensor, device=config.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.batch_fk_calc.batch_fk(x, device=device)

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Inverse Kinematics                                             ---
    # ---                                                                                                            ---

    def inverse_kinematics_klampt(
        self,
        pose: np.array,
        seed: Optional[np.ndarray] = None,
        positional_tolerance: float = 1e-3,
        n_tries: int = 50,
        verbosity: int = 0,
    ) -> Optional[np.array]:
        """Run klampts inverse kinematics solver with the given pose

        Note: If the solver fails to find a solution with the provided seed, it will rerun with a random seed

        Per http://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-IK.html#ik-solver:
        'To use the solver properly, you must understand how the solver uses the RobotModel:
            First, the current configuration of the robot is the seed configuration to the solver.
            Second, the robot's joint limits are used as the defaults.
            Third, the solved configuration is stored in the RobotModel's current configuration.'

        Args:
            pose (np.array): The target pose to solve for
            seed (Optional[np.ndarray], optional): A seed to initialize the optimization with. Defaults to None.
            verbosity (int): Set the verbosity of the function. 0: only fatal errors are printed. Defaults to 0.
        """
        assert len(pose.shape) == 1
        assert pose.size == 7
        if seed is not None:
            assert isinstance(seed, np.ndarray), f"seed must be a numpy array (currently {type(seed)})"
            assert len(seed.shape) == 1, f"Seed must be a 1D array (currently: {seed.shape})"
            assert seed.size == self.ndofs
            seed_q = self._x_to_qs(seed.reshape((1, self.ndofs)))[0]

        max_iterations = 150
        R = so3.from_quaternion(pose[3 : 3 + 4])
        obj = ik.objective(self._klampt_ee_link, t=pose[0:3].tolist(), R=R)

        for _ in range(n_tries):
            solver = IKSolver(self._klampt_robot)
            solver.add(obj)
            solver.setActiveDofs(self._klampt_active_dofs)
            solver.setMaxIters(max_iterations)
            # TODO(@jstmn): What does 'tolarance' mean for klampt? Positional error? Positional error + orientation error?
            solver.setTolerance(positional_tolerance)

            # `sampleInitial()` needs to come after `setActiveDofs()`, otherwise x,y,z,r,p,y of the robot will
            # be randomly set aswell <(*<_*)>
            if seed is None:
                solver.sampleInitial()
            else:
                # solver.setBiasConfig(seed_q)
                self._klampt_robot.setConfig(seed_q)

            res = solver.solve()
            if not res:
                if verbosity > 0:
                    print(
                        "  inverse_kinematics_klampt() IK failed after",
                        solver.lastSolveIters(),
                        "optimization steps, retrying (non fatal)",
                    )

                # Rerun the solver with a random seed
                if seed is not None:
                    return self.inverse_kinematics_klampt(
                        pose, seed=None, positional_tolerance=positional_tolerance, verbosity=verbosity
                    )

                continue

            if verbosity > 1:
                print("Solved in", solver.lastSolveIters(), "iterations")
                residual = solver.getResidual()
                print("Residual:", residual, " - L2 error:", np.linalg.norm(residual[0:3]))

            return self._qs_to_x([self._klampt_robot.getConfig()])

        if verbosity > 0:
            print("inverse_kinematics_klampt() - Failed to find IK solution after", n_tries, "optimization attempts")
        return None


# ----------------------------------------------------------------------------------------------------------------------
# The robots
#


class Atlas(RobotModel):
    name = "atlas"
    formal_robot_name = "ATLAS (2013) - Arm and Waist"

    def __init__(self):
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

        ndofs = len(actuated_joints)
        end_effector_link_name = "l_hand"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/atlas/atlas.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class AtlasArm(RobotModel):
    name = "atlas_arm"
    formal_robot_name = "ATLAS (2013) - Arm"

    def __init__(self):
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

        ndofs = len(actuated_joints)
        end_effector_link_name = "l_hand"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/atlas/atlas_arm.urdf"
        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class Baxter(RobotModel):
    name = "baxter"
    formal_robot_name = "Baxter"

    def __init__(self):
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
        ndofs = len(actuated_joints)
        end_effector_link_name = "left_hand"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/baxter/baxter.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )

    def forward_kinematics_batch(self, x: torch.tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("batch_fk isn't working for Baxter. Fix this bug before using this function")


class PandaArm(RobotModel):
    name = "panda_arm"
    formal_robot_name = "Panda"

    def __init__(self):
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
        ndofs = len(actuated_joints)
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
        base_link_name = "panda_link0"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


""" For Panda 2 with panda_tpm model. Note that the l2 error is fine but there is a huge angular error. This must be 
because of the different joint limits between the two urdfs 

Average L2 error:      7.4632 mm
Average angular error: 45.0683 deg
Average runtime:       13.5387 +/- 0.0011 ms (for 512 samples)
"""


class PandaArm2(RobotModel):
    name = "panda_arm2"
    formal_robot_name = "Panda"

    def __init__(self):
        actuated_joints_limits = [
            (-2.8973, 2.8973),  # panda_joint1
            (-1.7628, 1.7628),  # panda_joint2
            (-2.8973, 2.8973),  # panda_joint3
            (-3.0718, -0.0698),  # panda_joint4
            (-2.8973, 2.8973),  # panda_joint5
            (-0.0175, 3.7525),  # panda_joint6
            (-2.8973, 2.8973),  # panda_joint7
        ]
        actuated_joints = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        ndofs = len(actuated_joints)
        joint_chain = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_joint8",
        ]

        urdf_filepath = f"{config.URDFS_DIRECTORY}/panda_arm2/panda2.urdf"
        end_effector_link_name = "panda_link8"
        base_link_name = "panda_link0"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class Pr2(RobotModel):
    name = "pr2"
    formal_robot_name = "PR2"

    def __init__(self):
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

        ndofs = len(actuated_joints)
        urdf_filepath = f"{config.URDFS_DIRECTORY}/pr2/pr2.urdf"  # this is pr2_out.urdf in nn_ik repo

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )

    def forward_kinematics_batch(self, x: torch.tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("batch_fk isn't working for PR2. Fix this bug before using this function")


class Robonaut2(RobotModel):
    name = "robonaut2"
    formal_robot_name = "Robonaut 2 - Arm and Waist"

    def __init__(self):
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
        ndofs = len(actuated_joints)
        end_effector_link_name = "r2/left_palm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/robonaut2/r2b.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class Robonaut2Arm(RobotModel):
    name = "robonaut2_arm"
    formal_robot_name = "Robonaut 2 - Arm"

    def __init__(self):
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
        ndofs = len(actuated_joints)
        end_effector_link_name = "r2/left_palm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/robonaut2/r2b_arm.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class Valkyrie(RobotModel):
    name = "valkyrie"
    formal_robot_name = "Valkyrie - Whole Arm and Waist"

    def __init__(self):
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

        ndofs = len(actuated_joints)
        end_effector_link_name = "leftPalm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/valkyrie/valkyrie.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class ValkyrieArm(RobotModel):
    name = "valkyrie_arm"
    formal_robot_name = "Valkyrie - Lower Arm"

    def __init__(self):
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

        ndofs = len(actuated_joints)
        end_effector_link_name = "leftPalm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/valkyrie/valkyrie_arm.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


class ValkyrieArmShoulder(RobotModel):
    name = "valkyrie_arm_shoulder"
    formal_robot_name = "Valkyrie - Whole Arm"

    def __init__(self):
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

        ndofs = len(actuated_joints)
        end_effector_link_name = "leftPalm"
        urdf_filepath = f"{config.URDFS_DIRECTORY}/valkyrie/valkyrie_arm_shoulder.urdf"

        RobotModel.__init__(
            self,
            self.name,
            urdf_filepath,
            joint_chain,
            actuated_joints,
            actuated_joints_limits,
            ndofs,
            end_effector_link_name,
        )


ATLAS = Atlas.name
ATLAS_ARM = AtlasArm.name
ROBONAUT = Robonaut2.name
ROBONAUT_ARM = Robonaut2Arm.name
PANDA = PandaArm.name
PANDA2 = PandaArm2.name
PR2 = Pr2.name
VALKYRIE = Valkyrie.name
VALKYRIE_ARM = ValkyrieArm.name
VALKYRIE_ARM_SHOULDER = ValkyrieArmShoulder.name

ROBOT_NAMES = [
    ATLAS,
    ATLAS_ARM,
    ROBONAUT,
    ROBONAUT_ARM,
    PANDA,
    PANDA2,
    PR2,
    VALKYRIE,
    VALKYRIE_ARM,
    VALKYRIE_ARM_SHOULDER,
]


def get_all_3d_robots() -> List[RobotModel]:
    """
    Return all implemented 3d robots
    """
    return [
        # Baxter(), <- Batch_fk isn't working
        # Pr2(), <- RuntimeError: Invalid size of driver value vector
        Atlas(),
        AtlasArm(),
        PandaArm(),
        Robonaut2(),  # vis works too
        Robonaut2Arm(),
        Valkyrie(),
        ValkyrieArm(),
        ValkyrieArmShoulder(),
    ]


def get_robot(name: str, verbosity: int = 0) -> RobotModel:
    """
    Return a robot given its name.
    """
    if name == Atlas.name:
        return Atlas()
    if name == AtlasArm.name:
        return AtlasArm()
    if name == Baxter.name:
        return Baxter()
    if name == PandaArm.name:
        return PandaArm()
    if name == PandaArm2.name:
        return PandaArm2()
    if name == Pr2.name:
        return Pr2()
    if name == Robonaut2.name:
        return Robonaut2()
    if name == Robonaut2Arm.name:
        return Robonaut2Arm()
    if name == Valkyrie.name:
        return Valkyrie()
    if name == ValkyrieArm.name:
        return ValkyrieArm()
    if name == ValkyrieArmShoulder.name:
        return ValkyrieArmShoulder()
    raise ValueError(f"Unrecognized robot '{name}'.\nAvailable robots: '{ROBOT_NAMES}'")


def robot_name_to_fancy_robot_name(name: str) -> str:
    robot_cls = [
        Atlas,
        AtlasArm,
        Baxter,
        PandaArm,
        Pr2,
        Robonaut2,
        Robonaut2Arm,
        Valkyrie,
        ValkyrieArm,
        ValkyrieArmShoulder,
    ]

    for cls in robot_cls:
        if cls.name == name:
            return cls.formal_robot_name


if __name__ == "__main__":
    for robot in get_all_3d_robots():
        pass
