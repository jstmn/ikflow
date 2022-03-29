from typing import Callable, List, Tuple
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

import torch
import numpy as np

# import kinpy as kp
import src.kinpy.kinpy as kp
from klampt.math import so3, se3
import klampt

from src import config
from src.math_utils import R_from_rpy_batch, R_from_axis_angle
from src.kinpy.kinpy.chain import Chain

# from kinpy.chain import Chain


class Link:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"<Link(), '{self.name}'>\n"


class Joint:
    def __init__(self, name, parent, child, origin_rpy, origin_xyz, axis_xyz, joint_type):
        self.name = name
        self.parent = parent
        self.child = child
        self.origin_rpy = origin_rpy
        self.origin_xyz = origin_xyz
        self.axis_xyz = axis_xyz
        self.joint_type = joint_type

    def __str__(self):
        ret = f"<Joint(), '{self.name}'>\n"
        ret += f"  joint_type:\t{self.joint_type}\n"
        ret += f"  parent:\t\t{self.parent}\n"
        ret += f"  child:\t\t{self.child}\n"
        ret += f"  origin_rpy:\t{self.origin_rpy}\n"
        ret += f"  origin_xyz:\t{self.origin_xyz}\n"
        ret += f"  axis_xyz:\t\t{self.axis_xyz}"
        return ret


def parse_urdf(urdf_name, print_joints=False) -> Tuple[List[Joint], List[Link]]:
    """
    Return all joints and links in a urdf, represented as Joints and Links
    """

    def list_from_str(str_):
        """
        Utility function used in parse_urdf()
        """
        str_orig = str_
        str_ = str_.strip()
        str_ = str_.replace("[", "")
        str_ = str_.replace("]", "")
        str_ = str_.replace("  ", " ")
        space_split = str_.split(" ")
        ret = []
        for dig in space_split:
            if len(dig) == 0:
                print("Skipping '' for str:", str_orig)
                continue
            try:
                ret.append(float(dig))
            except ValueError:
                print(space_split)
                raise RuntimeError(f"list_from_str(): error parsing string '{dig}'  from  '{str_orig}'")
        if len(ret) < 3:
            print("err in list_from_str(), ret:", ret)
        return ret

    links = []
    joints = []
    with open(urdf_name, "r") as urdf_file:
        root = ET.fromstring(urdf_file.read())
        for child in root:
            child: Element
            if child.tag == "link":
                links.append(Link(child.attrib["name"]))

            elif child.tag == "joint":
                joint_type = child.attrib["type"]
                name = child.attrib["name"]

                origin_rpy, origin_xyz, axis_xyz = None, None, None
                parent, joint_child = None, None

                for subelem in child:
                    subelem: Element
                    if subelem.tag == "origin":

                        try:
                            origin_rpy = list_from_str(subelem.attrib["rpy"])
                        except KeyError:
                            # Per (http://library.isr.ist.utl.pt/docs/roswiki/urdf(2f)XML(2f)Joint.html):
                            #       "rpy (optional: defaults 'to zero vector 'if not specified)"
                            origin_rpy = [0, 0, 0]
                        try:
                            origin_xyz = list_from_str(subelem.attrib["xyz"])
                        except RuntimeError:
                            raise ValueError(
                                f"Error: joint <joint name='{child.get('name')}'> has no xyz attribute, or it's illformed"
                            )
                    elif subelem.tag == "axis":
                        axis_xyz = list_from_str(subelem.attrib["xyz"])
                    elif subelem.tag == "parent":
                        parent = subelem.attrib["link"]
                    elif subelem.tag == "child":
                        joint_child = subelem.attrib["link"]
                joint = Joint(name, parent, joint_child, origin_rpy, origin_xyz, axis_xyz, joint_type)
                joints.append(joint)
                if print_joints:
                    print()
                    print(joint)
    return joints, links


class BatchFK:
    def __init__(
        self,
        urdf_filepath: str,
        end_effector_link_name: str,
        joint_chain: List[str],
        actuated_joints: List[str],
        l2_loss_pts: List[Tuple[float, float, float]] = [],
        print_init=False,
        max_batch_size: int = 5000,
    ):
        """Initialize a BatchFK class. BatchFK saves an internal reprsentation of the robot across querrys which is this is a stateful class and not a function.

        Args:
            urdf_filepath (str): filepath to the urdf of the robot
            end_effector_link_name (str): The link name in the urdf of the robots end effector
            joint_chain (List[str]): A list of all joints in the kinematic sub chain whose inverse kinematics are being learned. Note: This includes all joints in the subchain - both actuated and non actuated.
            actuated_joints (List[str]): A list of all actuated joints in the kinematic sub chain being learned
        """

        self.urdf_filepath = urdf_filepath
        self.end_effector_link_name = end_effector_link_name
        self.joint_chain = joint_chain
        self.actuated_joints = actuated_joints
        self.max_batch_size = max_batch_size

        self.l2_loss_pts = l2_loss_pts
        if len(self.l2_loss_pts) > 0:
            R = R_from_rpy_batch([0, 0, 0], device="cpu").unsqueeze(0).repeat(self.max_batch_size, 1, 1)
            self.l2_loss_pts_rotations = []

        self.joints, self.links = parse_urdf(self.urdf_filepath, print_joints=print_init)

        self.joints_dict = {}
        for joint in self.joints:
            self.joints_dict[joint.name] = joint

        self.links_dict = {}
        for link in self.links:
            self.links_dict[link.name] = link

        # Error checking
        for joint in self.joint_chain:
            if not joint in self.joints_dict:
                raise ValueError(f"joint['{joint}'] not in self.joints")

        if self.end_effector_link_name not in self.links_dict:
            raise ValueError(f"link['{self.end_effector_link_name}'] not in self.links")

        for i in range(len(self.joint_chain) - 1):
            joint = self.joints_dict[self.joint_chain[i]]
            joint_plus1 = self.joints_dict[self.joint_chain[i + 1]]
            child_link_name = joint.child
            if joint_plus1.parent != child_link_name:
                raise ValueError(
                    f"Error: joint_{i}['{joint.name}'].child('{child_link_name}') != joint_{i + 1}.parent('{child_link_name}')"
                )

        final_joint_name = self.joint_chain[-1]
        final_joint_child_link_name = self.joints_dict[final_joint_name].child
        if self.links_dict[final_joint_child_link_name].name != self.end_effector_link_name:
            raise ValueError(
                f"Error: the final joint in joints_chain's['{final_joint_name}'] child['{final_joint_child_link_name}'] != "
                f"end_effector_link_name['{self.end_effector_link_name}']"
            )

        # Cache fixed rotations between links
        self.fixed_rotations = {}
        for joint in self.joints:
            try:
                R = R_from_rpy_batch(joint.origin_rpy, device="cpu").unsqueeze(0).repeat(self.max_batch_size, 1, 1)
                self.fixed_rotations[joint.name] = R
            except TypeError:
                # print(f"error parsing joint '{joint.name}' -> joint.origin_rpy:{joint.origin_rpy} is none")
                # TODO: This is broken for continuous joints (like wi/ Robosimian)
                pass

        if print_init:
            print(f"BatchFK().__init__\turdf_name: '{self.urdf_filepath}'")

    def batch_fk(self, x: torch.tensor, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate through each joint in `self.joint_chain` and apply the joint's fixed transformation. If
        the joint is revolute then apply rotation x[i] and increment i
        """

        if device is None:
            device = config.device

        batch_sz = x.shape[0]
        R0 = torch.eye(3)
        t0 = torch.zeros(3)

        R = R0.unsqueeze(0).repeat(batch_sz, 1, 1).to(device)  # ( batch_sz x 3 x 3 )
        t = t0.unsqueeze(0).repeat(batch_sz, 1).to(device)  # ( batch_sz x 3 )
        # default_rotation_amt = torch.zeros((batch_sz, 1), device=device)
        default_rotation_amt = torch.zeros((batch_sz, 1), device=device).to(config.device)

        assert batch_sz < self.max_batch_size, f"Error: batch size needs to be less than {self.max_batch_size}"

        # set fixed_rotations to current device
        for fixed_rotation_i in self.fixed_rotations:
            R_i = self.fixed_rotations[fixed_rotation_i].to(device)
            self.fixed_rotations[fixed_rotation_i] = R_i

        # Iterate through each joint in the joint chain
        x_i = 0
        for joint_name_i in self.joint_chain:

            joint = self.joints_dict[joint_name_i]
            t = t + torch.matmul(R, torch.tensor(joint.origin_xyz, device=device))

            # Rotation between joint frames
            R_parent_to_child = self.fixed_rotations[joint.name][0:batch_sz, :, :]

            rotation_axis = [1, 0, 0]
            rotation_amt = default_rotation_amt

            # TODO(@jeremysm): currently this script iterates through each joint in the robot's joint_chain. If the
            #                  joint is revolute then it is set to the joint angles stored at x[i] (i is then
            #                  incremented). If a joint is revolute but not in the robot's active_joint_list, then it's
            #                  currently unclear what should be done.
            assert not (
                joint.joint_type == "revolute" and joint_name_i not in self.actuated_joints
            ), f"joint '{joint_name_i}' is revolute but not in robot '{self.name}''s actuated_joints"

            if joint.joint_type == "revolute":
                rotation_amt = x[:, x_i]
                rotation_axis = joint.axis_xyz
                x_i += 1

            # actuator rotation about joint.axis_xyz by x[:, x_i] degrees.
            rotation_amt = rotation_amt.to(config.device)
            joint_rotation = R_from_axis_angle(rotation_axis, rotation_amt, device=device)[:, 0:3, 0:3]

            R = R.bmm(R_parent_to_child).bmm(joint_rotation)
        return t, R

    def batch_fk_w_l2_loss_pts(self, x: torch.tensor, device=config.device) -> torch.Tensor:
        assert len(self.l2_loss_pts) > 0

        batch_size = x.shape[0]
        world_T_ee = self.batch_fk(x, device=device)
        t, R = world_T_ee

        n_poses_per_sample = 1 + len(self.l2_loss_pts)
        t_returned = torch.zeros((batch_size, 3 * n_poses_per_sample)).to(device)
        t_returned[:, 0:3] = t

        for i, l2_loss_pt in enumerate(self.l2_loss_pts):
            ee_T_pt_just_translation = torch.tensor(l2_loss_pt).to(device)
            t_i = t + torch.matmul(R, ee_T_pt_just_translation)
            base_idx = 3 + 3 * i
            t_returned[:, base_idx : base_idx + 3] = t_i
        return t_returned


def kinpy_fk(
    kinpy_fk_chain: Chain, actuated_joints: List[str], end_effector_link_name: str, dim_x: int, x: np.array
) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x
    """

    assert not isinstance(x, torch.Tensor), "torch tensors not supported"
    assert len(x.shape) == 2, f"x must be (m, n), currently: {x.shape}"

    n = x.shape[0]
    y = np.zeros((n, 7))
    zero_transform = kp.transform.Transform()
    fk_dict = {}
    for joint_name in actuated_joints:
        fk_dict[joint_name] = 0.0

    def get_fk_dict(xs):
        for i in range(dim_x):
            fk_dict[actuated_joints[i]] = xs[i]
        return fk_dict

    for i in range(n):
        th = get_fk_dict(x[i])
        transform = kinpy_fk_chain.forward_kinematics(th, world=zero_transform)[end_effector_link_name]
        t = transform.pos
        quat = transform.rot

        y[i, 0:3] = t
        y[i, 3:] = quat

    return y


def klampt_fk(
    klampt_robot: klampt.RobotModel,
    klampt_end_effector_link: klampt.RobotModelLink,
    robot_configs: List[List[float]],
) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x. Forward kinemaitcs calculated with
    klampt
    """
    dim_y = 7
    n = len(robot_configs)
    y = np.zeros((n, dim_y))

    for i in range(n):
        q = robot_configs[i]
        klampt_robot.setConfig(q)

        R, t = klampt_end_effector_link.getTransform()
        y[i, 0:3] = np.array(t)
        y[i, 3:] = np.array(so3.quaternion(R))

    return y


def klampt_fk_w_l2_loss_pts(
    klampt_robot: klampt.RobotModel,
    klampt_end_effector_link: klampt.RobotModelLink,
    robot_configs: List[List[float]],
    l2_loss_pts: List[Tuple[float, float, float]],
) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x. Forward kinemaitcs calculated with
    klampt
    """
    dim_y = 7
    n = len(robot_configs)
    n_loss_pts = len(l2_loss_pts)
    n_poses_per_sample = 1 + n_loss_pts
    y = np.zeros((n, dim_y * n_poses_per_sample))

    # Get ee_T_l2_loss_pts
    ee_T_l2lps = []
    for l2lp in l2_loss_pts:
        ee_T_l2lp = ([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], l2lp)
        ee_T_l2lps.append(ee_T_l2lp)

    for i in range(n):
        q = robot_configs[i]
        klampt_robot.setConfig(q)

        world_T_ee = klampt_end_effector_link.getTransform()
        R, t = world_T_ee

        y[i, 0:3] = np.array(t)
        y[i, 3 : 3 + 4] = np.array(so3.quaternion(R))

        for j in range(len(l2_loss_pts)):
            root_idx = 7 * (1 + j)

            world_T_l2lp = se3.mul(world_T_ee, ee_T_l2lps[j])
            R_j, t_j = world_T_l2lp

            y[i, root_idx + 0 : root_idx + 3] = t_j
            y[i, root_idx + 3 : root_idx + 3 + 4] = np.array(so3.quaternion(R_j))

    return y
