from typing import Union

import torch
import numpy as np

from ikflow import config
from ikflow.config import DEFAULT_TORCH_DTYPE

PT_NP_TYPE = Union[np.ndarray, torch.Tensor]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion from wxyz to xyzw format"""
    assert q.size == 4
    # In:  w x y z
    # Out: x y z w
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion from xyzw to wxyz format"""
    assert q.size == 4
    # In:  x y z w
    # Out: w x y z
    return np.array([q[3], q[0], q[1], q[2]])


def MMD_multiscale(x, y, c_list, a_list, reduce=True):
    """Example usage:
        MMD_multiscale(x0, x1, rev_kernel_width, reverse_loss_a, reduce=False)


    Example usage in toy-inverse-kinematics:

        rev_kernel_width = 1.1827009364464547

        `backward_mmd(x0, x1, *y_args)`:
        mmd = MMD_multiscale(x0, x1, [c.rev_kernel_width, c.rev_kernel_width, c.rev_kernel_width], [0.2, 1.0, 2.0])

        `latent_mmd(y0, y1)`:
        mmd = MMD_multiscale(y0, y1, [0.1, 0.2, 0.5], [0.5, 1.0, 2.0])

    """
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = torch.clamp(rx.t() + rx - 2.0 * xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.0 * yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.0 * xy, 0, np.inf)

    XX = torch.zeros(xx.shape).to(config.device)
    YY = torch.zeros(xx.shape).to(config.device)
    XY = torch.zeros(xx.shape).to(config.device)

    for C, a in zip(c_list, a_list):
        XX += C**a * ((C + dxx) / a) ** -a
        YY += C**a * ((C + dyy) / a) ** -a
        XY += C**a * ((C + dxy) / a) ** -a

    if reduce:
        return torch.mean(XX + YY - 2.0 * XY)
    else:
        return XX + YY - 2.0 * XY


# TODO: Benchmark speed when running this with numpy. Does it matter if its slow?
def geodesic_distance_between_quaternions(q1: PT_NP_TYPE, q2: PT_NP_TYPE) -> PT_NP_TYPE:
    """
    Given rows of quaternions q1 and q2, compute the geodesic distance between each
    """

    assert len(q1.shape) == 2
    assert len(q2.shape) == 2
    assert q1.shape[0] == q2.shape[0]
    assert q1.shape[1] == q2.shape[1]

    if isinstance(q1, np.ndarray):
        q1_R9 = rotation_matrix_from_quaternion(torch.tensor(q1, device="cpu", dtype=DEFAULT_TORCH_DTYPE))
        q2_R9 = rotation_matrix_from_quaternion(torch.tensor(q2, device="cpu", dtype=DEFAULT_TORCH_DTYPE))

    if isinstance(q1, torch.Tensor):
        q1_R9 = rotation_matrix_from_quaternion(q1)
        q2_R9 = rotation_matrix_from_quaternion(q2)

    distance = geodesic_distance(q1_R9, q2_R9)
    if isinstance(q1, np.ndarray):
        distance = distance.numpy()
    return distance


def quaternion_to_rpy(q: np.array):
    """Return roll pitch yaw"""
    roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([roll, pitch, yaw])


def quaternion_to_rpy_batch(q: torch.Tensor, device=None) -> torch.Tensor:
    assert len(q.shape) == 2
    assert q.shape[1] == 4

    batch = q.shape[0]
    device = config.device if device is None else device

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    p = torch.asin(2 * (q0 * q2 - q3 * q1))

    rpy = torch.zeros((batch, 3)).to(device)
    rpy[:, 0] = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rpy[:, 1] = p
    rpy[:, 2] = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rpy


"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix
Args:
    angle_axis (torch.Tensor): tensor of 3d vector of axis-angle rotations.
Returns:
    torch.Tensor: tensor of 3x3 rotation matrices.
Shape:
    - Input: :math:`(N, 3)`
    - Output: :math:`(N, 3, 3)`
"""


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    assert isinstance(angle_axis, torch.Tensor), f"Expected torch.Tensor. Got {type(angle_axis)}"
    assert angle_axis.shape[-1] == 3, f"Expected tensor shape to tbe [batch x 3]. Got {angle_axis.shape}"

    """A simple fix is to add the already previously defined eps to theta2 instead of to theta. Although that could result in
    theta being very small compared to eps, so I've included theta2+eps and theta+eps."""

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0

        # With eps. fix
        theta = torch.sqrt(theta2 + eps)
        wxyz = angle_axis / (theta + eps)

        # Original code
        # theta = torch.sqrt(theta2)
        # wxyz = angle_axis / (theta + eps)

        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h
    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


# T_poses num*3
# r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch = r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch, 1, 3, 3).expand(batch, joint_num, 3, 3).contiguous().view(batch * joint_num, 3, 3)
    src_poses = (
        T_pose.view(1, joint_num, 3, 1).expand(batch, joint_num, 3, 1).contiguous().view(batch * joint_num, 3, 1)
    )

    out_poses = torch.matmul(r_matrices, src_poses)  # (batch*joint_num)*3*1

    return out_poses.view(batch, joint_num, 3)


_TORCH_EPS_CPU = torch.tensor(1e-8, dtype=DEFAULT_TORCH_DTYPE, device="cpu")
_TORCH_EPS_CUDA = torch.tensor(1e-8, dtype=DEFAULT_TORCH_DTYPE, device="cuda")


# batch*n
def normalize_vector(v: torch.Tensor, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch

    if v.is_cuda:
        v_mag = torch.max(v_mag, _TORCH_EPS_CUDA)
    else:
        v_mag = torch.max(v_mag, _TORCH_EPS_CPU)
    # v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(config.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag == True:
        return v, v_mag[:, 0]
    else:
        return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# in:  batch * rpy(x,y,z)
# out: batch*3*3
def compute_rotation_matrix_from_rpy(rpy):
    # rx , ry, rz
    n = rpy.shape[0]

    x_rots = rpy[:, 0]  # TODO: This can't be right???!
    y_rots = rpy[:, 0]
    z_rots = rpy[:, 0]

    Rx = torch.zeros((n, 3, 3)).to(config.device)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(x_rots)
    Rx[:, 1, 2] = -torch.sin(x_rots)
    Rx[:, 2, 1] = torch.sin(x_rots)
    Rx[:, 2, 2] = torch.cos(x_rots)

    Ry = torch.zeros((n, 3, 3)).to(config.device)
    Ry[:, 0, 0] = torch.cos(y_rots)
    Ry[:, 0, 2] = torch.sin(y_rots)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -torch.sin(y_rots)
    Ry[:, 2, 2] = torch.cos(y_rots)

    Rz = torch.zeros((n, 3, 3)).to(config.device)
    Rz[:, 0, 0] = torch.cos(z_rots)
    Rz[:, 0, 1] = -torch.sin(z_rots)
    Rz[:, 1, 1] = torch.sin(z_rots)
    Rz[:, 1, 1] = torch.cos(z_rots)
    Rz[:, 2, 2] = 1

    matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))
    return matrix


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


# in batch*6
# out batch*5
def stereographic_project(a):
    dim = a.shape[1]
    a = normalize_vector(a)
    out = a[:, 0 : dim - 1] / (1 - a[:, dim - 1])
    return out


# in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
    Inverse of stereographic projection: increases dimension by one.
    """
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)  # batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1] + 1).to(config.device))  # batch*6
    unproj = 2 * a / (s2 + 1).view(batch, 1).repeat(1, a.shape[1])  # batch*5
    if axis > 0:
        ans[:, :axis] = unproj[:, :axis]  # batch*(axis-0)
    ans[:, axis] = (s2 - 1) / (s2 + 1)  # batch
    ans[:, axis + 1 :] = unproj[
        :, axis:
    ]  # batch*(5-axis)    # Note that this is a no-op if the default option (last axis) is used
    return ans


# a batch*5
# out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  # 3
    proj_scale = (
        torch.autograd.Variable(torch.FloatTensor(proj_scale_np).to(config.device)).view(1, 3).repeat(batch, 1)
    )  # batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  # batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  # batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  # batch*4
    b = torch.cat((a[:, 0:2], u), 1)  # batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


# quaternion batch*4
def rotation_matrix_from_quaternion(quaternion: torch.Tensor):
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle(axisAngle):
    batch = axisAngle.shape[0]

    theta = torch.tanh(axisAngle[:, 0]) * np.pi  # [-180, 180]
    sin = torch.sin(theta)
    axis = normalize_vector(axisAngle[:, 1:4])  # batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*3 (x,y,z)*theta
def compute_rotation_matrix_from_Rodriguez(rod):
    batch = rod.shape[0]

    axis, theta = normalize_vector(rod, return_mag=True)

    sin = torch.sin(theta)

    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf(hopf):
    batch = hopf.shape[0]

    theta = (torch.tanh(hopf[:, 0]) + 1.0) * np.pi / 2.0  # [0, pi]
    phi = (torch.tanh(hopf[:, 1]) + 1.0) * np.pi  # [0,2pi)
    tao = (torch.tanh(hopf[:, 2]) + 1.0) * np.pi  # [0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# euler_sin_cos batch*6
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch = euler_sin_cos.shape[0]

    s1 = euler_sin_cos[:, 0].view(batch, 1)
    c1 = euler_sin_cos[:, 1].view(batch, 1)
    s2 = euler_sin_cos[:, 2].view(batch, 1)
    c2 = euler_sin_cos[:, 3].view(batch, 1)
    s3 = euler_sin_cos[:, 4].view(batch, 1)
    c3 = euler_sin_cos[:, 5].view(batch, 1)

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def geodesic_distance(m1: torch.Tensor, m2: torch.Tensor):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    # cos = torch.min(cos, torch.ones(batch, device=m1.device, dtype=m1.dtype))
    # cos = torch.max(cos, torch.ones(batch, device=m1.device, dtype=m1.dtype) * -1)
    # theta = torch.acos(cos)

    # See https://github.com/pytorch/pytorch/issues/8069#issuecomment-700397641
    epsilon = 1e-7 
    theta = torch.acos(torch.clamp(cos, -1 + epsilon, 1 - epsilon))
    return theta


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(config.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(config.device)) * -1)

    theta = torch.acos(cos)

    return theta


def get_sampled_rotation_matrices_by_quat(batch):
    # quat = torch.autograd.Variable(torch.rand(batch,4).to(config.device))
    quat = torch.autograd.Variable(torch.randn(batch, 4).to(config.device))
    matrix = rotation_matrix_from_quaternion(quat)
    return matrix


def get_sampled_rotation_matrices_by_hpof(batch):
    theta = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(0, 1, batch) * np.pi).to(config.device)
    )  # [0, pi]
    phi = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).to(config.device)
    )  # [0,2pi)
    tao = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).to(config.device)
    )  # [0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# axisAngle batch*4 angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle(batch, return_quaternion=False):
    theta = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(-1, 1, batch) * np.pi).to(config.device)
    )  # [0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).to(config.device))
    axis = normalize_vector(axis)  # batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    quaternion = torch.cat((qw.view(batch, 1), qx.view(batch, 1), qy.view(batch, 1), qz.view(batch, 1)), 1)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    if return_quaternion == True:
        return matrix, quaternion
    else:
        return matrix


def R_from_rpy_batch(rpy, device=config.device):
    """ """
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]

    Rx = torch.eye(3).to(device)
    Rx[1, 1] = np.cos(r)
    Rx[1, 2] = -np.sin(r)
    Rx[2, 1] = np.sin(r)
    Rx[2, 2] = np.cos(r)

    Ry = torch.eye(3).to(device)
    Ry[0, 0] = np.cos(p)
    Ry[0, 2] = np.sin(p)
    Ry[2, 0] = -np.sin(p)
    Ry[2, 2] = np.cos(p)

    Rz = torch.eye(3).to(device)
    Rz[0, 0] = np.cos(y)
    Rz[0, 1] = -np.sin(y)
    Rz[1, 0] = np.sin(y)
    Rz[1, 1] = np.cos(y)

    R = Rz.mm(Ry.mm(Rx))
    return R


def R_from_axis_angle(axis, ang: torch.tensor, device=config.device):
    """
    axis: (3,) vector
    ang:  (batch_sz, 1) matrix
    """
    angleaxis = torch.tensor(axis).unsqueeze(0).repeat(ang.shape[0], 1).to(device)
    ang = ang.view(-1, 1)
    angleaxis = angleaxis * ang
    R = angle_axis_to_rotation_matrix(angleaxis)
    return R
