from typing import Iterable, Tuple, List, Union

from jrl.robots import Robot
from FrEIA.modules.base import InvertibleModule
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn
import torch

from ikflow import config
from ikflow.config import SIGMOID_SCALING_ABS_MAX
from ikflow.utils import assert_joint_angle_tensor_in_joint_limits

_VERBOSE = False


class IkflowModelParameters:
    def __init__(self):
        self.coupling_layer = "glow"
        self.nb_nodes = 12
        self.dim_latent_space = 9
        self.coeff_fn_config = 3
        self.coeff_fn_internal_size = 1024
        self.permute_random_enabled = True
        self.sigmoid_on_output = False

        # ___ Loss parameters
        self.lambd_predict = 1.0  # Fit Loss lambda
        self.init_scale = 0.04473500291638653
        self.rnvp_clamp = 2.5
        self.y_noise_scale = 1e-7  # Add some noise to the cartesian poses
        self.zeros_noise_scale = 1e-3  # Also add noise to the padding

        self.softflow_noise_scale = 0.01
        self.softflow_enabled = True

    def __str__(self) -> str:
        s = "IkflowModelParameters\n"
        for k, v in self.__dict__.items():
            s += f"  {k}: \t{v}\n"
        return s


# Convenience variable for testing purposes
TINY_MODEL_PARAMS = IkflowModelParameters()
TINY_MODEL_PARAMS.nb_nodes = 3
TINY_MODEL_PARAMS.coeff_fn_config = 2
TINY_MODEL_PARAMS.coeff_fn_internal_size = 256


def subnet_constructor(internal_size: int, n_layers: int, ch_in: int, ch_out: int) -> nn.Sequential:
    """Create a MLP with width `internal_size`, depth `n_layers`, and input/output sizes of `ch_in`, `ch_out`

    Args:
        internal_size (int): Internal width of the network
        n_layers (int): Depth of the MLP
        ch_in (int): Input dimension of the MLP
        ch_out (int): Output dimension of the MLP
    """
    assert n_layers in [1, 2, 3, 4], "Number of layers `n_layers` must be in [1, ..., 4]"

    if n_layers == 1:
        return nn.Sequential(nn.Linear(ch_in, internal_size), nn.LeakyReLU(), nn.Linear(internal_size, ch_out))

    if n_layers == 2:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 3:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 4:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )


# Borrowed from 'InvertibleSigmoidFlipped' in FrEIA/modules/fixed_transforms.py
# Changes: Runs sigmoid-inverse in the forward direction and sigmoid on the reverse direction
class InvertibleSigmoidFlipped(InvertibleModule):
    """Applies the sigmoid function element-wise across all batches, and the associated
    inverse function in reverse pass. Contains no trainable parameters.
    Sigmoid function S(x) and its corresponding inverse function is given by

    .. math::

        S(x)      &= \\frac{1}{1 + \\exp(-x)} \\\\
        S^{-1}(x) &= \\log{\\frac{x}{1-x}}.

    The returning Jacobian is computed as

    .. math::

        J = \\log \\det \\frac{1}{(1+\\exp{x})(1+\\exp{-x})}.

    """

    def __init__(self, dims_in, **kwargs):
        super().__init__(dims_in, **kwargs)

    def output_dims(self, dims_in):
        return dims_in

    def forward(
        self, x_or_z: Iterable[torch.Tensor], c: Iterable[torch.Tensor] = None, rev: bool = False, jac: bool = True
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        x_or_z = x_or_z[0]
        if rev:  # if not rev:
            # S(x)
            result = 1 / (1 + torch.exp(-x_or_z))
        else:
            # S^-1(z)
            # only defined within range 0-1, non-inclusive; else, it will returns nan.
            result = torch.log(x_or_z / (1 - x_or_z))
        if not jac:
            return (result,)

        # always compute jacobian using the forward direction, but with different inputs
        _input = result if not rev else x_or_z  # _input = result if rev else x_or_z
        # the following is the diagonal Jacobian as sigmoid is an element-wise op
        logJ = torch.log(1 / ((1 + torch.exp(_input)) * (1 + torch.exp(-_input))))
        # determinant of a log diagonal Jacobian is simply the sum of its diagonals
        detLogJ = logJ.sum(1)
        if rev:  # if not rev:
            return ((result,), detLogJ)
        else:
            return ((result,), -detLogJ)


# Borrowed from 'FixedLinearTransform' in FrEIA/modules/fixed_transforms.py
# Changes: assertions to check that input is in the correct range
class IkFlowFixedLinearTransform(InvertibleModule):
    """Fixed linear transformation for 1D input tesors. The transformation is
    :math:`y = Mx + b`. With *d* input dimensions, *M* must be an invertible *d x d* tensor,
    and *b* is an optional offset vector of length *d*."""

    def __init__(
        self,
        dims_in,
        dims_c=None,
        M: torch.Tensor = None,
        b: Union[None, torch.Tensor] = None,
        joint_limits: List[Tuple[float, float]] = None,
    ):
        """Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          M: Square, invertible matrix, with which each input is multiplied. Shape ``(d, d)``.
          b: Optional vector which is added element-wise. Shape ``(d,)``.
        """
        super().__init__(dims_in, dims_c)

        # TODO: it should be possible to give conditioning instead of M, so that the condition
        # provides M and b on each forward pass.

        if M is None:
            raise ValueError("Need to specify the M argument, the matrix to be multiplied.")

        self.M = nn.Parameter(M.t(), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse(), requires_grad=False)

        if b is None:
            self.b = 0.0
        else:
            self.b = nn.Parameter(b.unsqueeze(0), requires_grad=False)

        self.joint_limits = joint_limits
        self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        j = self.logDetM.expand(x[0].shape[0])

        # Forward
        if not rev:
            assert_joint_angle_tensor_in_joint_limits(self.joint_limits, x[0], "forward", 1e-5)
            out = x[0].mm(self.M) + self.b
            assert torch.max(out).item() <= 1.0, (
                f"Output of IkFlowFixedLinearTransform node is greater than 1: {torch.max(out).item()}. This output is"
                " passed to a inverse-sigmoid so the value should be in [0, 1]\nOut"
            )
            assert torch.min(out).item() >= 0.0, (
                f"Output of IkFlowFixedLinearTransform node is less than 0: {torch.min(out).item()}. This output is"
                " passed to a inverse-sigmoid so the value should be in [0, 1]"
            )
            if torch.isnan(out).any():
                torch.save(out, "out.pt")
                torch.save(x[0], "x[0].pt")
                raise ValueError("[reverse] 'out' is NaN")
            return (out,), j
        else:
            assert torch.max(x[0]).item() <= 1.0, (
                f"Input to IkFlowFixedLinearTransform node is greater than 1: {torch.max(x[0]).item()}. This input"
                " comes from the output of a sigmoid so the value should be in [0, 1]"
            )
            assert torch.min(x[0]).item() >= 0.0, (
                f"Input to IkFlowFixedLinearTransform node is less than 0: {torch.min(x[0]).item()}. This input comes"
                " from the output of a sigmoid so the value should be in [0, 1]"
            )
            out = (x[0] - self.b).mm(self.M_inv)

            assert_joint_angle_tensor_in_joint_limits(self.joint_limits, out, "reverse", 1e-5)

            if torch.isnan(x[0]).any():
                torch.save(out, "out.pt")
                torch.save(x[0], "x[0].pt")
                raise ValueError("[reverse] x[0] is NaN")

            if torch.isnan(out).any():
                torch.save(out, "out.pt")
                torch.save(x[0], "x[0].pt")
                raise ValueError("[reverse] 'out' is NaN")
            return (out,), -j

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


def get_pre_sigmoid_scaling_node(ndim_tot: int, robot: Robot, nodes: List[Ff.Node]):
    """This node scales the joint angles to be in the range [0, 1] on the forward pass. These scaled values are then
    passed to the sigmoid function.
    """

    M_scaling = torch.eye(ndim_tot, device=config.device)
    M_offset = torch.zeros(ndim_tot, device=config.device)

    output_up = 1.0
    output_low = 0.0

    def get_offset_and_scale(input_low, input_up):
        # From https://stackoverflow.com/a/5732390/5191069
        # output = output_low + slope * (input_ - input_low)
        #       == output_low - (slope * input_low) +  slope * input_
        slope = (output_up - output_low) / (input_up - input_low)
        offset = output_low - (slope * input_low)
        return offset, slope

    # TODO: What to set for the columns of non actuated joints? For now, can enforce dim_total = ndofs
    for i, (input_low, input_up) in enumerate(robot.actuated_joints_limits):
        offset, scale = get_offset_and_scale(input_low, input_up)
        M_offset[i] = offset
        M_scaling[i, i] = scale

    # The non joint columns need to be scaled to (-SIGMOID_SCALING_ABS_MAX, SIGMOID_SCALING_ABS_MAX)
    if robot.n_dofs < ndim_tot:
        input_up = SIGMOID_SCALING_ABS_MAX
        input_low = -SIGMOID_SCALING_ABS_MAX

        for i in range(ndim_tot - robot.n_dofs):
            idx = robot.n_dofs + i
            slope = (output_up - output_low) / (input_up - input_low)
            M_offset[idx] = output_low - (slope * input_low)
            M_scaling[idx, idx] = slope

    # Return an direct implementation for unit testing
    node_impl = IkFlowFixedLinearTransform(
        [ndim_tot], M=M_scaling, b=M_offset, joint_limits=robot.actuated_joints_limits
    )
    return (
        Ff.Node(
            [nodes[-1].out0],
            IkFlowFixedLinearTransform,
            {"M": M_scaling, "b": M_offset, "joint_limits": robot.actuated_joints_limits},
        ),
        node_impl,
    )


def glow_cNF_model(params: IkflowModelParameters, robot: Robot, dim_cond: int, ndim_tot: int):
    """
    Build an conditional invertible neural network consisting of a sequence of glow coupling layers, and permutation layers
    """

    def subnet_constructor_wrapper(ch_in: int, ch_out: int):
        return subnet_constructor(params.coeff_fn_internal_size, params.coeff_fn_config, ch_in, ch_out)

    # Input Node
    input_node = Ff.InputNode(ndim_tot, name="input")
    nodes = [input_node]
    cond = Ff.ConditionNode(dim_cond)

    if params.sigmoid_on_output:
        # First map x from joint space to [0, 1]
        nodes.append(get_pre_sigmoid_scaling_node(ndim_tot, robot, nodes)[0])
        nodes.append(Ff.Node([nodes[-1].out0], InvertibleSigmoidFlipped, {}))

    else:
        # Transform Node to map x_i from joint space to [-1, 1]
        x_invSig = torch.eye(ndim_tot)
        x_Mu = torch.zeros(ndim_tot)
        for i in range(robot.n_dofs):
            x_invSig[i, i] = 1.0 / max(abs(robot.actuated_joints_limits[i][0]), abs(robot.actuated_joints_limits[i][1]))

        nodes.append(Ff.Node([nodes[-1].out0], Fm.FixedLinearTransform, {"M": x_invSig, "b": x_Mu}))

    coupling_block = Fm.GLOWCouplingBlock

    """ Note: Changes were made to `coupling_layers.py` in the Freia library after February 2021 that were not backwards
    compatabile. Specificaly, the calculation for the dimensionality of sub networks for coupling layers changed. 
    Following this change, the values for `split_len1`, `split_len2` (defined in `_BaseCouplingLayer` in 
    thirdparty/FrEIA/FrEIA/modules/coupling_layers.py) are different when the total network width is odd. The pretrained 
    models that are released with this repo were trained using a version of Freia from before this change was made, thus
    we manually set the `split_len1`, `split_len2` using the old calculation 

    See https://github.com/VLL-HD/FrEIA/commits/master/FrEIA/modules/coupling_layers.py for the git history of this 
    file. I believe commit `af72941f2867f18ea6155239413964a252c551e9` broke this compatability. Unit tests should have 
    been written to prevent this, but hey, this is research code after all ¯\_(ツ)_/¯

    The original `split_len1/2` calculation is as follows:

        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2
    """
    split_dimension = ndim_tot // 2  # // is a floor division operator

    for i in range(params.nb_nodes):
        permute_node = Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {"seed": i})
        nodes.append(permute_node)

        glow_node = Ff.Node(
            nodes[-1].out0,
            coupling_block,
            {
                "subnet_constructor": subnet_constructor_wrapper,
                "clamp": params.rnvp_clamp,
                "split_len": split_dimension,
            },
            conditions=cond,
        )
        nodes.append(glow_node)

    model = Ff.GraphINN(nodes + [cond, Ff.OutputNode([nodes[-1].out0], name="output")], verbose=_VERBOSE)
    model.to(config.device)
    return model
