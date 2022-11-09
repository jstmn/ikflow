import config

from src.supporting_types import IkflowModelParameters
from src.robots import RobotModel

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch.nn as nn
import torch

VERBOSE = False


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


def glow_cNF_model(params: IkflowModelParameters, robot_model: RobotModel, dim_cond: int, ndim_tot: int):
    """
    Build an conditional invertible neural network consisting of a sequence of glow coupling layers, and permutation layers
    """

    def subnet_constructor_wrapper(ch_in: int, ch_out: int):
        return subnet_constructor(params.coeff_fn_internal_size, params.coeff_fn_config, ch_in, ch_out)

    # Input Node
    input_node = Ff.InputNode(ndim_tot, name="input")
    nodes = [input_node]
    cond = Ff.ConditionNode(dim_cond)

    # Transform Node to map x_i from joint space to [-1, 1]
    x_invSig = torch.eye(ndim_tot)
    x_Mu = torch.zeros(ndim_tot)
    for i in range(robot_model.ndofs):
        x_invSig[i, i] = 1.0 / max(
            abs(robot_model.actuated_joints_limits[i][0]), abs(robot_model.actuated_joints_limits[i][1])
        )
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

    model = Ff.GraphINN(nodes + [cond, Ff.OutputNode([nodes[-1].out0], name="output")], verbose=VERBOSE)
    model.to(config.device)
    return model
