from abc import abstractclassmethod
from typing import List, Union, Tuple, Optional
import os
import pickle
from time import time

from FrEIA.framework.graph_inn import GraphINN

from src.mdn.mdn.models import MixtureDensityNetwork
from src import config, robot_models
from src.robot_models import KlamptRobotModel
from src.training_parameters import MDNParameters, cINN_SoftflowParameters, MultiCINN_Parameters, IResNetParameters

import numpy as np
from wandb.wandb_run import Run
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn


# Need to call this when running on cluster. otherwise will get "tkinter.TclError: no display name and no $DISPLAY environment variable"
if config.is_cluster:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERBOSE = False


def glow_subnet(internal_size: int, n_layers: int, ch_in: int, ch_out: int):
    """Create a coefficient network"""

    assert n_layers in [1, 2, 3, 4, 5, 6, 7], "Number of layers `n_layers` must be in [1, ..., 7]"

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

    if n_layers == 5:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 6:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )

    if n_layers == 7:
        return nn.Sequential(
            nn.Linear(ch_in, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, ch_out),
        )


COUPLING_BLOCKS = {
    "glow": Fm.GLOWCouplingBlock,
    "rnvp": Fm.RNVPCouplingBlock,
}


def glow_nn_model(params: cINN_SoftflowParameters, dim_cond: int, ndim_tot: int, robot_model: robot_models.RobotModel):
    """
    Build a nn_model consisting of a sequence of Glow coupling layers
    """

    def glow_subnet_wrapper(ch_in: int, ch_out: int):
        return glow_subnet(params.coeff_fn_internal_size, params.coeff_fn_config, ch_in, ch_out)

    # Input Node
    input_node = Ff.InputNode(ndim_tot, name="input")
    nodes = [input_node]
    cond = Ff.ConditionNode(dim_cond)

    # Transform Node to map x_i from joint space to [-1, 1]
    x_invSig = torch.eye(ndim_tot)
    x_Mu = torch.zeros(ndim_tot)
    for i in range(robot_model.dim_x):
        x_invSig[i, i] = 1.0 / max(
            abs(robot_model.actuated_joints_limits[i][0]), abs(robot_model.actuated_joints_limits[i][1])
        )
    nodes.append(Ff.Node([nodes[-1].out0], Fm.FixedLinearTransform, {"M": x_invSig, "b": x_Mu}))

    coupling_block = COUPLING_BLOCKS[params.coupling_layer]

    permute_random = True
    if "permute_random_enabled" in params.__dict__:
        permute_random = params.permute_random_enabled

    for i in range(params.nb_nodes):

        if permute_random:
            permute_node = Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {"seed": i})
            nodes.append(permute_node)

        glow_node = Ff.Node(
            nodes[-1].out0,
            coupling_block,
            {"subnet_constructor": glow_subnet_wrapper, "clamp": params.rnvp_clamp},
            conditions=cond,
        )

        nodes.append(glow_node)

    model = Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode([nodes[-1].out0], name="output")], verbose=VERBOSE)
    model.to(config.device)
    return model


def multi_cinn_model(params: MultiCINN_Parameters, robot_model: robot_models.RobotModel) -> List[GraphINN]:
    """
    Build a nn_model consisting of a number of individual graphs that are then combine to stand in for a larger network
    """

    # Dimension of the robots joints, and end effector
    dim_x = robot_model.dim_x
    dim_y = robot_model.dim_y

    # Base dimension of the condition value
    base_dim_cond = dim_y
    if params.softflow_enabled:
        base_dim_cond += 1

    # Coupling layer used
    coupling_block = COUPLING_BLOCKS[params.coupling_layer]

    def glow_subnet_wrapper(ch_in: int, ch_out: int):
        return glow_subnet(params.subnet_coeff_fn_width, params.subnet_coeff_fn_depth, ch_in, ch_out)

    reversible_graph_list = []

    for i in range(dim_x):

        # Input Node
        input_node = Ff.InputNode(params.subnet_network_width, name="input")
        nodes = [input_node]
        cond = Ff.ConditionNode(base_dim_cond + i)

        # Transform Node to map x_i from joint space to [-1, 1]
        # x_invSig = torch.eye(2)
        # x_Mu = torch.zeros(2)
        # x_invSig[0,0] = 1.0 / max(
        #         abs(robot_model.actuated_joints_limits[i][0]), abs(robot_model.actuated_joints_limits[i][1])
        #     )
        # nodes.append(Ff.Node([nodes[-1].out0], Fm.FixedLinearTransform, {"M": x_invSig, "b": x_Mu}))

        for j in range(params.subnet_nb_nodes):
            # for j in range(params.nb_nodes):
            permute_node = Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {"seed": j})
            nodes.append(permute_node)

            glow_node = Ff.Node(
                nodes[-1].out0,
                coupling_block,
                {"subnet_constructor": glow_subnet_wrapper, "clamp": params.rnvp_clamp},
                conditions=cond,
            )

            nodes.append(glow_node)

        model = Ff.GraphINN(nodes + [cond, Ff.OutputNode([nodes[-1].out0], name="output")], verbose=VERBOSE)
        model.to(config.device)
        reversible_graph_list.append(model)

    return reversible_graph_list


def mdn_model(n_components: int, robot_model: robot_models.RobotModel):
    """Build and return a Mixture Density Network. See https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf. Implementation from https://github.com/tonyduan/mdn

    Args:
        n_components (int): Number of components in the model

    Returns:
        mdn-model [MixtureDensityNetwork]: A Mixture density network
    """
    return MixtureDensityNetwork(robot_model.dim_y, robot_model.dim_x, n_components)


def draw_latent_noise(user_specified_latent_noise, latent_noise_distribution, latent_noise_scale, shape):
    """Draw a sample from the latent noise distribution for running inference"""
    assert latent_noise_distribution in ["gaussian", "uniform"]
    assert latent_noise_scale > 0
    assert len(shape) == 2
    if user_specified_latent_noise is not None:
        return user_specified_latent_noise
    if latent_noise_distribution == "gaussian":
        return latent_noise_scale * torch.randn(shape).to(config.device)
    elif latent_noise_distribution == "uniform":
        return 2 * latent_noise_scale * torch.rand(shape).to(config.device) - latent_noise_scale


class ModelWrapper:
    """Superclass to represent ModelWrappers around nn's. Provides a standard set of functions that will be implemented by subclasses"""

    def __init__(
        self,
        nn_model_type: str,
        nn_model: Union[MixtureDensityNetwork, Ff.ReversibleGraphNet],
        robot_model: robot_models.RobotModel,
        cuda_out_of_memory=False,
    ):
        """Initialize a ModelWrapper.

        Args:
            nn_model_type (str): The type of the model. One of ["cinn_softflow", "mdn"]

        """
        assert nn_model_type in ["cinn_softflow", "mdn", "iresnet", "multi_cinn"]

        self.nn_model_type = nn_model_type
        self.nn_model = nn_model
        self.robot_model = robot_model
        self.dim_x = self.robot_model.dim_x
        self.dim_y = robot_model.dim_y

        # TODO(@jeremysm): Clean up this tech. debts
        self.n_parameters = -1

        # Check to see if the gpu is out of memory after building the inn. Return uninitialized if so.
        self.cuda_out_of_memory = cuda_out_of_memory
        if self.cuda_out_of_memory:
            print("ModelWrapper ERROR: Cuda is out of memory. created object will be uninitilized")
            return

        if self.nn_model_type == "multi_cinn":
            self.n_parameters = 0
            for model in self.nn_model:
                self.n_parameters = self.n_parameters + sum(p.numel() for p in model.parameters())
        else:
            self.n_parameters = sum(p.numel() for p in self.nn_model.parameters())

        print(f"Created model of type '{self.nn_model_type}' ( w/ {self.n_parameters} parameters)")

    @property
    def robot(self) -> KlamptRobotModel:
        return self.robot_model

    def load(self, artifact_name: str, wandb_run: Run):
        """
        Load the parameters given the `artifact_name` where the model is saved under in wandb
        """

        artifact = wandb_run.use_artifact(artifact_name + ":latest")
        download_path = artifact.download()
        state_dict_filepath = os.path.join(download_path, "model.pkl")

        with open(state_dict_filepath, "rb") as f:
            # self.nn_model = torch.load(f, map_location=torch.device('cpu'))
            self.nn_model.load_state_dict(pickle.load(f))

        print(f"Initialized nn_model from wandb artifact '{artifact_name}' (filepath: '{state_dict_filepath}')")

    def load_state_dict(self, state_dict_filename: str):
        """Set the nn_models state_dict"""
        with open(state_dict_filename, "rb") as f:
            self.nn_model.load_state_dict(pickle.load(f))

    @abstractclassmethod
    def make_samples(
        self,
        y: List[float],
        m: int,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y

        Args:
            y (List[float]):                 Target endpose. For a 3d robot, should be [x, y, z, q0, q1, q2, q3]
            m (int):                         The number of samples to draw
            rev_inputs (torch.Tensor):       A (_ x dim_total) tensor that is the reverse input to the model. Note that `y`
                                             and `m` will be ignored if this variable is passed
            latent_noise (torch.Tensor):     A (m x dim_total) tensor that is the reverse input to the model. If this value
                                             is passed, `latent_noise_distribution` and `latent_noise_scale` will be ignored
            latent_noise_distribution (str): One of ["gaussian", "uniform"]
            latent_noise_scale (float):      Scaling factor for the latent noise distribution. If distribution is gaussian,
                                             the sampled gaussian is multiplied by this value. If the distribution is
                                             uniform, this value describes the width/2 of the distribution. For example, if
                                             the `latent_noise_distribution` is "uniform", and `latent_noise_scale` = .75,
                                             the latent noise will be drawn from a uniform[-.75, .75] distribution.
        """
        # Implemented by subclasses
        raise NotImplementedError()

    @abstractclassmethod
    def make_samples_mult_y(
        self,
        ys: np.array,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        raise NotImplementedError()

    def plot_distribution(
        self,
        pose: np.array,
        n_samples: int,
        ik_samples: np.array,
        n_bins=100,
        output_file="",
        show_fig=False,
        fig_title="",
    ):
        assert "numpy" in str(type(pose))
        #         plt.rcParams["figure.constrained_layout.use"] = True
        def plot(ax, _values, label):
            # See https://github.com/matplotlib/matplotlib/issues/10398/#issuecomment-366480230
            weights = np.ones(len(_values)) / len(_values)
            ax.hist(
                _values,
                bins=n_bins,
                histtype="step",
                range=(-np.pi, np.pi),
                label=label,
                density=False,
                weights=weights,
            )

        samples, _ = self.make_samples(pose, m=n_samples)

        if "TORCH" in str(type(samples)).upper():
            samples = samples.cpu().detach().numpy()
        if "TORCH" in str(type(ik_samples)).upper():
            ik_samples = ik_samples.cpu().detach().numpy()

        # plt.figure(figsize=(10, 6))
        #         fig, axes = plt.subplots(self.dim_x, 1, constrained_layout=True)
        fig, axes = plt.subplots(self.dim_x, 1)
        fig.set_figheight(20)
        fig.set_figwidth(12)
        if len(fig_title) > 0:
            fig.suptitle(fig_title)

        for i, ax in enumerate(axes):
            # ax = axes[i, 0]
            plot(ax, samples[:, i], "Generated samples")
            plot(ax, ik_samples[:, i], "IK generated")
            ax.set_xlabel(f"Joint_{i} values")
            ax.set_ylabel(f"Probability")
            # ax.set_title(f"Joint_{i} angle pdf for pose: {pose}")
            ax.legend()

        if len(output_file) > 0:
            dpi = "figure"
            plt.savefig(output_file, dpi=dpi)

        if show_fig:
            # plt.tight_layout()
            plt.show()
        plt.close()

    def __str__(self):
        s = f"ModelWrapper('{self.nn_model_type}')\n"
        s += "  robot:" + self.robot_model.__str__()
        return s


class ModelWrapper_cINN_Softflow(ModelWrapper):
    """A ModelWrapper subclass for a conditional invertible neural network with softflow. That's a mouthful"""

    nn_model_type = "cinn_softflow"

    def __init__(self, hyper_parameters: cINN_SoftflowParameters, robot_model: robot_models.RobotModel):
        """Initialize a ModelWrapper_cINN_Softflow."""
        assert isinstance(hyper_parameters, cINN_SoftflowParameters)

        if hyper_parameters.softflow_enabled:
            self.dim_cond = robot_model.dim_y + 1
        else:
            self.dim_cond = robot_model.dim_y

        # TODO(@jeremysm): Add `dim_tot` as a member value to ModelWrapper
        self.dim_tot = max(hyper_parameters.dim_latent_space, robot_model.dim_x)
        cuda_out_of_memory = False
        nn_model = None
        try:
            nn_model = glow_nn_model(hyper_parameters, self.dim_cond, self.dim_tot, robot_model)
        except RuntimeError as e:
            print(f"ModelWrapper_cINN_Softflow() caught error '{e}' in __init__() function.")
            if "out of memory" in str(e) and "CUDA" in str(e):
                cuda_out_of_memory = True

        ModelWrapper.__init__(
            self, ModelWrapper_cINN_Softflow.nn_model_type, nn_model, robot_model, cuda_out_of_memory=cuda_out_of_memory
        )

    def make_samples(
        self,
        y: List[float],
        m: int,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y
                y: endpose [x, y, z, q0, q1, q2, q3]
                m: Number of samples
        """
        assert len(y) == self.dim_y
        assert latent_noise_distribution in ["gaussian", "uniform"]

        # Note: No code change required here to handle using/not using softflow.
        conditional = torch.zeros(m, self.dim_cond)

        if self.robot_model.dim_y == 7:
            # (:, 0:3) is x, y, z
            conditional[:, 0:3] = torch.FloatTensor(y[:3])
            # (:, 3:7) is quat
            conditional[:, 3 : 3 + 4] = torch.FloatTensor([y[3:]])
        else:
            conditional[:, 0:2] = torch.FloatTensor(y)

        # (:, 7: 7 + ndim_tot) is the sigma of the normal distribution that noise is drawn from and added to x. 0 in this case
        conditional = conditional.to(config.device)

        shape = (m, self.dim_tot)
        latent_noise = draw_latent_noise(latent_noise, latent_noise_distribution, latent_noise_scale, shape)
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.dim_tot

        t0 = time()
        output_rev, jac = self.nn_model(latent_noise, c=conditional, rev=True)
        return output_rev[:, 0 : self.dim_x], time() - t0

    def make_samples_mult_y(
        self,
        ys: np.array,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y. y is (m x 7)
        """
        assert ys.shape[1] == self.dim_y
        assert self.robot_model.dim_y == 7
        assert latent_noise_distribution in ["gaussian", "uniform"]

        ys = torch.FloatTensor(ys)
        m = ys.shape[0]

        # Note: No code change required here to handle using/not using softflow.
        conditional = torch.zeros(m, self.dim_cond)
        conditional[:, 0:7] = ys
        conditional = conditional.to(config.device)
        shape = (m, self.dim_tot)
        latent_noise = draw_latent_noise(latent_noise, latent_noise_distribution, latent_noise_scale, shape)
        assert latent_noise.shape[0] == m
        assert latent_noise.shape[1] == self.dim_tot
        t0 = time()
        output_rev, jac = self.nn_model(latent_noise, c=conditional, rev=True)
        return output_rev[:, 0 : self.dim_x], time() - t0


class ModelWrapper_MultiCINN(ModelWrapper):
    """A ModelWrapper subclass for multiple conditional invertible neural network with softflow. That's a mouthful"""

    nn_model_type = "multi_cinn"

    def __init__(self, hyper_parameters: MultiCINN_Parameters, robot_model: robot_models.RobotModel):
        """Initialize a ModelWrapper_MultiCINN"""
        assert isinstance(hyper_parameters, MultiCINN_Parameters)

        # Save specific architecure parameters
        if hyper_parameters.softflow_enabled:
            self.base_dim_cond = robot_model.dim_y + 1
        else:
            self.base_dim_cond = robot_model.dim_y
        self.subnet_network_width = hyper_parameters.subnet_network_width

        # TODO(@jeremysm): Add `dim_tot` as a member value to ModelWrapper
        # self.dim_tot = robot_model.dim_x
        cuda_out_of_memory = False
        nn_model = None
        try:
            nn_model = multi_cinn_model(hyper_parameters, robot_model)
        except RuntimeError as e:
            print(f"ModelWrapper_cINN_Softflow() caught error '{e}' in __init__() function.")
            if "out of memory" in str(e) and "CUDA" in str(e):
                cuda_out_of_memory = True

        ModelWrapper.__init__(
            self, ModelWrapper_MultiCINN.nn_model_type, nn_model, robot_model, cuda_out_of_memory=cuda_out_of_memory
        )

    def make_samples(
        self,
        y: List[float],
        m: int,
        latent_noise: Optional[torch.Tensor] = None,
        latent_noise_distribution: str = "gaussian",
        latent_noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y
                y: endpose [x, y, z, q0, q1, q2, q3]
                m: Number of samples
        """
        assert len(y) == self.dim_y

        t0 = time()
        output = torch.zeros(m, self.dim_x)
        latent_noise_shape = (m, self.dim_tot)
        # output = torch.zeros(m, self.dim_tot)

        for i in range(self.dim_x):
            # for i in range(self.dim_tot):

            # Note: No code change required here to handle using/not using softflow.
            conditional = torch.zeros(m, self.base_dim_cond + i)

            if self.robot_model.dim_y == 7:

                # (:, 0:3) is x, y, z
                conditional[:, 0:3] = torch.FloatTensor(y[:3])

                # (:, 3:7) is quat
                conditional[:, 3 : 3 + 4] = torch.FloatTensor([y[3:]])
            else:
                conditional[:, 0:2] = torch.FloatTensor(y)

            if i > 0:
                # conditional[:, 7:7 + i] = output[:, 0:i]
                conditional[:, self.base_dim_cond : self.base_dim_cond + i] = output[:, 0:i]

            # (:, 7: 7 + ndim_tot) is the sigma of the normal distribution that noise is drawn from and added to x. 0 in this case
            conditional = conditional.to(config.device)

            latent_noise = draw_latent_noise(
                latent_noise, latent_noise_distribution, latent_noise_scale, latent_noise_shape
            )
            assert latent_noise.shape[0] == m
            assert latent_noise.shape[1] == self.dim_tot

            output_rev, jac = self.nn_model[i](latent_noise, c=conditional, rev=True)
            output[:, i] = output_rev[:, 0]
        return output, time() - t0


class ModelWrapper_MDN(ModelWrapper):
    """A ModelWrapper subclass for a"""

    nn_model_type = "mdn"

    def __init__(self, hyper_parameters: MDNParameters, robot_model: robot_models.RobotModel):
        """Initialize a ModelWrapper_MDN object"""
        assert isinstance(hyper_parameters, MDNParameters)
        nn_model = mdn_model(hyper_parameters.n_components, robot_model)
        ModelWrapper.__init__(self, ModelWrapper_MDN.nn_model_type, nn_model, robot_model)

    def make_samples(
        self, y: List[float], m: int, rev_inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y
                y: endpose [x, y, z, q0, q1, q2, q3]
                m: Number of samples
        """
        assert len(y) == self.dim_y
        assert rev_inputs is None, "`rev_inputs` not implemented for MDN"
        t0 = time()
        conditional = torch.zeros(m, self.dim_y)
        return self.nn_model.sample(conditional)[:, 0 : self.dim_x], time() - t0


class ModelWrapper_IResNet(ModelWrapper):
    """A ModelWrapper subclass for an invertible res. net"""

    nn_model_type = "iresnet"

    def __init__(self, hyper_parameters: IResNetParameters, robot_model: robot_models.RobotModel):
        """Initialize a ModelWrapper_IResNet object"""
        assert isinstance(hyper_parameters, IResNetParameters)
        self.dim_tot = robot_model.dim_x
        self.dim_cond = robot_model.dim_y
        nn_model = iresnet_model(hyper_parameters, self.dim_cond, self.dim_tot, robot_model)
        ModelWrapper.__init__(self, ModelWrapper_IResNet.nn_model_type, nn_model, robot_model)

    def make_samples(
        self, y: List[float], m: int, rev_inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the network in reverse to generate samples conditioned on a pose y
                y: endpose [x, y, z, q0, q1, q2, q3]
                m: Number of samples
        """
        assert len(y) == self.dim_y
        assert rev_inputs is None, "`rev_inputs` not implemented for iresnet"
        conditional = torch.zeros(m, self.dim_cond)
        rev_inputs = torch.randn(m, self.dim_tot).to(config.device)

        t0 = time()
        output_rev, jac = self.nn_model(rev_inputs, c=conditional, rev=True, jac=False)
        return output_rev[:, 0 : self.dim_x], time() - t0
