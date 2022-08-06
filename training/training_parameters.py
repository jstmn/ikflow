from typing import Dict
from utils.supporting_types import Config


class TrainingConfig(Config):
    """Class stores the configuration for a training run"""

    def __init__(self):
        self.test_description = "not-set"
        self.n_epochs = 150  # total number of epochs
        self.batch_size = 64
        self.use_small_dataset = False
        self.optimizer = "ranger"  # adam
        self.learning_rate = 1e-3

        # Step the learning rate every k batches
        self.step_lr_every_k = int(int(2.5 * 1e6) / 64)

        self.gamma = 0.9794578299341784  # For weight scheduler
        self.l2_reg = 1.855446930271331e-05  # for optimizer

        self.gradient_clamp = 5.0
        Config.__init__(self, "TrainingConfig")


class IkflowModelParameters(Config):
    def __init__(self):
        self.coupling_layer = "glow"
        self.nb_nodes = 12
        self.dim_latent_space = 9
        self.coeff_fn_config = 3
        self.coeff_fn_internal_size = 1024
        self.permute_random_enabled = True

        # ___ Loss parameters
        self.lambd_predict = 1.0  # Fit Loss lambda
        self.init_scale = 0.04473500291638653
        self.rnvp_clamp = 2.5
        self.y_noise_scale = 1e-7  # Add some noise to the cartesian poses
        self.zeros_noise_scale = 1e-3  # Also add noise to the padding

        self.softflow_noise_scale = 0.01
        self.softflow_enabled = True

        Config.__init__(self, "IkflowModelParameters")


class MultiCINN_Parameters(Config):
    def __init__(self):
        self.coupling_layer = "glow"
        self.subnet_nb_nodes = 5
        self.subnet_network_width = 10
        self.subnet_coeff_fn_depth = 3
        self.subnet_coeff_fn_width = 512

        # Only the first column of a `sample` tensor will have joint values. Fill the rest with a sample from a normal
        # gaussian multiplied by `non_joint_sample_noise`
        self.non_joint_sample_noise = 1e-3

        # ___ Loss parameters
        self.lambd_predict = 1.0  # Fit Loss lambda

        # Initialization scale
        self.init_scale = -1
        self.rnvp_clamp = 2.5

        self.softflow_noise_scale = -1
        self.softflow_enabled = False

        Config.__init__(self, "CompositeNetParameters")


class IResNetParameters(Config):
    def __init__(self):
        self.nb_nodes = 6
        self.n_internal_layers = 3
        self.coeff_fn_internal_size = 256
        self.jacobian_iterations = 20
        self.hutchinson_samples = 1
        self.fixed_point_iterations = 50
        self.lipschitz_iterations = 10
        self.lipschitz_batchsize = 10
        self.spectral_norm_max = 0.8
        self.iresnet_loss_fn = 1

        Config.__init__(self, "IResNetParameters")


class MDNParameters(Config):
    def __init__(self):
        self.n_components = 1

        Config.__init__(self, "MDNParameters")
