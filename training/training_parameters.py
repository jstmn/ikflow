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


class MDNParameters(Config):
    def __init__(self):
        self.n_components = 1
        Config.__init__(self, "MDNParameters")
