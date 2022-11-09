class IkflowModelParameters:
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


class MDNParameters:
    def __init__(self):
        self.n_components = 1
