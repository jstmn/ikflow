import torch


def get_softflow_noise(x: torch.Tensor, softflow_noise_scale: float):
    """
    Return noise and noise magnitude for softflow. See https://arxiv.org/abs/2006.04604
    Return:
        c (torch.Tensor): An (batch_sz x 1) array storing the magnitude of the sampled gausian `eps` along each row
        eps (torch.Tensor): a (batch_sz x dim_x) array, where v[i] is a sample drawn from a zero mean gaussian with variance = c[i]**2
    """
    dim_x = x.shape[1]

    # (batch_size x 1) from uniform(0, 1)
    c = torch.rand_like(x[:, 0]).unsqueeze(1)

    # (batch_size x dim_y) *  (batch_size x dim_y)  | element wise multiplication
    eps = torch.randn_like(x) * torch.cat(dim_x * [c], dim=1) * softflow_noise_scale
    return c, eps
