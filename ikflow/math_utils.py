import torch
import numpy as np

from ikflow import config


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
