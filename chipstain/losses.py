import torch
import torch.nn.functional as F


def l1_loss(mu, logvar, y):
    return F.l1_loss(mu, y)


def gaussian_nll(mu, logvar, y, beta: float = 0.5):
    """beta-NLL (Seitzer et al., 2022): NLL with per-pixel weight sigma^(2*beta)
    detached, which prevents the variance head from swallowing the signal early."""
    nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp())
    if beta > 0:
        nll = nll * (logvar.exp().detach() ** beta)
    return nll.mean()


LOSSES = {"l1": l1_loss, "nll": gaussian_nll}
