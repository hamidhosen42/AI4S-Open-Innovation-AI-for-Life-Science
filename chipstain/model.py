"""U-Net with optional heteroscedastic (mean + log-variance) output head."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ChipStainNet(nn.Module):
    """U-Net (segmentation_models_pytorch). If ``uncertainty`` the network predicts
    2 channels: mean mu and log-variance s = log sigma^2 of a per-pixel Gaussian."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True, uncertainty: bool = True):
        super().__init__()
        self.uncertainty = uncertainty
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,
            classes=2 if uncertainty else 1,
            activation=None,
        )

    def forward(self, x):
        out = self.net(x)
        if self.uncertainty:
            mu, logvar = out[:, :1], out[:, 1:2]
            logvar = logvar.clamp(-10.0, 4.0)
            return mu, logvar
        return out, None


@torch.no_grad()
def predict_tta(model: ChipStainNet, x: torch.Tensor):
    """8-fold dihedral test-time augmentation.

    Returns (mu, var_aleatoric, var_epistemic): the mean prediction, the mean
    predicted variance, and the variance of the 8 predictions (a cheap
    epistemic proxy). For a non-uncertainty model var_aleatoric is None.
    """
    mus, vars_ = [], []
    for k in range(4):
        for flip in (False, True):
            xi = torch.rot90(x, k, dims=(-2, -1))
            if flip:
                xi = torch.flip(xi, dims=(-1,))
            mu, logvar = model(xi)
            if flip:
                mu = torch.flip(mu, dims=(-1,))
                logvar = torch.flip(logvar, dims=(-1,)) if logvar is not None else None
            mu = torch.rot90(mu, -k, dims=(-2, -1))
            mus.append(mu)
            if logvar is not None:
                vars_.append(torch.rot90(logvar, -k, dims=(-2, -1)).exp())
    mus = torch.stack(mus)
    mu = mus.mean(0)
    epi = mus.var(0, unbiased=False)
    ale = torch.stack(vars_).mean(0) if vars_ else None
    return mu, ale, epi
