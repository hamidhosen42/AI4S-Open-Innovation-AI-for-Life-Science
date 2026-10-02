"""Predict nuclear fluorescence + uncertainty for a single bright-field image.

    python scripts/inference.py --image path/to/brightfield.tif --weights weights/chipstain.pt --out outputs/pred
Accepts .tif/.png/.jpg (single-channel or RGB -> converted to grey). Writes
<out>/prediction.tif (float32, normalised 0-1), <out>/uncertainty.tif (sigma) and a
side-by-side <out>/panel.png.
"""
import argparse
import os

import numpy as np
import tifffile
import torch
from skimage import io

from chipstain.data import normalize_input, pad_to_multiple
from chipstain.model import ChipStainNet, predict_tta


def read_grey(path):
    """Read a bright-field image as a 2-D float32 array. Accepts grey, grey+alpha, RGB/RGBA
    (averaged to grey) and channel-first stacks (first channel)."""
    if str(path).lower().endswith((".tif", ".tiff")):
        a = tifffile.imread(path)
    else:
        a = io.imread(path)
    a = np.squeeze(a)
    if a.ndim == 3:
        if a.shape[-1] in (3, 4):          # RGB / RGBA, channel-last
            a = a[..., :3].mean(-1)
        elif a.shape[-1] == 2:             # grey + alpha
            a = a[..., 0]
        elif a.shape[0] <= 4:              # channel-first stack
            a = a[0]
        else:
            raise ValueError(f"cannot interpret image of shape {a.shape} as a single bright-field plane")
    if a.ndim != 2 or min(a.shape) < 32:
        raise ValueError(f"expected a 2-D bright-field image of at least 32x32 px, got shape {a.shape}")
    return a.astype(np.float32)


def load_model(weights, device):
    ck = torch.load(weights, map_location="cpu")
    cfg = ck["config"]
    m = ChipStainNet(cfg["encoder"], False, cfg["uncertainty"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def run(model, bf, device, tta=True):
    x = normalize_input(bf)
    h, w = x.shape
    xp, _ = pad_to_multiple(x)
    xt = torch.from_numpy(xp)[None, None].to(device)
    if tta:
        mu, ale, epi = predict_tta(model, xt)
        var = (ale if ale is not None else 0) + epi
    else:
        mu, logvar = model(xt)
        var = logvar.exp() if logvar is not None else torch.zeros_like(mu)
    pred = np.clip(mu[0, 0, :h, :w].cpu().numpy(), 0, 1)
    sigma = np.sqrt(var[0, 0, :h, :w].cpu().numpy())
    return pred, sigma


def panel(bf, pred, sigma, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    ax[0].imshow(bf, cmap="gray"); ax[0].set_title("Bright-field input")
    ax[1].imshow(pred, cmap="magma", vmin=0, vmax=1); ax[1].set_title("Predicted nuclei (H2B)")
    im = ax[2].imshow(sigma, cmap="viridis"); ax[2].set_title("Uncertainty (sigma)")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="weights/chipstain.pt")
    ap.add_argument("--out", default="outputs/pred")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    model, _ = load_model(a.weights, device)
    bf = read_grey(a.image)
    pred, sigma = run(model, bf, device, tta=not a.no_tta)
    os.makedirs(a.out, exist_ok=True)
    tifffile.imwrite(os.path.join(a.out, "prediction.tif"), pred.astype(np.float32))
    tifffile.imwrite(os.path.join(a.out, "uncertainty.tif"), sigma.astype(np.float32))
    panel(bf, pred, sigma, os.path.join(a.out, "panel.png"))
    print(f"wrote {a.out}/prediction.tif, uncertainty.tif, panel.png  | mean sigma={sigma.mean():.4f}")


if __name__ == "__main__":
    main()
