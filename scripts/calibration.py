"""Is sigma calibrated? Empirical coverage of mu ± k*sigma, a reliability curve, and a single
variance-scaling factor fitted on the VALIDATION split and applied to test.

    python scripts/calibration.py --cache outputs/cache --cache_val outputs/cache_val --out report/results
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm

from chipstain.data import PairDataset, load_mask, make_splits

LEVELS = [0.5, 0.683, 0.8, 0.9, 0.95]


def targets(samples):
    ds = PairDataset(samples, crop=None, augment=False, cache=False)
    return np.stack([ds[i][1][0, :540, :540].numpy() for i in range(len(ds))])


def coverage(mu, sig, y, scale=1.0, stride=3, mask=None):
    """Fraction of pixels with |y - mu| <= z * scale * sigma for each nominal level; mask restricts to
    a subset of pixels (e.g. inside the reference nuclei)."""
    z = np.abs(y[:, ::stride, ::stride] - mu[:, ::stride, ::stride]) / np.maximum(sig[:, ::stride, ::stride] * scale, 1e-6)
    if mask is not None:
        z = z[mask[:, ::stride, ::stride]]
    return {lvl: float((z <= norm.ppf(0.5 + lvl / 2)).mean()) for lvl in LEVELS}


def fit_scale(mu, sig, y, stride=3):
    """Scalar s minimising the Gaussian NLL of y under N(mu, (s*sigma)^2): s^2 = mean((y-mu)^2/sigma^2)."""
    r2 = ((y - mu)[:, ::stride, ::stride] / np.maximum(sig[:, ::stride, ::stride], 1e-6)) ** 2
    return float(np.sqrt(np.mean(r2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--cache_val", default="outputs/cache_val")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    _, val, test = make_splits("data/raw/hela_kyoto")
    y_test, y_val = targets(test), targets(val)
    fg = np.stack([load_mask(s) > 0 for s in test])  # pixels inside reference nuclei
    rows = []
    for tag_base in ["chipstain_nll", "pretrained_l1", "baseline_unet"]:  # ChipStain first in the table
        for tta in ["", "_tta"]:
            if tag_base in ("baseline_unet", "pretrained_l1") and tta == "":
                continue  # a point estimate has no sigma
            for s in a.seeds:
                tag = f"{tag_base}_s{s}{tta}"
                ft = os.path.join(a.cache, f"{tag}.npz")
                if not os.path.exists(ft):
                    continue
                z = np.load(ft)
                mu, sig = z["mu"].astype(np.float32), z["sigma"].astype(np.float32)
                fv = os.path.join(a.cache_val, f"{tag}.npz")
                scale = np.nan
                if os.path.exists(fv):
                    zv = np.load(fv)
                    scale = fit_scale(zv["mu"].astype(np.float32), zv["sigma"].astype(np.float32), y_val)
                raw = coverage(mu, sig, y_test)
                cal = coverage(mu, sig, y_test, scale) if np.isfinite(scale) else {k: np.nan for k in LEVELS}
                fgc = coverage(mu, sig, y_test, mask=fg)
                rows.append({"model": tag_base + tta, "seed": s, "val_scale": scale,
                             **{f"raw_{k}": v for k, v in raw.items()}, **{f"cal_{k}": v for k, v in cal.items()}, **{f"fg_{k}": v for k, v in fgc.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "calibration.csv"), index=False)
    order = ["chipstain_nll_tta", "chipstain_nll", "pretrained_l1_tta", "baseline_unet_tta"]
    g = df.groupby("model").mean(numeric_only=True).reindex([o for o in order if o in set(df.model)])
    md = ["Empirical coverage of μ ± z·σ on the 125 test images (every 3rd pixel in each direction, i.e. 1 in 9 pixels), mean over seeds. "
          "'recalibrated' multiplies σ by one scalar fitted on the validation split.", "",
          "| model | σ-scale (val) | " + " | ".join(f"nominal {int(l*100) if l != 0.683 else 68.3} %" for l in LEVELS) + " |",
          "|---|---|" + "---|" * len(LEVELS)]
    pretty = {"baseline_unet_tta": "Scratch U-Net + TTA (view s.d.)", "pretrained_l1_tta": "ImageNet-L1 U-Net + TTA (view s.d.)",
              "chipstain_nll": "ChipStain, single pass", "chipstain_nll_tta": "ChipStain + TTA"}
    for m, r in g.iterrows():
        md.append(f"| {pretty.get(m, m)}, raw | — | " + " | ".join(f"{100*r[f'raw_{l}']:.1f} %" for l in LEVELS) + " |")
        if np.isfinite(r["val_scale"]):
            md.append(f"| {pretty.get(m, m)}, recalibrated | ×{r['val_scale']:.2f} | " + " | ".join(f"{100*r[f'cal_{l}']:.1f} %" for l in LEVELS) + " |")
        md.append(f"| {pretty.get(m, m)}, raw, nuclei pixels only | — | " + " | ".join(f"{100*r[f'fg_{l}']:.1f} %" for l in LEVELS) + " |")
    open(os.path.join(a.out, "calibration.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
