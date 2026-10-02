"""Distribution-shift stress test: does sigma rise when the input drifts away from training?

    python scripts/shift_test.py --runs runs/chipstain_nll_s0 runs/baseline_unet_s0 --out report/results

Simulated shifts that occur when moving to organ-on-a-chip imaging (thick PDMS, curved
channels, different optics): defocus blur, sensor noise, reduced contrast, and a modality
swap (digital phase contrast instead of bright-field, same fields). Uses the 50 test images
at time-points 10 and 100. For each model and condition we report mean sigma and MAE /
Pearson against the real H2B image; the question is whether sigma tracks the error.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # allow `from scripts...`

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from tqdm import tqdm

from chipstain.data import load_pair, make_splits, normalize_input, normalize_target, pad_to_multiple
from chipstain.metrics import image_metrics
from scripts.evaluate import get_device, load_run, predict

CONDITIONS = [
    ("clean", None),
    ("defocus σ=1", ("blur", 1.0)), ("defocus σ=2", ("blur", 2.0)), ("defocus σ=4", ("blur", 4.0)),
    ("noise 5 %", ("noise", 0.05)), ("noise 10 %", ("noise", 0.10)), ("noise 20 %", ("noise", 0.20)),
    ("contrast ×0.5", ("contrast", 0.5)), ("contrast ×0.25", ("contrast", 0.25)),
    ("modality: DPC", ("dpc", None)),
]


def corrupt(bf, dpc, kind, rng):
    if kind is None:
        return bf
    k, v = kind
    if k == "blur":
        return ndi.gaussian_filter(bf, v)
    if k == "noise":
        lo, hi = np.percentile(bf, [1, 99])
        return bf + rng.normal(0, v * (hi - lo), bf.shape)
    if k == "contrast":
        m = bf.mean()
        return m + (bf - m) * v + rng.normal(0, 0.01 * (np.percentile(bf, 99) - np.percentile(bf, 1)), bf.shape)
    if k == "dpc":
        return dpc
    raise ValueError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/chipstain_nll_s0", "runs/baseline_unet_s0"])
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    device = get_device()
    _, _, test = make_splits(a.data)
    samples = [s for s in test if s.timepoint in (10, 100)]
    rows = []
    for run in a.runs:
        model, cfg = load_run(run, device)
        tag = os.path.basename(run.rstrip("/"))
        for i, s in enumerate(tqdm(samples, desc=tag)):
            bf, fl = load_pair(s)
            dpc = tifffile.imread(s.base + "_dpc.tif").astype(np.float32)
            y = normalize_target(fl)
            for name, kind in CONDITIONS:
                rng = np.random.default_rng(1000 * i + len(name))
                x = normalize_input(corrupt(bf.astype(np.float32), dpc, kind, rng))
                xp, _ = pad_to_multiple(x)
                mu, var = predict(model, torch.from_numpy(xp)[None, None].to(device), tta=True)
                p = mu[0, 0, :540, :540].cpu().numpy()
                sig = np.sqrt(var[0, 0, :540, :540].cpu().numpy())
                m = image_metrics(p, y)
                rows.append({"run": tag, "condition": name, "image": i, "mean_sigma": float(sig.mean()), **m})
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "shift_test.csv"), index=False)
    order = [c for c, _ in CONDITIONS]
    lines = []
    for run, g in df.groupby("run"):
        t = g.groupby("condition")[["mean_sigma", "mae", "pearson"]].mean().loc[order]
        base = t.loc["clean", "mean_sigma"]
        rho = spearmanr(g["mean_sigma"], g["mae"]).statistic
        lines += [f"### {run}", f"Spearman ρ(mean σ, MAE) over all {len(g)} (image, condition) pairs: {rho:.3f}", "",
                  "| condition | mean σ | σ / clean | MAE | Pearson r |", "|---|---|---|---|---|"]
        for c, r in t.iterrows():
            lines.append(f"| {c} | {r.mean_sigma:.4f} | {r.mean_sigma / base:.2f}× | {r.mae:.4f} | {r.pearson:.3f} |")
        lines.append("")
    open(os.path.join(a.out, "shift_test.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
