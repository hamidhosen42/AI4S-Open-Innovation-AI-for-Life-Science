"""Evaluate trained runs on the held-out test well.

    python scripts/evaluate.py --runs runs/baseline_unet_s0 runs/chipstain_nll_s0 [--tta] --out outputs/eval

Writes per-image metrics (CSV), a summary table (CSV + Markdown) and
per-run predictions for figure making.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from chipstain.data import PairDataset, load_mask, make_splits, pad_to_multiple
from chipstain.metrics import calibration_metrics, image_metrics, match_f1, segment_nuclei
from chipstain.model import ChipStainNet, predict_tta


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_run(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    cfg = ck["config"]
    model = ChipStainNet(cfg["encoder"], False, cfg["uncertainty"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict(model, x, tta):
    if tta:
        mu, ale, epi = predict_tta(model, x)
        unc = (ale + epi) if ale is not None else epi
    else:
        mu, logvar = model(x)
        unc = logvar.exp() if logvar is not None else None
    return mu, unc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--save_preds", type=int, default=8, help="save first N predictions per run as npz")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    _, val, test = make_splits(a.data)
    samples = test if a.split == "test" else val
    ds = PairDataset(samples, crop=None, augment=False, cache=False)

    # GT segmentation of the *real* fluorescence through the same pipeline = upper bound
    rows = []
    for run in a.runs:
        model, cfg = load_run(run, device)
        tag = os.path.basename(run.rstrip("/")) + ("_tta" if a.tta else "")
        preds = []
        for i in tqdm(range(len(ds)), desc=tag):
            x, y = ds[i]
            mu, unc = predict(model, x[None].to(device), a.tta)
            h, w = 540, 540
            p = mu[0, 0, :h, :w].cpu().numpy()
            g = y[0, :h, :w].numpy()
            u = unc[0, 0, :h, :w].cpu().numpy() if unc is not None else None
            m = image_metrics(p, g)
            # downstream nuclei segmentation vs StarDist reference masks
            ref = load_mask(samples[i])
            seg_pred = segment_nuclei(np.clip(p, 0, 1))
            seg_gt = segment_nuclei(g)
            f_pred = match_f1(seg_pred, ref)
            f_gt = match_f1(seg_gt, ref)
            m.update({"seg_f1": f_pred["f1"], "seg_f1_realfluo": f_gt["f1"], "n_ref": f_pred["n_gt"], "n_pred": f_pred["n_pred"]})
            if u is not None:
                m.update(calibration_metrics(p, g, np.sqrt(u)))
                m["mean_sigma"] = float(np.sqrt(u).mean())
            m.update({"run": tag, "well": samples[i].well, "field": samples[i].field, "timepoint": samples[i].timepoint})
            rows.append(m)
            if i < a.save_preds:
                preds.append({"bf": x[0, :h, :w].numpy(), "gt": g, "pred": p, "unc": None if u is None else np.sqrt(u), "ref": ref, "seg_pred": seg_pred})
        np.savez_compressed(os.path.join(a.out, f"preds_{tag}.npz"), **{f"{k}_{j}": v for j, d in enumerate(preds) for k, v in d.items() if v is not None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"per_image_{a.split}.csv"), index=False)
    cols = [c for c in ["mae", "psnr", "ssim", "pearson", "seg_f1", "seg_f1_realfluo", "spearman_unc_err", "ause", "mean_sigma"] if c in df]
    summ = df.groupby("run")[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(a.out, f"summary_{a.split}.csv"))
    # markdown table
    md = ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for run, r in summ.iterrows():
        md.append(f"| {run} | " + " | ".join(f"{r[(c,'mean')]:.3f} ± {r[(c,'std')]:.3f}" for c in cols) + " |")
    open(os.path.join(a.out, f"summary_{a.split}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    # image-level: does mean sigma predict image-level error?
    if "mean_sigma" in df:
        from scipy.stats import spearmanr
        for run, d in df.groupby("run"):
            if d["mean_sigma"].notna().any():
                r = spearmanr(d["mean_sigma"], d["mae"]).statistic
                print(f"[{run}] image-level Spearman(mean sigma, MAE) = {r:.3f}")


if __name__ == "__main__":
    main()
