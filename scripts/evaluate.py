"""Evaluate trained runs on the held-out test well.

    python scripts/evaluate.py --runs runs/baseline_unet_s0 runs/chipstain_nll_s0 [--tta] --out outputs/eval [--cache outputs/cache]

Per image: fidelity (MAE, PSNR, SSIM, Pearson), downstream nuclei segmentation F1 against the
StarDist reference labels, and - for every pixel-ranking score available - Spearman rho with
|error|, AUSE and the MAE reduction after discarding the 20 % highest-score pixels. Scores:
the model's sigma (if any) plus two uncertainty-free proxies computed from the prediction
itself (intensity "mu" and edge strength "grad"), and the oracle gain for reference.
Everything is computed on all test images. With --cache, predictions (mu, sigma) are stored
as float16 so ensembles and re-scoring do not need new inference.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from chipstain.data import PairDataset, load_mask, make_splits
from chipstain.metrics import calibration_metrics, image_metrics, match_f1, proxy_scores, removal_gain, segment_nuclei
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


def score_image(p, g, sigma, ref):
    """All per-image metrics for prediction p, target g (both normalised), sigma (or None)
    and the reference nuclei label image."""
    m = image_metrics(p, g)
    seg_pred = segment_nuclei(np.clip(p, 0, 1))
    f_pred = match_f1(seg_pred, ref)
    f_gt = match_f1(segment_nuclei(g), ref)
    m.update({"seg_f1": f_pred["f1"], "seg_precision": f_pred["precision"], "seg_recall": f_pred["recall"],
              "seg_f1_realfluo": f_gt["f1"], "n_ref": f_pred["n_gt"], "n_pred": f_pred["n_pred"], "n_realfluo": f_gt["n_pred"]})
    err = np.abs(p - g)
    if sigma is not None:
        m.update(calibration_metrics(p, g, sigma))
        m["mean_sigma"] = float(sigma.mean())
    for name, sc in proxy_scores(p).items():
        c = calibration_metrics(p, g, sc)
        m.update({f"spearman_{name}_err": c["spearman_unc_err"], f"ause_{name}": c["ause"], f"gain20_{name}": c["gain20"]})
    m["gain20_oracle"] = removal_gain(err, err)
    return m, seg_pred


def summarise(df, out, split):
    cols = [c for c in ["mae", "psnr", "ssim", "pearson", "seg_f1", "seg_f1_realfluo", "spearman_unc_err", "ause", "gain20",
                        "spearman_mu_err", "ause_mu", "gain20_mu", "ause_grad", "gain20_grad", "gain20_oracle", "mean_sigma"] if c in df]
    summ = df.groupby("run")[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(out, f"summary_{split}.csv"))
    md = ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for run, r in summ.iterrows():
        md.append(f"| {run} | " + " | ".join("—" if pd.isna(r[(c, 'mean')]) else f"{r[(c,'mean')]:.3f} ± {r[(c,'std')]:.3f}" for c in cols) + " |")
    open(os.path.join(out, f"summary_{split}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if "mean_sigma" in df:
        for run, d in df.groupby("run"):
            if d["mean_sigma"].notna().any():
                print(f"[{run}] image-level Spearman(mean sigma, MAE) = {spearmanr(d['mean_sigma'], d['mae']).statistic:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--cache", default=None, help="directory to store float16 predictions per run")
    ap.add_argument("--save_preds", type=int, default=8, help="save the first N full-precision predictions per run (figures only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    _, val, test = make_splits(a.data)
    samples = test if a.split == "test" else val
    ds = PairDataset(samples, crop=None, augment=False, cache=False)
    rows = []
    for run in a.runs:
        model, cfg = load_run(run, device)
        tag = os.path.basename(run.rstrip("/")) + ("_tta" if a.tta else "")
        preds, mus, sigs = [], [], []
        for i in tqdm(range(len(ds)), desc=tag):
            x, y = ds[i]
            mu, unc = predict(model, x[None].to(device), a.tta)
            h, w = 540, 540
            p = mu[0, 0, :h, :w].cpu().numpy()
            g = y[0, :h, :w].numpy()
            u = np.sqrt(unc[0, 0, :h, :w].cpu().numpy()) if unc is not None else None
            ref = load_mask(samples[i])
            m, seg_pred = score_image(p, g, u, ref)
            m.update({"run": tag, "well": samples[i].well, "field": samples[i].field, "timepoint": samples[i].timepoint})
            rows.append(m)
            if a.cache:
                mus.append(p.astype(np.float16))
                sigs.append(u.astype(np.float16) if u is not None else np.zeros((h, w), np.float16))
            if i < a.save_preds:
                preds.append({"bf": x[0, :h, :w].numpy(), "gt": g, "pred": p, "unc": u, "ref": ref, "seg_pred": seg_pred})
        if a.cache:
            os.makedirs(a.cache, exist_ok=True)
            np.savez(os.path.join(a.cache, f"{tag}.npz"), mu=np.stack(mus), sigma=np.stack(sigs), has_sigma=np.array(cfg["uncertainty"] or a.tta))
        if preds:
            np.savez_compressed(os.path.join(a.out, f"preds_{tag}.npz"), **{f"{k}_{j}": v for j, d in enumerate(preds) for k, v in d.items() if v is not None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"per_image_{a.split}.csv"), index=False)
    summarise(df, a.out, a.split)


if __name__ == "__main__":
    main()
