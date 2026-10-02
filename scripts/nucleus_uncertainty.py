"""Per-nucleus uncertainty: which score best flags wrong detections?

    python scripts/nucleus_uncertainty.py --cache outputs/cache --out report/results

Uses cached predictions (scripts/evaluate.py --cache). For every model/seed, nuclei are
segmented on the prediction; each predicted nucleus is "correct" if it matches a StarDist
reference nucleus at IoU >= 0.5. Candidate scores per nucleus:
  sigma_learned   mean ChipStain sigma inside the nucleus (learned aleatoric + TTA term)
  sigma_tta_unet  mean TTA view-standard-deviation of the plain U-Net
  sigma_ensemble  mean std of the 3-seed ensemble (variance of member means + member sigma^2)
  dim             negative mean predicted intensity (dim objects are often debris/false positives)
  small           negative object area
Reported: AUROC for flagging unmatched detections (per seed, mean ± sd), and the precision /
recall / F1 "quality gate" curve when the highest-score detections are discarded.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from chipstain.data import load_mask, make_splits
from chipstain.metrics import segment_nuclei


def roc_auc(positive, score):
    """AUROC via the Mann-Whitney U statistic (no scikit-learn dependency)."""
    positive = np.asarray(positive, bool)
    score = np.asarray(score, float)
    u = mannwhitneyu(score[positive], score[~positive], alternative="two-sided").statistic
    return float(u / (positive.sum() * (~positive).sum()))


def object_matches(pred_lab, gt_lab, thr=0.5):
    """Boolean per predicted object (ids 1..max): matched to a reference object at IoU >= thr.
    At IoU > 0.5 a match is unique; ties at exactly 0.5 are rejected, as in chipstain.metrics.match_f1."""
    n_p, n_g = int(pred_lab.max()), int(gt_lab.max())
    if n_p == 0:
        return np.zeros(0, bool)
    both = (pred_lab > 0) & (gt_lab > 0)
    inter = np.zeros((n_p + 1, n_g + 1))
    np.add.at(inter, (pred_lab[both], gt_lab[both]), 1)
    ap = np.bincount(pred_lab.ravel(), minlength=n_p + 1)[:, None]
    ag = np.bincount(gt_lab.ravel(), minlength=n_g + 1)[None, :]
    iou = inter / (ap + ag - inter + 1e-9)  # same epsilon as chipstain.metrics.match_f1: exact 0.5 ties are rejected by both
    iou[0, :] = 0
    iou[:, 0] = 0
    return iou.max(1)[1:] >= thr


def nuclei_table(mu, sigmas, refs, tag):
    """One row per predicted nucleus with every available score."""
    rows = []
    for i in range(mu.shape[0]):
        p = np.clip(mu[i].astype(np.float32), 0, 1)
        lab = segment_nuclei(p)
        if lab.max() == 0:
            continue
        ok = object_matches(lab, refs[i])
        idx = np.arange(1, lab.max() + 1)
        mu_obj = ndi.mean(p, lab, idx)
        d = {"run": tag, "image": i, "matched": ok, "dim": -mu_obj, "small": -ndi.sum(np.ones_like(p), lab, idx)}
        for name, s in sigmas.items():
            sg = ndi.mean(s[i].astype(np.float32), lab, idx)
            d[name] = sg
            d[name + "_rel"] = sg / np.maximum(mu_obj, 1e-3)               # relative uncertainty sigma / mu
            d[name + "_imgnorm"] = sg / max(np.median(sg), 1e-6)           # sigma relative to the image's own nuclei
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def within_image_auroc(g, score):
    vals = [roc_auc(~h["matched"].values, h[score].values) for _, h in g.groupby("image") if 0 < h["matched"].sum() < len(h)]
    return float(np.mean(vals)) if vals else np.nan


def gate_curve(df, score, n_ref, fracs=(0, 0.05, 0.1, 0.2, 0.3)):
    d = df.sort_values(score).reset_index(drop=True)  # low score = keep first
    out = []
    for f in fracs:
        k = int(round(len(d) * (1 - f)))
        tp = d["matched"].iloc[:k].sum()
        p, r = tp / k, tp / n_ref
        out.append({"discard": f, "precision": p, "recall": r, "f1": 2 * p * r / (p + r)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    _, _, test = make_splits(a.data)
    refs = [load_mask(s) for s in test]
    n_ref = int(sum(len(np.unique(r)) - 1 for r in refs))
    load = lambda t: np.load(os.path.join(a.cache, f"{t}.npz"))  # noqa: E731
    tables = []
    for s in tqdm(a.seeds, desc="seeds"):
        cs, un = load(f"chipstain_nll_s{s}_tta"), load(f"baseline_unet_s{s}_tta")
        tables.append(nuclei_table(cs["mu"], {"sigma_learned": cs["sigma"]}, refs, f"ChipStain + TTA s{s}"))
        tables.append(nuclei_table(un["mu"], {"sigma_tta_unet": un["sigma"]}, refs, f"U-Net + TTA s{s}"))
    # 3-seed ChipStain ensemble
    zs = [load(f"chipstain_nll_s{s}_tta") for s in a.seeds]
    mus = np.stack([z["mu"].astype(np.float32) for z in zs])
    ens_mu = mus.mean(0)
    ens_sig = np.sqrt(mus.var(0) + np.mean([z["sigma"].astype(np.float32) ** 2 for z in zs], axis=0))
    tables.append(nuclei_table(ens_mu, {"sigma_ensemble": ens_sig}, refs, "ChipStain ensemble (3 seeds + TTA)"))
    df = pd.concat(tables, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "nucleus_uncertainty.csv"), index=False)

    rows = []
    for run, g in df.groupby("run"):
        for base in ["sigma_learned", "sigma_tta_unet", "sigma_ensemble"]:
            for score in [base, base + "_rel", base + "_imgnorm"]:
                if score in g and g[score].notna().any():
                    rows.append({"run": run, "model": run.rsplit(" s", 1)[0] if " s" in run and run[-1].isdigit() else run, "score": score,
                                 "auroc": roc_auc(~g["matched"].values, g[score].values), "auroc_within_image": within_image_auroc(g, score),
                                 "n_nuclei": len(g), "precision_all": g["matched"].mean()})
        for score in ["dim", "small"]:
            rows.append({"run": run, "model": run.rsplit(" s", 1)[0] if " s" in run and run[-1].isdigit() else run, "score": score,
                         "auroc": roc_auc(~g["matched"].values, g[score].values), "auroc_within_image": within_image_auroc(g, score),
                         "n_nuclei": len(g), "precision_all": g["matched"].mean()})
    au = pd.DataFrame(rows)
    au.to_csv(os.path.join(a.out, "nucleus_auroc.csv"), index=False)
    summ = au.groupby(["model", "score"]).agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"), within=("auroc_within_image", "mean"), n=("auroc", "size")).reset_index()
    gates = []
    for run, g in df.groupby("run"):
        main_score = next(sc for sc in ["sigma_learned", "sigma_tta_unet", "sigma_ensemble"] if sc in g and g[sc].notna().any())
        for score in [main_score, main_score + "_rel", "dim"]:
            gc = gate_curve(g, score, n_ref)
            gc["run"], gc["score"] = run, score
            gates.append(gc)
    gates = pd.concat(gates, ignore_index=True)
    gates.to_csv(os.path.join(a.out, "nucleus_gate.csv"), index=False)
    md = [f"Reference nuclei: {n_ref}. AUROC of each per-nucleus score for flagging detections that match no reference nucleus (IoU < 0.5); mean ± sd over seeds.", "",
          "Pooled AUROC ranks nuclei across all images; within-image AUROC is the mean over images of the AUROC computed inside each image "
          "(removes differences in σ level between sparse and dense frames). _rel = σ/μ; _imgnorm = σ divided by the median σ of the image's nuclei.", "",
          "| model | score | pooled AUROC | within-image AUROC | seeds |", "|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        md.append(f"| {r.model} | {r.score} | {r.auroc_mean:.3f}" + (f" ± {r.auroc_sd:.3f}" if r.n > 1 else "") + f" | {r.within:.3f} | {r.n} |")
    gm = gates.assign(model=gates["run"].str.replace(r" s\d$", "", regex=True)).groupby(["model", "score", "discard"])[["precision", "recall", "f1"]].mean().reset_index()
    md += ["", "Quality gate (discard the highest-score detections; mean over seeds):", "", "| model | score | discarded | precision | recall | F1 |", "|---|---|---|---|---|---|"]
    for _, r in gm.iterrows():
        md.append(f"| {r.model} | {r.score} | {r.discard:.0%} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} |")
    open(os.path.join(a.out, "nucleus_uncertainty.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
