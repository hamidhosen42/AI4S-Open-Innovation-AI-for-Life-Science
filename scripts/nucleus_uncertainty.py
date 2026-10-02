"""Per-nucleus uncertainty: does sigma identify wrong detections?

    python scripts/nucleus_uncertainty.py --run runs/chipstain_nll_s0 --out report/results

For every test image: predict (TTA), segment nuclei on the prediction, give each predicted
nucleus the mean sigma inside it, and mark it correct if it matches a StarDist reference
nucleus at IoU >= 0.5. Then rank all predicted nuclei by sigma and report
(i) AUROC of sigma for flagging unmatched detections, and (ii) precision / F1 when the
most-uncertain detections are discarded (a "quality gate").
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # allow `from scripts...`

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from chipstain.data import PairDataset, load_mask, make_splits
from chipstain.metrics import segment_nuclei
from scripts.evaluate import get_device, load_run, predict


def roc_auc(positive, score):
    """AUROC via the Mann-Whitney U statistic (no scikit-learn dependency)."""
    positive = np.asarray(positive, bool)
    score = np.asarray(score, float)
    u = mannwhitneyu(score[positive], score[~positive], alternative="two-sided").statistic
    return float(u / (positive.sum() * (~positive).sum()))


def object_matches(pred_lab, gt_lab, thr=0.5):
    """Boolean per predicted object (ids 1..max): matched to some reference object at IoU >= thr.
    At IoU > 0.5 a match is unique, so this agrees with chipstain.metrics.match_f1."""
    n_p, n_g = int(pred_lab.max()), int(gt_lab.max())
    if n_p == 0:
        return np.zeros(0, bool), n_g
    both = (pred_lab > 0) & (gt_lab > 0)
    inter = np.zeros((n_p + 1, n_g + 1))
    np.add.at(inter, (pred_lab[both], gt_lab[both]), 1)
    ap = np.bincount(pred_lab.ravel(), minlength=n_p + 1)[:, None]
    ag = np.bincount(gt_lab.ravel(), minlength=n_g + 1)[None, :]
    iou = inter / np.maximum(ap + ag - inter, 1)
    iou[0, :] = 0
    iou[:, 0] = 0
    matched = iou.max(1)[1:] >= thr
    return matched, int((np.unique(gt_lab) > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/chipstain_nll_s0")
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    device = get_device()
    model, _ = load_run(a.run, device)
    _, _, test = make_splits(a.data)
    ds = PairDataset(test, crop=None, augment=False, cache=False)
    rows, n_ref_total = [], 0
    for i in tqdm(range(len(ds))):
        x, _ = ds[i]
        mu, var = predict(model, x[None].to(device), tta=True)
        p = np.clip(mu[0, 0, :540, :540].cpu().numpy(), 0, 1)
        sig = np.sqrt(var[0, 0, :540, :540].cpu().numpy())
        lab = segment_nuclei(p)
        ref = load_mask(test[i])
        ok, n_ref = object_matches(lab, ref)
        n_ref_total += n_ref
        if lab.max() == 0:
            continue
        idx = np.arange(1, lab.max() + 1)
        s_mean = ndi.mean(sig, lab, idx)
        area = ndi.sum(np.ones_like(lab), lab, idx)
        for j in range(len(idx)):
            rows.append({"image": i, "field": test[i].field, "timepoint": test[i].timepoint, "sigma": s_mean[j], "area": area[j], "matched": bool(ok[j]), "n_ref_image": n_ref})
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "nucleus_uncertainty.csv"), index=False)

    auroc = roc_auc(~df["matched"], df["sigma"])
    auroc_area = roc_auc(~df["matched"], -df["area"])  # naive alternative: small objects are suspicious
    order = df.sort_values("sigma").reset_index(drop=True)
    tp_all, n_all = order["matched"].sum(), len(order)
    gate = []
    for keep in [1.0, 0.95, 0.9, 0.8, 0.7]:
        k = int(round(n_all * keep))
        tp = order["matched"].iloc[:k].sum()
        prec, rec = tp / k, tp / n_ref_total
        gate.append({"keep_fraction": keep, "n_kept": k, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec)})
    gate = pd.DataFrame(gate)
    gate.to_csv(os.path.join(a.out, "nucleus_gate.csv"), index=False)
    # random-order baseline for the same keep fractions
    rng = np.random.default_rng(0)
    rand_prec = {k: np.mean([df["matched"].sample(frac=1, random_state=int(s)).iloc[: int(round(n_all * k))].mean() for s in rng.integers(0, 1e6, 50)]) for k in gate["keep_fraction"]}
    lines = [f"Predicted nuclei: {n_all} (reference nuclei: {n_ref_total}); matched at IoU>=0.5: {tp_all} ({100*tp_all/n_all:.1f} %)",
             f"AUROC of per-nucleus mean sigma for flagging unmatched detections: {auroc:.3f} (object area alone: {auroc_area:.3f})",
             f"mean sigma matched = {df.loc[df.matched,'sigma'].mean():.4f}, unmatched = {df.loc[~df.matched,'sigma'].mean():.4f}", "",
             "| keep (lowest-σ) | precision | random-order precision | recall | F1 |", "|---|---|---|---|---|"]
    for _, r in gate.iterrows():
        lines.append(f"| {r.keep_fraction:.0%} | {r.precision:.3f} | {rand_prec[r.keep_fraction]:.3f} | {r.recall:.3f} | {r.f1:.3f} |")
    open(os.path.join(a.out, "nucleus_uncertainty.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
