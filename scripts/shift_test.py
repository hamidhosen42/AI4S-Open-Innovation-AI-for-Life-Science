"""Distribution-shift stress test: does sigma notice when the input drifts away from training?

    python scripts/shift_test.py --runs runs/{baseline_unet,pretrained_l1,chipstain_nll}_s{0,1,2} --out report/results

Simulated shifts (crude proxies for what changes between plates, chips and microscopes): Gaussian blur
(a defocus proxy), additive sensor noise, and a modality swap (the dataset's digital phase-contrast
rendering of the same fields). Affine contrast changes are NOT tested: the per-image percentile
normalisation cancels them exactly. 50 test images (time-points 10 and 100), 8x TTA.
Reported per model (mean ± s.d. over seeds):
  * shift detection   AUROC of per-image mean sigma, shifted vs clean images
  * failure detection AUROC of per-image mean sigma, failed (Pearson r < 0.5) vs successful (r > 0.7) images,
                      pooled over clean and shifted conditions
  * validation gate   threshold = 95th percentile of per-image mean sigma on the clean VALIDATION split
                      (from outputs/multiseed_val); fraction of images flagged per condition
For ChipStain, sigma is also split into its learned (aleatoric) and TTA-disagreement terms.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

from chipstain.data import load_pair, make_splits, normalize_input, normalize_target, pad_to_multiple
from chipstain.metrics import image_metrics
from chipstain.model import predict_tta
from scripts.evaluate import get_device, load_run

CONDITIONS = [
    ("clean", None),
    ("blur σ=1 px", ("blur", 1.0)), ("blur σ=2 px", ("blur", 2.0)), ("blur σ=4 px", ("blur", 4.0)),
    ("noise 10 %", ("noise", 0.10)), ("noise 20 %", ("noise", 0.20)),
    ("modality: DPC", ("dpc", None)),
]
FAMILIES = {"blur": ["blur σ=1 px", "blur σ=2 px", "blur σ=4 px"], "noise": ["noise 10 %", "noise 20 %"], "modality": ["modality: DPC"]}
NAME = {"baseline_unet": "U-Net + TTA", "pretrained_l1": "+ ImageNet encoder + TTA", "chipstain_nll": "ChipStain + TTA"}


def corrupt(bf, dpc, kind, rng):
    if kind is None:
        return bf
    k, v = kind
    if k == "blur":
        return ndi.gaussian_filter(bf, v)
    if k == "noise":
        lo, hi = np.percentile(bf, [1, 99])
        return bf + rng.normal(0, v * (hi - lo), bf.shape)
    if k == "dpc":
        return dpc
    raise ValueError(k)


def auroc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    return mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"runs/{c}_s{s}" for c in NAME for s in (0, 1, 2)])
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--val", default="outputs/multiseed_val/per_image_val.csv")
    ap.add_argument("--out", default="report/results")
    ap.add_argument("--summarise_only", action="store_true", help="recompute tables from the saved shift_test.csv")
    a = ap.parse_args()
    device = get_device()
    _, _, test = make_splits(a.data)
    samples = [s for s in test if s.timepoint in (10, 100)]
    rows = []
    for run in ([] if a.summarise_only else a.runs):
        if not os.path.exists(os.path.join(run, "best.pt")):
            print("skip", run)
            continue
        model, cfg = load_run(run, device)
        tag = os.path.basename(run.rstrip("/"))
        for i, s in enumerate(tqdm(samples, desc=tag)):
            bf, fl = load_pair(s)
            dpc = tifffile.imread(s.base + "_dpc.tif").astype(np.float32)
            y = normalize_target(fl)
            for name, kind in CONDITIONS:
                rng = np.random.default_rng(1000 * i + len(name))
                xp, _ = pad_to_multiple(normalize_input(corrupt(bf.astype(np.float32), dpc, kind, rng)))
                xt = torch.from_numpy(xp)[None, None].to(device)
                with torch.no_grad():
                    mu, ale, epi = predict_tta(model, xt)
                    mu1, lv1 = model(xt)
                crop = lambda t: t[0, 0, :540, :540].cpu().numpy()  # noqa: E731
                p = crop(mu)
                s_epi = np.sqrt(crop(epi))
                s_ale = np.sqrt(crop(ale)) if ale is not None else None
                s_tot = np.sqrt(crop(epi) + (crop(ale) if ale is not None else 0))
                m = image_metrics(p, y)
                rows.append({"run": tag, "cfg": re.sub(r"_s\d+$", "", tag), "seed": int(tag.rsplit("_s", 1)[1]), "condition": name, "image": i,
                             "mean_sigma": float(s_tot.mean()), "mean_sigma_tta_views": float(s_epi.mean()),
                             "mean_sigma_learned_tta": float(s_ale.mean()) if s_ale is not None else np.nan,
                             "mean_sigma_single_pass": float(np.sqrt(crop(lv1.exp())).mean()) if lv1 is not None else np.nan, **m})
    os.makedirs(a.out, exist_ok=True)
    if a.summarise_only:
        df = pd.read_csv(os.path.join(a.out, "shift_test.csv"))
    else:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, "shift_test.csv"), index=False)

    # validation thresholds (clean validation images, same model and seed, TTA)
    thr = {}
    if os.path.exists(a.val):
        v = pd.read_csv(a.val)
        for run, g in v.groupby("run"):
            thr[run.replace("_tta", "")] = float(g["mean_sigma"].quantile(0.95))
    order = [c for c, _ in CONDITIONS]
    terms = [("mean_sigma", "total σ"), ("mean_sigma_learned_tta", "learned head (TTA mean)"), ("mean_sigma_tta_views", "TTA disagreement"), ("mean_sigma_single_pass", "learned head, single pass")]
    det, fail, gate, rob = [], [], [], []
    for (cfg, seed), g in df.groupby(["cfg", "seed"]):
        clean = g[g.condition == "clean"]
        for term, lab in terms:
            if g[term].isna().all():
                continue
            for fam, conds in FAMILIES.items():
                det.append({"cfg": cfg, "seed": seed, "term": lab, "family": fam, "auroc": auroc(clean[term], g[g.condition.isin(conds)][term])})
            sh = g  # clean + shifted images: failed (r < 0.5) vs successful (r > 0.7)
            fail.append({"cfg": cfg, "seed": seed, "term": lab, "auroc": auroc(sh[sh.pearson > 0.7][term], sh[sh.pearson < 0.5][term]),
                         "n_failed": int((sh.pearson < 0.5).sum()), "within_rho": np.nanmean([spearmanr(h[term], h["mae"]).statistic for _, h in g.groupby("condition")])})
        t = thr.get(f"{cfg}_s{seed}")
        for c in order:
            h = g[g.condition == c]
            rob.append({"cfg": cfg, "seed": seed, "condition": c, "pearson": h.pearson.mean(), "mae": h.mae.mean(), "sigma_ratio": h.mean_sigma.mean() / clean.mean_sigma.mean()})
            if t is not None:
                gate.append({"cfg": cfg, "seed": seed, "condition": c, "flagged": float((h.mean_sigma > t).mean())})
    det, fail, rob = pd.DataFrame(det), pd.DataFrame(fail), pd.DataFrame(rob)
    for name, d in [("shift_detection", det), ("shift_failure", fail), ("shift_robustness", rob)]:
        d.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    pm = lambda x: f"{np.nanmean(x):.2f} ± {np.nanstd(x, ddof=1):.2f}" if np.sum(~np.isnan(x)) > 1 else f"{np.nanmean(x):.2f}"  # noqa: E731
    lines = ["## Failure detection (AUROC of per-image mean σ: failed images r < 0.5 vs successful r > 0.7, clean and shifted images pooled; mean ± s.d. over seeds)", "",
             "| model | σ term | failure AUROC | failed images | within-condition ρ(σ, MAE) |", "|---|---|---|---|---|"]
    for (cfg, term), g in fail.groupby(["cfg", "term"], sort=False):
        lines.append(f"| {NAME.get(cfg, cfg)} | {term} | {pm(g.auroc.values)} | {g.n_failed.sum()} | {pm(g.within_rho.values)} |")
    lines += ["", "## Shift detection (AUROC of per-image mean σ, shifted vs clean; 0.5 = no signal, < 0.5 = σ falls under shift)", "",
              "| model | σ term | blur | noise | modality (DPC) |", "|---|---|---|---|---|"]
    for (cfg, term), g in det.groupby(["cfg", "term"], sort=False):
        lines.append(f"| {NAME.get(cfg, cfg)} | {term} | " + " | ".join(pm(g[g.family == f].auroc.values) for f in FAMILIES) + " |")
    lines += ["", "## Accuracy and σ per condition (mean over seeds)", "", "| model | condition | Pearson r | MAE | mean σ / clean |", "|---|---|---|---|---|"]
    for (cfg, c), g in rob.groupby(["cfg", "condition"], sort=False):
        lines.append(f"| {NAME.get(cfg, cfg)} | {c} | {g.pearson.mean():.3f} | {g.mae.mean():.4f} | {g.sigma_ratio.mean():.2f}× |")
    if gate:
        gate = pd.DataFrame(gate)
        gate.to_csv(os.path.join(a.out, "shift_gate.csv"), index=False)
        lines += ["", "## Validation-calibrated gate (threshold = 95th percentile of mean σ on clean validation images): fraction of images flagged", "",
                  "| model | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for cfg, g in gate.groupby("cfg", sort=False):
            lines.append(f"| {NAME.get(cfg, cfg)} | " + " | ".join(f"{100 * g[g.condition == c].flagged.mean():.0f} %" for c in order) + " |")
    open(os.path.join(a.out, "shift_test.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
