"""Aggregate the 3-seed evaluation and run paired significance tests.

    python scripts/multiseed_summary.py --plain outputs/multiseed --tta outputs/multiseed_tta --out report/results

Seed spread: each metric is averaged over the 125 test images per (config, seed), then
mean ± std is taken over the 3 seeds.

Paired tests: the 125 test images are 25 fields x 5 time-points of one time-lapse, so images
of the same field are not independent. The test unit is therefore the FIELD (n = 25): each
metric is averaged over time-points and seeds per field, and configs are compared with a
two-sided Wilcoxon signed-rank test; p-values are Holm-corrected over all tests reported.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

METRICS = ["pearson", "ssim", "psnr", "mae", "seg_f1", "spearman_unc_err", "ause", "mean_sigma"]
LABEL = {
    "baseline_unet": "U-Net baseline (L1)",
    "pretrained_l1": "+ ImageNet encoder (L1)",
    "baseline_unet+tta": "U-Net baseline + TTA (σ = view variance)",
    "chipstain_nll": "ChipStain (β-NLL head)",
    "chipstain_nll+tta": "ChipStain + TTA (full)",
}
ORDER = list(LABEL)
# (a, b, metrics, higher_is_better per metric) -> test "a better than b"
TESTS = [
    ("chipstain_nll+tta", "baseline_unet+tta", ["spearman_unc_err", "ause"]),
    ("chipstain_nll", "baseline_unet+tta", ["spearman_unc_err", "ause"]),
    ("chipstain_nll+tta", "baseline_unet", ["pearson", "ssim", "seg_f1"]),
    ("chipstain_nll", "baseline_unet", ["pearson", "ssim", "seg_f1"]),
    ("pretrained_l1", "baseline_unet", ["pearson", "ssim", "seg_f1"]),
]
LOWER_BETTER = {"mae", "ause"}


def load(path, tta):
    df = pd.read_csv(os.path.join(path, "per_image_test.csv"))
    m = df["run"].str.extract(r"^(?P<cfg>.+)_s(?P<seed>\d+)(?P<t>_tta)?$")
    df["cfg"] = m["cfg"] + np.where(m["t"].notna(), "+tta", "")
    df["seed"] = m["seed"].astype(int)
    return df


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="outputs/multiseed")
    ap.add_argument("--tta", default="outputs/multiseed_tta")
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([load(a.plain, False), load(a.tta, True)], ignore_index=True)
    df.to_csv(os.path.join(a.out, "multiseed_per_image.csv"), index=False)
    cfgs = [c for c in ORDER if c in set(df["cfg"])]

    # ---- per-seed means, then mean ± std over seeds
    per_seed = df.groupby(["cfg", "seed"])[[m for m in METRICS if m in df]].mean()
    rho = df.dropna(subset=["mean_sigma"]).groupby(["cfg", "seed"]).apply(lambda g: spearmanr(g["mean_sigma"], g["mae"]).statistic)
    per_seed["img_rho_sigma_mae"] = rho
    per_seed.to_csv(os.path.join(a.out, "multiseed_seed_means.csv"))
    agg = per_seed.groupby("cfg").agg(["mean", "std", "count"])

    cols = [("pearson", "Pearson r ↑"), ("ssim", "SSIM ↑"), ("psnr", "PSNR ↑"), ("seg_f1", "seg-F1 ↑"),
            ("spearman_unc_err", "ρ(σ, err) ↑"), ("ause", "AUSE ↓"), ("img_rho_sigma_mae", "image ρ(σ̄, MAE) ↑")]
    md = ["| Model | " + " | ".join(h for _, h in cols) + " |", "|---|" + "---|" * len(cols)]
    rows_html = []
    for c in cfgs:
        cells = []
        for k, _ in cols:
            mu, sd = agg.loc[c, (k, "mean")], agg.loc[c, (k, "std")]
            cells.append("—" if pd.isna(mu) else (f"{mu:.2f} ± {sd:.2f}" if k == "psnr" else f"{mu:.3f} ± {sd:.3f}"))
        md.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
        rows_html.append((c, cells))
    real = df.groupby("seed")["seg_f1_realfluo"].mean().mean()
    md.append(f"| Real fluorescence, same segmentation pipeline (ceiling) | — | — | — | {real:.3f} | — | — | — |")
    n_seeds = int(agg[("pearson", "count")].min())

    # ---- paired tests at field level (n = 25)
    field = df.groupby(["cfg", "well", "field"])[[m for m in METRICS if m in df]].mean()
    trows = []
    for x, y, ms in TESTS:
        if x not in cfgs or y not in cfgs:
            continue
        for m in ms:
            fx, fy = field.loc[x][m], field.loc[y][m]
            j = fx.index.intersection(fy.index)
            d = (fx.loc[j] - fy.loc[j]).values
            if m in LOWER_BETTER:
                d = -d
            stat = wilcoxon(d, alternative="two-sided")
            wins = int((d > 0).sum())
            seed_wins = int(sum(
                (per_seed.loc[(x, s), m] < per_seed.loc[(y, s), m]) if m in LOWER_BETTER else (per_seed.loc[(x, s), m] > per_seed.loc[(y, s), m])
                for s in sorted(set(df["seed"])) if (x, s) in per_seed.index and (y, s) in per_seed.index))
            trows.append({"a": x, "b": y, "metric": m, "mean_a": fx.loc[j].mean(), "mean_b": fy.loc[j].mean(),
                          "diff_a_minus_b": (fx.loc[j] - fy.loc[j]).mean(), "fields_a_better": wins, "n_fields": len(j),
                          "seeds_a_better": seed_wins, "n_seeds": n_seeds, "p": stat.pvalue})
    tests = pd.DataFrame(trows)
    tests["p_holm"] = holm(tests["p"].values)
    tests.to_csv(os.path.join(a.out, "multiseed_tests.csv"), index=False)

    md += ["", f"Mean ± std over {n_seeds} seeds (each seed: mean over the 125 test images of well R05-C03).", "",
           "Paired tests (Wilcoxon signed-rank, field-level n = 25, Holm-corrected):", "",
           "| A vs B | metric | A | B | A better in fields | A better in seeds | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        md.append(f"| {LABEL[r.a]} vs {LABEL[r.b]} | {r.metric} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.fields_a_better}/{r.n_fields} | {r.seeds_a_better}/{r.n_seeds} | {r.p_holm:.2g} |")
    open(os.path.join(a.out, "multiseed_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
