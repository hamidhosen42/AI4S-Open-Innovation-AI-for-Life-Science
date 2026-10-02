"""Aggregate the 3-seed evaluations, compare uncertainty with uncertainty-free proxies, and run
paired significance tests.

    python scripts/multiseed_summary.py --dirs outputs/multiseed outputs/multiseed_tta outputs/ablate outputs/ablate_tta --out report/results

Seed spread: each metric is averaged over the 125 test images per (config, seed); mean ± sd is
then taken over the 3 seeds.
Paired tests: the 125 test images are 25 fields x 5 time-points of one time-lapse, so images of
the same field are not independent. The test unit is the FIELD (n = 25): metrics are averaged
over time-points and seeds per field and compared with a two-sided Wilcoxon signed-rank test,
Holm-corrected over all tests; because seeds are averaged first, inference relies on the
hierarchical seed+field bootstrap CI and per-seed tests instead.
Image level: Spearman correlation of per-image mean sigma with per-image error, next to the
same correlation for the predicted nucleus count (cell density), and the partial correlation
of mean sigma with error controlling for the predicted count.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

METRICS = ["pearson", "ssim", "psnr", "mae", "seg_f1", "spearman_unc_err", "ause", "gain20", "mean_sigma",
           "spearman_mu_err", "ause_mu", "gain20_mu", "spearman_grad_err", "ause_grad", "gain20_grad", "gain20_oracle", "n_pred", "n_ref"]
LABEL = {
    "baseline_unet": "U-Net baseline (scratch, L1, lr 1e-3)",
    "ablate_scratch_l1_lr5e4": "U-Net from scratch (L1, lr 5e-4)",
    "pretrained_l1": "+ ImageNet encoder (L1, lr 5e-4)",
    "ablate_pretrained_mse": "+ ImageNet encoder (MSE, lr 5e-4)",
    "baseline_unet+tta": "U-Net baseline + TTA (σ = view s.d.)",
    "pretrained_l1+tta": "+ ImageNet encoder + TTA (σ = view s.d.)",
    "ablate_nll_beta0": "ChipStain head, β = 0 (plain NLL)",
    "ablate_nll_beta0+tta": "ChipStain head, β = 0, + TTA",
    "ablate_nll_beta1": "ChipStain head, β = 1",
    "ablate_nll_beta1+tta": "ChipStain head, β = 1, + TTA",
    "chipstain_nll": "ChipStain (β-NLL, β = 0.5)",
    "chipstain_nll+tta": "ChipStain + TTA (full)",
}
ORDER = list(LABEL)
MAIN = ["baseline_unet", "pretrained_l1", "baseline_unet+tta", "pretrained_l1+tta", "chipstain_nll", "chipstain_nll+tta"]
FID = ["pearson", "ssim", "mae", "psnr", "seg_f1"]
UNC = ["spearman_unc_err", "ause"]
TESTS = [
    ("chipstain_nll+tta", "baseline_unet+tta", FID + UNC),
    ("chipstain_nll+tta", "pretrained_l1+tta", FID + UNC),   # matched control: same encoder and LR, only the head/loss differ
    ("pretrained_l1+tta", "baseline_unet+tta", UNC),          # what the encoder / LR alone does to the TTA uncertainty
    ("chipstain_nll", "pretrained_l1+tta", UNC),
    ("chipstain_nll", "baseline_unet+tta", ["spearman_unc_err", "ause"]),
    ("chipstain_nll+tta", "baseline_unet", FID),
    ("chipstain_nll", "baseline_unet", FID),
    ("pretrained_l1", "baseline_unet", FID),
    # de-confounded ablation steps
    ("ablate_scratch_l1_lr5e4", "baseline_unet", FID),          # learning rate only
    ("pretrained_l1", "ablate_scratch_l1_lr5e4", FID),          # pre-training only (same LR)
    ("ablate_pretrained_mse", "pretrained_l1", FID),            # L1 -> MSE only
    ("chipstain_nll", "ablate_pretrained_mse", FID),            # variance head (Gaussian NLL vs MSE)
    ("chipstain_nll", "pretrained_l1", FID),                    # head + loss together
    ("chipstain_nll", "ablate_nll_beta0", FID + ["spearman_unc_err", "ause"]),
    ("chipstain_nll", "ablate_nll_beta1", FID + ["spearman_unc_err", "ause"]),
]
LOWER_BETTER = {"mae", "ause", "ause_mu", "ause_grad"}


def load(path):
    df = pd.read_csv(os.path.join(path, "per_image_test.csv"))
    m = df["run"].str.extract(r"^(?P<cfg>.+)_s(?P<seed>\d+)(?P<t>_tta)?$")
    df = df[m["cfg"].notna()].copy()
    m = m[m["cfg"].notna()]
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


def hier_bootstrap(sf_a, sf_b, n_boot=4000, seed=0):
    """95 % CI of mean(A) - mean(B) resampling seeds (independently per config) and fields.
    sf_* : DataFrame indexed by (seed, field) with one value. Accounts for run-to-run variance."""
    rng = np.random.default_rng(seed)
    A = sf_a.unstack(0)  # rows = field, cols = seed
    B = sf_b.unstack(0)
    fields = A.index.intersection(B.index)
    A, B = A.loc[fields].values, B.loc[fields].values
    out = np.empty(n_boot)
    for i in range(n_boot):
        f = rng.integers(0, len(fields), len(fields))
        sa = rng.integers(0, A.shape[1], A.shape[1])
        sb = rng.integers(0, B.shape[1], B.shape[1])
        out[i] = A[np.ix_(f, sa)].mean() - B[np.ix_(f, sb)].mean()
    return np.percentile(out, [2.5, 97.5])


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z (Pearson on ranks of residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def fmt(mu, sd, k):
    if pd.isna(mu):
        return "—"
    if k == "psnr":
        return f"{mu:.2f} ± {sd:.2f}"
    if k == "mae":
        return f"{mu:.4f} ± {sd:.4f}"
    if k.startswith("gain20"):
        return f"{100*mu:.1f} ± {100*sd:.1f} %"
    return f"{mu:.3f} ± {sd:.3f}"


def table(agg, cfgs, cols):
    md = ["| Model | " + " | ".join(h for _, h in cols) + " |", "|---|" + "---|" * len(cols)]
    for c in cfgs:
        md.append(f"| {LABEL.get(c, c)} | " + " | ".join(fmt(agg.loc[c, (k, 'mean')], agg.loc[c, (k, 'std')], k) if (k, "mean") in agg.columns else "—" for k, _ in cols) + " |")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs/multiseed", "outputs/multiseed_tta", "outputs/ablate", "outputs/ablate_tta"])
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([load(d) for d in a.dirs if os.path.exists(os.path.join(d, "per_image_test.csv"))], ignore_index=True)
    df.to_csv(os.path.join(a.out, "multiseed_per_image.csv"), index=False)
    cfgs = [c for c in ORDER if c in set(df["cfg"])]
    mets = [m for m in METRICS if m in df]
    per_seed = df.groupby(["cfg", "seed"])[mets].mean()
    img = []
    for (c, s), g in df.dropna(subset=["mean_sigma"]).groupby(["cfg", "seed"]):
        img.append({"cfg": c, "seed": s,
                    "img_rho_sigma_mae": spearmanr(g["mean_sigma"], g["mae"]).statistic,
                    "img_rho_npred_mae": spearmanr(g["n_pred"], g["mae"]).statistic,
                    "img_partial_sigma_mae_given_npred": partial_spearman(g["mean_sigma"], g["mae"], g["n_pred"]),
                    "img_rho_sigma_1mf1": spearmanr(g["mean_sigma"], 1 - g["seg_f1"]).statistic,
                    "img_rho_npred_1mf1": spearmanr(g["n_pred"], 1 - g["seg_f1"]).statistic,
                    "img_rho_sigma_1mr": spearmanr(g["mean_sigma"], 1 - g["pearson"]).statistic})
    if img:
        per_seed = per_seed.join(pd.DataFrame(img).set_index(["cfg", "seed"]))
    per_seed.to_csv(os.path.join(a.out, "multiseed_seed_means.csv"))
    agg = per_seed.groupby("cfg").agg(["mean", "std", "count"])
    n_seeds = int(agg[("pearson", "count")].min())
    real = df[df["cfg"] == cfgs[0]].groupby("seed")["seg_f1_realfluo"].mean().mean()

    md = [f"## Main comparison (mean ± sd over {n_seeds} seeds; each seed = mean over the 125 test images of well R05-C03)", ""]
    md += table(agg, [c for c in MAIN if c in cfgs], [("pearson", "Pearson r ↑"), ("ssim", "SSIM ↑"), ("mae", "MAE ↓"), ("psnr", "PSNR ↑"), ("seg_f1", "seg-F1 ↑"),
                                                     ("spearman_unc_err", "ρ(σ, err) ↑"), ("ause", "AUSE ↓"), ("gain20", "MAE drop, top-20 % σ removed ↑")])
    md += [f"| Real fluorescence, same segmentation pipeline (reference level, not a bound) | — | — | — | — | {real:.3f} | — | — | — |", ""]
    abl = [c for c in ORDER if c in cfgs and c not in ("baseline_unet+tta", "chipstain_nll+tta")]
    if any(c.startswith("ablate") for c in cfgs):
        md += ["## Ablation arms (no TTA)", ""]
        md += table(agg, abl, [("pearson", "Pearson r ↑"), ("ssim", "SSIM ↑"), ("mae", "MAE ↓"), ("seg_f1", "seg-F1 ↑"), ("spearman_unc_err", "ρ(σ, err) ↑"), ("ause", "AUSE ↓")])
        md.append("")
    md += ["## Does σ beat uncertainty-free proxies at ranking pixel errors?", "",
           "Proxies are computed from the model's own prediction: intensity = predicted μ, edges = Sobel edge strength of μ.", ""]
    md += table(agg, [c for c in cfgs if (("spearman_unc_err", "mean") in agg.columns and not pd.isna(agg.loc[c, ("spearman_unc_err", "mean")]))],
                [("spearman_unc_err", "ρ: σ"), ("spearman_mu_err", "ρ: intensity"), ("spearman_grad_err", "ρ: edges"), ("ause", "AUSE: σ"), ("ause_mu", "AUSE: intensity"), ("ause_grad", "AUSE: edges"),
                 ("gain20", "gain: σ"), ("gain20_mu", "gain: intensity"), ("gain20_grad", "gain: edges"), ("gain20_oracle", "gain: oracle")])
    md.append("")
    if img:
        md += ["## Image level: σ versus cell density", ""]
        md += table(agg, [c for c in cfgs if ("img_rho_sigma_mae", "mean") in agg.columns and not pd.isna(agg.loc[c, ("img_rho_sigma_mae", "mean")])],
                    [("img_rho_sigma_mae", "ρ(mean σ, MAE)"), ("img_rho_npred_mae", "ρ(n_pred, MAE)"), ("img_partial_sigma_mae_given_npred", "partial ρ(σ, MAE) given n_pred"),
                     ("img_rho_sigma_1mf1", "ρ(mean σ, 1−F1)"), ("img_rho_npred_1mf1", "ρ(n_pred, 1−F1)"), ("img_rho_sigma_1mr", "ρ(mean σ, 1−r)")])
        md.append("")

    field = df.groupby(["cfg", "well", "field"])[mets].mean()
    trows = []
    for fam, (x, y, ms) in enumerate(TESTS):
        if x not in cfgs or y not in cfgs:
            continue
        for m in ms:
            if m not in field.columns or field.loc[x][m].isna().all() or field.loc[y][m].isna().all():
                continue
            fx, fy = field.loc[x][m], field.loc[y][m]
            j = fx.index.intersection(fy.index)
            d = (fx.loc[j] - fy.loc[j]).values * (-1 if m in LOWER_BETTER else 1)
            p = wilcoxon(d, alternative="two-sided").pvalue if np.any(d != 0) else 1.0
            better = lambda s: (per_seed.loc[(x, s), m] < per_seed.loc[(y, s), m]) if m in LOWER_BETTER else (per_seed.loc[(x, s), m] > per_seed.loc[(y, s), m])  # noqa: E731
            seeds = [s for s in sorted(set(df["seed"])) if (x, s) in per_seed.index and (y, s) in per_seed.index]
            seeds = seeds
            sfa = df[df.cfg == x].groupby(["seed", "field"])[m].mean()
            sfb = df[df.cfg == y].groupby(["seed", "field"])[m].mean()
            ci = hier_bootstrap(sfa, sfb)
            sig_seeds = sig_worse = 0
            for sa in seeds:
                da = sfa.loc[sa]; db = sfb.loc[sa] if sa in sfb.index.get_level_values(0) else None
                if db is None:
                    continue
                jj = da.index.intersection(db.index)
                dd = (da.loc[jj] - db.loc[jj]).values * (-1 if m in LOWER_BETTER else 1)
                if np.any(dd != 0) and wilcoxon(dd).pvalue < 0.05:
                    sig_seeds += int(dd.mean() > 0)
                    sig_worse += int(dd.mean() < 0)
            trows.append({"family": fam, "a": x, "b": y, "metric": m, "mean_a": fx.loc[j].mean(), "mean_b": fy.loc[j].mean(),
                          "ci_lo": ci[0], "ci_hi": ci[1], "seeds_sig_better": sig_seeds, "seeds_sig_worse": sig_worse,
                          "diff_a_minus_b": (fx.loc[j] - fy.loc[j]).mean(), "fields_a_better": int((d > 0).sum()), "n_fields": len(j),
                          "seeds_a_better": int(sum(better(s) for s in seeds)), "n_seeds": len(seeds), "p": p})
    tests = pd.DataFrame(trows)
    tests["p_holm"] = holm(tests["p"].values)
    tests.to_csv(os.path.join(a.out, "multiseed_tests.csv"), index=False)
    md += ["## Paired tests", "",
           "Inference rests on the 95 % CI of A − B from a hierarchical bootstrap that resamples seeds (per configuration) and fields, and on per-seed tests "
           "(field-level Wilcoxon within each seed pair, p < 0.05): 'seeds better / worse' counts the seed pairs in which A is significantly better / worse. "
           "The pooled field-level Wilcoxon averages over seeds and only measures consistency within this single well (one plate, one imaging session); "
           "it is shown, Holm-corrected over all tests, for the uncertainty metrics only.", "",
           "| A vs B | metric | A | B | A − B (95 % CI, seeds + fields) | seeds sig. better / worse | fields A better (seed-avg) | p (pooled, Holm) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        p = f"{r.p_holm:.2g}" if r.metric in UNC else "—"
        md.append(f"| {LABEL.get(r.a, r.a)} vs {LABEL.get(r.b, r.b)} | {r.metric} | {r.mean_a:.4f} | {r.mean_b:.4f} | {r.mean_a - r.mean_b:+.4f} ({r.ci_lo:+.4f}, {r.ci_hi:+.4f}) | {r.seeds_sig_better} / {r.seeds_sig_worse} of {r.n_seeds} | {r.fields_a_better}/{r.n_fields} | {p} |")
    open(os.path.join(a.out, "multiseed_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
