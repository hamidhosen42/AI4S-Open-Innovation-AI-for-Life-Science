"""Figures for the multi-seed ablation, the distribution-shift test and the per-nucleus gate.

    python scripts/make_figures_extra.py --res report/results --out report/figures
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# validated categorical slots (light surface): 1 blue, 2 orange, 3 aqua; neutral context gray
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.facecolor": "white",
})
CFG = ["baseline_unet", "pretrained_l1", "baseline_unet+tta", "chipstain_nll", "chipstain_nll+tta"]
NAME = {"baseline_unet": "U-Net baseline (L1)", "pretrained_l1": "+ ImageNet encoder (L1)",
        "baseline_unet+tta": "U-Net baseline + TTA", "chipstain_nll": "ChipStain (β-NLL)", "chipstain_nll+tta": "ChipStain + TTA (full)"}


def multiseed(res, path):
    ps = pd.read_csv(os.path.join(res, "multiseed_seed_means.csv"))
    panels = [("pearson", "Pearson r  (higher is better)"), ("seg_f1", "Nuclei seg-F1  (higher is better)"),
              ("spearman_unc_err", "ρ(σ, |error|)  (higher is better)"), ("ause", "AUSE  (lower is better)")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        for yi, c in enumerate(CFG):
            v = ps.loc[ps.cfg == c, m].dropna().values
            if len(v) == 0:
                ax.text(0.02, yi, "n/a (no σ)", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
                continue
            col = BLUE if c == "chipstain_nll+tta" else INK2
            ax.plot([v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1)], [yi, yi], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(v, [yi] * len(v), s=14, facecolor="white", edgecolor=col, lw=1, zorder=3)
            ax.scatter([v.mean()], [yi], s=46, marker="D", color=col, edgecolor="white", lw=1.2, zorder=4)
        if m == "seg_f1":
            real = pd.read_csv(os.path.join(res, "multiseed_per_image.csv"))["seg_f1_realfluo"].mean()
            ax.axvline(real, color=MUTED, lw=1, ls=(0, (3, 2)))
            ax.text(real, len(CFG) - 0.45, " real-stain ceiling", fontsize=7, color=INK2, va="bottom")
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(CFG) - 0.2)
    axes[0].set_yticks(range(len(CFG)))
    axes[0].set_yticklabels([NAME[c] for c in CFG])
    fig.text(0.01, -0.02, "◇ mean of 3 seeds · ○ individual seeds · bar = ±1 s.d. across seeds · each seed = mean over 125 test images (well R05-C03)", fontsize=7.5, color=INK2)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def shift(res, path):
    df = pd.read_csv(os.path.join(res, "shift_test.csv"))
    order = ["clean", "defocus σ=1", "defocus σ=2", "defocus σ=4", "noise 5 %", "noise 10 %", "noise 20 %", "contrast ×0.5", "contrast ×0.25", "modality: DPC"]
    runs = [("chipstain_nll_s0", "ChipStain + TTA (learned σ)", BLUE), ("baseline_unet_s0", "U-Net baseline + TTA (view-variance σ)", ORANGE)]
    t = df.groupby(["run", "condition"])[["pearson", "mean_sigma", "mae"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    y = np.arange(len(order))[::-1]
    for k, (run, lab, col) in enumerate(runs):
        tt = t.loc[run].loc[order]
        off = 0.14 if k == 0 else -0.14
        axes[0].scatter(tt["pearson"], y + off, s=30, color=col, edgecolor="white", lw=1, zorder=3, label=lab)
        rel = tt["mean_sigma"] / tt.loc["clean", "mean_sigma"]
        axes[1].scatter(rel, y + off, s=30, color=col, edgecolor="white", lw=1, zorder=3, label=lab)
    axes[0].set_title("A  Accuracy under shift: Pearson r vs real stain")
    axes[0].set_xlim(0, 0.85)
    axes[1].set_xscale("log", base=2)
    axes[1].axvline(1, color=MUTED, lw=1, ls=(0, (3, 2)))
    axes[1].set_title("B  Does the uncertainty notice? mean σ relative to clean")
    axes[1].set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8, 16])
    axes[1].set_xticklabels(["⅛×", "¼×", "½×", "1×", "2×", "4×", "8×", "16×"])
    axes[1].text(1.05, y[0] + 0.45, "no change", fontsize=7, color=INK2)
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for yy in y[1:]:
            pass
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    for b in [y[3] - 0.5, y[6] - 0.5, y[8] - 0.5]:
        for ax in axes:
            ax.axhline(b, color=GRID, lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def nucleus(res, path):
    df = pd.read_csv(os.path.join(res, "nucleus_uncertainty.csv"))
    n_ref = df.drop_duplicates("image")["n_ref_image"].sum()
    d = df.sort_values("sigma").reset_index(drop=True)
    fr = np.linspace(0, 0.5, 51)
    prec, rec, f1 = [], [], []
    for f in fr:
        k = int(round(len(d) * (1 - f)))
        tp = d["matched"].iloc[:k].sum()
        p, r = tp / k, tp / n_ref
        prec.append(p); rec.append(r); f1.append(2 * p * r / (p + r))
    base = d["matched"].mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(fr * 100, prec, color=BLUE, lw=2)
    ax.plot(fr * 100, [base] * len(fr), color=MUTED, lw=1.5, ls=(0, (3, 2)))
    ax.plot(fr * 100, rec, color=ORANGE, lw=2)
    ax.plot(fr * 100, f1, color=AQUA, lw=2)
    for vals, lab, col in [(prec, "precision, highest-σ removed first", BLUE), ([base] * len(fr), "precision, random removal", INK2), (rec, "recall", ORANGE), (f1, "F1", AQUA)]:
        ax.text(fr[-1] * 100 + 1, vals[-1], lab, color=INK if col != INK2 else INK2, fontsize=7.5, va="center")
        ax.scatter([fr[-1] * 100], [vals[-1]], s=12, color=col, zorder=3)
    ax.set_xlabel("% of predicted nuclei discarded (most uncertain first)")
    ax.set_ylabel("proportion")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.3, 0.85)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Per-nucleus quality gate on the test well")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="report/results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    multiseed(a.res, os.path.join(a.out, "multiseed.png"))
    shift(a.res, os.path.join(a.out, "shift_test.png"))
    nucleus(a.res, os.path.join(a.out, "nucleus_gate.png"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
