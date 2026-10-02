"""Build report figures from outputs/eval.

    python scripts/make_figures.py --eval outputs/eval --out report/figures
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chipstain.metrics import sparsification

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_preds(path):
    z = np.load(path)
    n = max(int(k.split("_")[-1]) for k in z.files) + 1
    out = []
    for j in range(n):
        d = {k[: -len(f"_{j}")]: z[k] for k in z.files if k.endswith(f"_{j}")}
        out.append(d)
    return out


def qualitative(preds, path, idx=(0, 3, 6)):
    cols = ["bf", "gt", "pred", "err", "unc"]
    titles = ["Bright-field input", "Real H2B fluorescence", "Predicted H2B", "|error|", "Predicted σ"]
    fig, ax = plt.subplots(len(idx), 5, figsize=(13, 2.7 * len(idx)))
    for r, i in enumerate(idx):
        d = preds[i]
        err = np.abs(d["pred"] - d["gt"])
        ims = [d["bf"], d["gt"], np.clip(d["pred"], 0, 1), err, d.get("unc")]
        cmaps = ["gray", "magma", "magma", "inferno", "viridis"]
        for c in range(5):
            a = ax[r, c]
            if ims[c] is None:
                a.axis("off"); continue
            kw = {"vmin": 0, "vmax": 1} if c in (1, 2) else ({"vmin": 0, "vmax": 0.3} if c == 3 else {})
            a.imshow(ims[c], cmap=cmaps[c], **kw)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(titles[c])
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def sparsification_plot(preds, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for j, d in enumerate(preds[:6]):
        if d.get("unc") is None:
            continue
        err = np.abs(d["pred"] - d["gt"])
        fr, cu, co, ause = sparsification(err, d["unc"])
        ax.plot(fr, cu / cu[0], color="#3E6FD9", alpha=0.6, lw=1.2, label="by predicted σ" if j == 0 else None)
        ax.plot(fr, co / co[0], color="#12A36F", alpha=0.6, lw=1.2, ls="--", label="oracle (by |error|)" if j == 0 else None)
    ax.plot([0, 1], [1, 1], color="#999", lw=1, ls=":", label="random")
    ax.set_xlabel("fraction of most-uncertain pixels removed"); ax.set_ylabel("remaining MAE (normalised)")
    ax.set_title("Sparsification (test images)"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def calib_scatter(df, path):
    d = df[df["mean_sigma"].notna()]
    if d.empty:
        return
    fig, ax = plt.subplots(1, len(d["run"].unique()), figsize=(4.2 * len(d["run"].unique()), 3.4), squeeze=False)
    for a, (run, g) in zip(ax[0], d.groupby("run", observed=True)):
        a.scatter(g["mean_sigma"], g["mae"], s=14, alpha=0.7, color="#3E6FD9")
        from scipy.stats import spearmanr
        r = spearmanr(g["mean_sigma"], g["mae"]).statistic
        a.set_title(f"{run}\nimage-level Spearman ρ = {r:.2f}")
        a.set_xlabel("mean predicted σ per image"); a.set_ylabel("MAE per image")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def ablation_bars(df, path):
    metrics = ["pearson", "ssim", "seg_f1"]
    g = df.groupby("run", observed=True)[metrics].agg(["mean", "std"])
    runs = list(g.index)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, m in zip(ax, metrics):
        a.bar(range(len(runs)), g[(m, "mean")], yerr=g[(m, "std")], color=["#9AA5A0", "#B8C0BB", "#7FA3F2", "#3E6FD9", "#12A36F"][: len(runs)], capsize=3)
        a.set_xticks(range(len(runs))); a.set_xticklabels(runs, rotation=20, ha="right", fontsize=8)
        a.set_title(m); a.set_ylim(min(0.0, g[(m, "mean")].min() - 0.05), 1.0)
        if m == "seg_f1" and "seg_f1_realfluo" in df:
            a.axhline(df["seg_f1_realfluo"].mean(), color="#C97A12", ls="--", lw=1, label="real fluorescence")
            a.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def uncertainty_bars(df, path):
    d = df[df["spearman_unc_err"].notna()]
    if d.empty:
        return
    g = d.groupby("run", observed=True)[["spearman_unc_err", "ause"]].agg(["mean", "std"])
    runs = list(g.index)
    names = {"baseline_unet_s0_tta": "TTA variance only\n(plain U-Net)", "chipstain_nll_s0": "ChipStain\nlearned σ", "chipstain_nll_s0_tta": "ChipStain + TTA\nlearned σ + epistemic"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for a, m, t in zip(ax, ["spearman_unc_err", "ause"], ["Spearman ρ(σ, |error|)  ↑ better", "AUSE  ↓ better"]):
        a.bar(range(len(runs)), g[(m, "mean")], yerr=g[(m, "std")], color=["#7FA3F2", "#3E6FD9", "#12A36F"][: len(runs)], capsize=3)
        a.set_xticks(range(len(runs))); a.set_xticklabels([names.get(r, r) for r in runs], fontsize=8)
        a.set_title(t)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def training_curves(path):
    import json
    runs = [f"runs/{r}/log.json" for r in ("baseline_unet_s0", "pretrained_l1_s0", "chipstain_nll_s0") if os.path.exists(f"runs/{r}/log.json")]
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    colors = {"baseline_unet_s0": "#9AA5A0", "pretrained_l1_s0": "#B8C0BB", "chipstain_nll_s0": "#3E6FD9"}
    for r in runs:
        name = os.path.basename(os.path.dirname(r))
        log = json.load(open(r))
        ep = [l["epoch"] for l in log]
        ax[0].plot(ep, [l["val_pearson"] for l in log], label=name, color=colors.get(name))
        ax[1].plot(ep, [l["val_ssim"] for l in log], label=name, color=colors.get(name))
    ax[0].set_title("validation Pearson r"); ax[1].set_title("validation SSIM")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/eval")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("outputs/eval*/per_image_test.csv"))])
    order = ["baseline_unet_s0", "pretrained_l1_s0", "baseline_unet_s0_tta", "chipstain_nll_s0", "chipstain_nll_s0_tta"]
    df["run"] = pd.Categorical(df["run"], [r for r in order if r in set(df["run"])], ordered=True)
    training_curves(os.path.join(a.out, "training_curves.png"))
    # qualitative grids used in the report (seed-0 checkpoint): full method, its single pass, and the TTA U-Net
    for tag in ["chipstain_nll_s0_tta", "chipstain_nll_s0", "baseline_unet_s0_tta"]:
        p = next((q for q in sorted(glob.glob(f"outputs/*/preds_{tag}.npz"))), None)
        if p:
            qualitative(load_preds(p), os.path.join(a.out, f"qualitative_{tag}.png"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
