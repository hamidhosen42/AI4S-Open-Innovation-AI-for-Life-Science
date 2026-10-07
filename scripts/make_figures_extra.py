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
CFG = ["baseline_unet", "pretrained_l1", "baseline_unet+tta", "pretrained_l1+tta", "chipstain_nll", "chipstain_nll+tta"]
NAME = {"baseline_unet": "Scratch U-Net (L1)", "pretrained_l1": "ImageNet-L1 U-Net",
        "baseline_unet+tta": "Scratch U-Net + TTA", "pretrained_l1+tta": "ImageNet-L1 U-Net + TTA (matched)", "chipstain_nll": "ChipStain (β-NLL)", "chipstain_nll+tta": "ChipStain + TTA (full)"}


def multiseed(res, path, grid=(1, 4), figsize=(11, 2.9)):
    ps = pd.read_csv(os.path.join(res, "multiseed_seed_means.csv"))
    panels = [("pearson", "Pearson r  (higher is better)"), ("seg_f1", "Nuclei seg-F1  (higher is better)"),
              ("spearman_unc_err", "ρ(σ, |error|)  (higher is better)"), ("ause", "AUSE  (lower is better)")]
    fig, axes = plt.subplots(*grid, figsize=figsize, sharey=True)
    axes = np.ravel(axes)
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
            ax.text(real, len(CFG) - 0.45, " real-stain reference (not a bound)", fontsize=7, color=INK2, va="bottom")
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(CFG) - 0.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    for ax in axes[::grid[1]]:
        ax.set_yticks(range(len(CFG)))
        ax.set_yticklabels([NAME[c] for c in CFG])
    fig.text(0.01, -0.02, "◇ mean of 3 seeds · ○ individual seeds · bar = ±1 s.d. across seeds · each seed = mean over 125 test images (well R05-C03)", fontsize=7.5, color=INK2)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def shift(res, path):
    df = pd.read_csv(os.path.join(res, "shift_test.csv"))
    if "cfg" not in df:  # legacy single-seed format
        df["cfg"] = df["run"].str.replace(r"_s\d+$", "", regex=True)
    order = [c for c in ["clean", "blur σ=1 px", "blur σ=2 px", "blur σ=4 px", "noise 10 %", "noise 20 %", "modality: DPC"] if c in set(df.condition)] or list(dict.fromkeys(df.condition))
    runs = [("chipstain_nll", "ChipStain + TTA (learned σ + TTA)", BLUE), ("pretrained_l1", "ImageNet-L1 U-Net + TTA (TTA spread)", AQUA), ("baseline_unet", "Scratch U-Net + TTA (TTA spread)", ORANGE)]
    runs = [r for r in runs if r[0] in set(df.cfg)]
    per_seed = df.groupby(["cfg", "run", "condition"])[["pearson", "mean_sigma"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    y = np.arange(len(order))[::-1]
    offs = np.linspace(0.2, -0.2, len(runs))
    for (cfg, lab, col), off in zip(runs, offs):
        d = per_seed[per_seed.cfg == cfg]
        clean = d[d.condition == "clean"].set_index("run")["mean_sigma"]
        d = d.assign(rel=d.apply(lambda r: r.mean_sigma / clean[r.run], axis=1))
        m = d.groupby("condition")[["pearson"]].mean().reindex(order)
        axes[0].scatter(m["pearson"], y + off, s=32, color=col, edgecolor="white", lw=1, zorder=3, label=lab)
        for c_i, c in enumerate(order):
            v = d[d.condition == c]
            axes[0].plot([v.pearson.min(), v.pearson.max()], [y[c_i] + off] * 2, color=col, lw=1, alpha=.6)
            # panel B: every seed as its own dot (a mean over seeds would hide seeds in which sigma falls)
            axes[1].scatter(v.rel, [y[c_i] + off] * len(v), s=22, facecolor=col, edgecolor="white", lw=0.6, zorder=3, alpha=.9)
    axes[0].set_title("A  Accuracy under shift: Pearson r vs real stain"); axes[0].set_xlim(0, 0.85)
    axes[1].set_xscale("log", base=2); axes[1].axvline(1, color=MUTED, lw=1, ls=(0, (3, 2)))
    axes[1].set_title("B  Does the uncertainty notice? mean σ relative to clean")
    lo_, hi_ = axes[1].get_xlim()
    ticks = [2.0 ** k for k in range(int(np.floor(np.log2(max(lo_, 1e-3)))), int(np.ceil(np.log2(hi_))) + 1, 2)]
    axes[1].set_xticks(ticks); axes[1].set_xticklabels([(f"{t:g}×" if t >= 1 else f"1/{1 / t:g}×") for t in ticks])
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_yticks(y); axes[0].set_yticklabels(order)
    h, l = axes[0].get_legend_handles_labels()
    n_seeds = df.groupby("cfg")["run"].nunique().max()
    fig.legend(h, l, loc="lower center", ncol=len(runs), frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.text(0.01, -0.12, f"A: dots = mean over {n_seeds} seeds, bars = range over seeds. B: one dot per seed (mean σ of 50 images relative to clean). 8× TTA.", fontsize=7.5, color=INK2)
    plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def nucleus(res, path):
    """Precision / recall / F1 of detections kept after discarding the highest-score nuclei."""
    f = os.path.join(res, "nucleus_gate.csv")
    if not os.path.exists(f):
        return
    g = pd.read_csv(f)
    g["model"] = g["run"].str.replace(r" s\d$", "", regex=True)
    m = g.groupby(["model", "score", "discard"])[["precision", "recall", "f1"]].mean().reset_index()
    series = [("ChipStain + TTA", "sigma_learned", BLUE, "-", "ChipStain σ"),
              ("U-Net + TTA", "sigma_tta_unet", ORANGE, "-", "U-Net TTA s.d."),
              ("ChipStain ensemble (3 seeds + TTA)", "sigma_ensemble", AQUA, "-", "ensemble σ"),
              ("ChipStain + TTA", "dim", INK2, (0, (3, 2)), "dim-nucleus score")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for model, score, col, ls, lab in series:
        d = m[(m.model == model) & (m.score == score)].sort_values("discard")
        if d.empty:
            continue
        axes[0].plot(d.discard * 100, d.precision, color=col, ls=ls, lw=2, marker="o", ms=3.5, label=lab)
        axes[1].plot(d.discard * 100, d.f1, color=col, ls=ls, lw=2, marker="o", ms=3.5, label=lab)
    axes[0].set_title("A  Precision of the detections kept")
    axes[1].set_title("B  F1 of the detections kept")
    for ax in axes:
        ax.set_xlabel("% of predicted nuclei discarded (highest score first)")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7.5)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _targets():
    import sys
    sys.path.insert(0, ".")
    from chipstain.data import PairDataset, make_splits
    _, _, test = make_splits("data/raw/hela_kyoto")
    ds = PairDataset(test, crop=None, augment=False, cache=False)
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x[0, :540, :540].numpy()); ys.append(y[0, :540, :540].numpy())
    return test, np.stack(xs), np.stack(ys)


def sparsification_all(cache, path, stride=2):
    """Mean sparsification curves over all 125 test images (seed 0): sigma vs proxies vs oracle."""
    import sys
    sys.path.insert(0, ".")
    from skimage import filters
    f_cs, f_un = os.path.join(cache, "chipstain_nll_s0_tta.npz"), os.path.join(cache, "baseline_unet_s0_tta.npz")
    if not (os.path.exists(f_cs) and os.path.exists(f_un)):
        return
    _, _, ys = _targets()
    cs, un = np.load(f_cs), np.load(f_un)
    fr = np.linspace(0, 0.95, 39)
    curves = {k: [] for k in ["σ ChipStain + TTA", "σ U-Net + TTA (view s.d.)", "predicted intensity μ", "edge strength |∇μ|", "oracle (true error)"]}
    for i in range(ys.shape[0]):
        mu = cs["mu"][i].astype(np.float32)
        err = np.abs(mu - ys[i])[::stride, ::stride].ravel()
        scores = {"σ ChipStain + TTA": cs["sigma"][i].astype(np.float32), "predicted intensity μ": np.clip(mu, 0, 1),
                  "edge strength |∇μ|": filters.sobel(filters.gaussian(np.clip(mu, 0, 1), 1.0)), "oracle (true error)": np.abs(mu - ys[i])}
        for k, sc in scores.items():
            order = np.argsort(-sc[::stride, ::stride].ravel(), kind="stable")
            e = err[order]
            curves[k].append([e[int(len(e) * f):].mean() / err.mean() for f in fr])
        mu_u = un["mu"][i].astype(np.float32)
        err_u = np.abs(mu_u - ys[i])[::stride, ::stride].ravel()
        e = err_u[np.argsort(-un["sigma"][i].astype(np.float32)[::stride, ::stride].ravel(), kind="stable")]
        curves["σ U-Net + TTA (view s.d.)"].append([e[int(len(e) * f):].mean() / err_u.mean() for f in fr])
    style = {"σ ChipStain + TTA": (BLUE, "-", 2.2), "σ U-Net + TTA (view s.d.)": (ORANGE, "-", 2), "predicted intensity μ": (INK2, (0, (4, 2)), 1.4),
             "edge strength |∇μ|": (AQUA, (0, (1, 1.5)), 1.8), "oracle (true error)": (MUTED, "-", 1.2)}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, c in curves.items():
        m = np.mean(c, 0)
        col, ls, lw = style[k]
        ax.plot(fr * 100, m, color=col, ls=ls, lw=lw)
        ax.text(fr[-1] * 100 + 1.5, m[-1], k, fontsize=7.5, color=INK if col not in (MUTED, INK2) else INK2, va="center")
    ax.axvline(20, color=GRID, lw=1)
    ax.set_xlim(0, 95); ax.set_ylim(0, max(1.05, max(np.mean(c, 0).max() for c in curves.values()) + 0.05))
    ax.set_xlabel("% of pixels removed, highest score first"); ax.set_ylabel("MAE of remaining pixels / MAE of all")
    ax.set_title("Which score finds the errors? (mean over 125 test images, seed 0)")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def failure_gallery(cache, path, n=4):
    """The worst test images by MAE across the ChipStain + TTA seeds."""
    import glob
    files = sorted(glob.glob(os.path.join(cache, "chipstain_nll_s*_tta.npz")))
    if not files:
        return
    test, xs, ys = _targets()
    cand = []
    for f in files:
        z = np.load(f)
        seed = os.path.basename(f).split("_s")[1][0]
        for i in range(ys.shape[0]):
            cand.append((float(np.abs(z["mu"][i].astype(np.float32) - ys[i]).mean()), f, int(seed), i))
    cand.sort(reverse=True)
    picked, seen = [], set()
    for c in cand:
        if (c[3]) not in seen:
            picked.append(c); seen.add(c[3])
        if len(picked) == n:
            break
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.3 * n))
    titles = ["bright-field input", "real H2B", "ChipStain prediction", "|error|", "predicted σ"]
    for r, (mae, f, seed, i) in enumerate(picked):
        z = np.load(f)
        mu, sg = z["mu"][i].astype(np.float32), z["sigma"][i].astype(np.float32)
        ims = [xs[i], ys[i], np.clip(mu, 0, 1), np.abs(mu - ys[i]), sg]
        cm = ["gray", "magma", "magma", "inferno", "viridis"]
        for c in range(5):
            kw = {"vmin": 0, "vmax": 1} if c in (1, 2) else ({"vmin": 0, "vmax": 0.4} if c == 3 else {})
            axes[r, c].imshow(ims[c], cmap=cm[c], **kw); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=8.5)
        axes[r, 0].set_ylabel(f"seed {seed} · F{test[i].field} t{test[i].timepoint}\nMAE {mae:.3f}", fontsize=8)
    fig.text(0.01, -0.01, "Data: R. Guiet, EPFL BIOP, Zenodo 10.5281/zenodo.6140064, CC BY 4.0", fontsize=7, color=INK2)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def proliferation_fig(res, path):
    f = os.path.join(res, "proliferation_curves.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    cols = {"StarDist reference (real H2B)": (INK2, (0, (3, 2))), "ChipStain + TTA": (BLUE, "-"), "U-Net + TTA": (ORANGE, "-")}
    for name, (col, ls) in cols.items():
        g = d[d["source"] == name]
        if g.empty:
            continue
        m = g.groupby("hours")["count"].mean()
        axes[0].plot(m.index, m.values, color=col, ls=ls, lw=2, marker="o", ms=4, label=name)
    axes[0].set_yscale("log"); axes[0].set_xlabel("hours"); axes[0].set_ylabel("nuclei per field (mean of 25 fields)")
    axes[0].set_title("A  Label-free growth curves (test well)"); axes[0].legend(frameon=False, fontsize=7.5)
    pf = os.path.join(res, "proliferation_per_field.csv")
    if os.path.exists(pf):
        p = pd.read_csv(pf, index_col=0)
        ref = p["reference"]
        for name, col in [("ChipStain + TTA", BLUE), ("U-Net + TTA", ORANGE)]:
            cs = [c for c in p.columns if c.startswith(name)]
            for k, c in enumerate(cs):
                axes[1].scatter(ref, p[c], s=14, color=col, alpha=0.75, edgecolor="white", lw=0.5, label=name if k == 0 else None)
        lo, hi = np.nanpercentile(ref, 1) * 0.7, np.nanpercentile(p.drop(columns="reference").values, 99) * 1.1
        axes[1].plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (3, 2)))
        axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, min(hi, 80))
        axes[1].set_xlabel("doubling time from reference counts (h)"); axes[1].set_ylabel("from label-free counts (h)")
        axes[1].set_title("B  Per-field doubling time, all seeds"); axes[1].legend(frameon=False, fontsize=7.5)
    for ax in axes:
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def timelapse_fig(res, path, strip_path):
    f = os.path.join(res, "timelapse.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    axes[0].plot(d.hours, d.n_real, color=INK2, lw=1.4, ls=(0, (3, 2)), label="counted on the real H2B stain")
    axes[0].plot(d.hours, d.n_pred, color=BLUE, lw=1.8, label="counted on the label-free prediction")
    axes[0].set_ylabel("nuclei in field"); axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_title("60 h of nuclear read-out from bright-field only (field R05-C03-F0, a frame every 15 min)")
    axes[1].plot(d.hours, d.mean_sigma, color=BLUE, lw=1.4)
    axes[1].set_ylabel("mean σ"); axes[1].set_xlabel("hours")
    for ax in axes:
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    fz = "outputs/timelapse_frames.npz"
    if os.path.exists(fz):
        z = np.load(fz)
        ts = sorted({int(k.split("_")[1]) for k in z.files})
        fig, axes = plt.subplots(3, len(ts), figsize=(2.2 * len(ts), 6.8))
        for c, t in enumerate(ts):
            bf = z[f"bf_{t}"]
            lo, hi = np.percentile(bf, [1, 99])
            axes[0, c].imshow(bf, cmap="gray", vmin=lo, vmax=hi); axes[0, c].set_title(f"{(t - 1) * 0.25:.0f} h", fontsize=9)
            axes[1, c].imshow(np.clip(z[f"pred_{t}"], 0, 1), cmap="magma", vmin=0, vmax=1)
            axes[2, c].imshow(z[f"sigma_{t}"], cmap="viridis", vmin=0, vmax=np.percentile(z[f"sigma_{ts[-1]}"], 99))
            for r in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        for r, lab in enumerate(["bright-field", "predicted H2B", "σ"]):
            axes[r, 0].set_ylabel(lab, fontsize=9)
        fig.text(0.01, -0.01, "Data: R. Guiet, EPFL BIOP, Zenodo 10.5281/zenodo.6139958, CC BY 4.0", fontsize=7, color=INK2)
        plt.tight_layout(); plt.savefig(strip_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def neural_fig(res, path):
    f = os.path.join(res, "neural", "finetune.csv")
    z = os.path.join(res, "neural", "zeroshot.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for init, col in [("from ChipStain (HeLa)", BLUE), ("from ImageNet", ORANGE)]:
        g = d[d["init"] == init].groupby("k_wells")[["pearson", "seg_f1_vs_real", "mean_sigma"]].mean()
        for ax, m in zip(axes, ["pearson", "seg_f1_vs_real", "mean_sigma"]):
            ax.plot(g.index, g[m], marker="o", color=col, lw=2, label=init)
    if os.path.exists(z):
        zz = pd.read_csv(z).groupby("model")[["pearson", "seg_f1_vs_real", "mean_sigma"]].mean()
        for ax, m in zip(axes, ["pearson", "seg_f1_vs_real", "mean_sigma"]):
            for name, ls in [("ChipStain (HeLa, zero-shot)", (0, (3, 2)))]:
                if name in zz.index:
                    ax.axhline(zz.loc[name, m], color=BLUE, ls=ls, lw=1)
                    ax.text(ax.get_xlim()[0] if False else 1, zz.loc[name, m], " zero-shot", fontsize=7, color=INK2, va="bottom")
    for ax, t in zip(axes, ["Pearson r vs real nuclear stain", "nuclei F1 vs real-stain segmentation", "mean σ (test wells)"]):
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 20]); ax.set_xticklabels(["1", "2", "5", "20"])
        ax.set_xlabel("training wells used for fine-tuning"); ax.set_title(t)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7.5)
    plt.tight_layout(); plt.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def neural_examples(path, grid=(1, 6), figsize=(13, 2.6)):
    fz, ff = "outputs/neural_example_zeroshot.npz", "outputs/neural_example_finetuned.npz"
    if not (os.path.exists(fz) and os.path.exists(ff)):
        return
    z, f = np.load(fz), np.load(ff)
    smax = float(np.percentile(np.concatenate([z["sigma"].ravel(), f["sigma"].ravel()]).astype(np.float32), 99))
    panels = [(z["bf"], "bright-field (neurons)", "gray", None), (z["dapi"], "real nuclear stain (Hoechst)", "magma", (0, 1)),
              (z["pred"], "zero-shot prediction", "magma", (0, 1)), (z["sigma"], "zero-shot σ", "viridis", (0, smax)),
              (f["pred"], "fine-tuned (20 wells)", "magma", (0, 1)), (f["sigma"], "fine-tuned σ", "viridis", (0, smax))]
    if grid == (2, 3):  # rows: input / real stain, zero-shot / fine-tuned prediction, zero-shot / fine-tuned sigma
        panels = [panels[i] for i in (0, 2, 3, 1, 4, 5)]
    fig, ax = plt.subplots(*grid, figsize=figsize)
    for a, (im, t, cm, rng) in zip(np.ravel(ax), panels):
        im = im.astype(np.float32)
        kw = {"vmin": rng[0], "vmax": rng[1]} if rng else {"vmin": np.percentile(im, 1), "vmax": np.percentile(im, 99)}
        a.imshow(np.clip(im, 0, 1) if cm == "magma" else im, cmap=cm, **kw); a.set_title(t, fontsize=8.5 if grid[0] == 1 else 11); a.set_xticks([]); a.set_yticks([])
    fig.text(0.01, -0.03, "Data: Christiansen et al., Cell 2018, in-silico-labeling Condition A, CC BY 4.0 (rescaled, normalised, colour-mapped). σ panels share one colour scale.", fontsize=7, color=INK2)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="report/results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    multiseed(a.res, os.path.join(a.out, "multiseed.png"))
    shift(a.res, os.path.join(a.out, "shift_test.png"))
    nucleus(a.res, os.path.join(a.out, "nucleus_gate.png"))
    sparsification_all("outputs/cache", os.path.join(a.out, "uncertainty_baselines.png"))
    failure_gallery("outputs/cache", os.path.join(a.out, "failure_gallery.png"))
    proliferation_fig(a.res, os.path.join(a.out, "proliferation.png"))
    timelapse_fig(a.res, os.path.join(a.out, "timelapse.png"), os.path.join(a.out, "timelapse_strip.png"))
    neural_fig(a.res, os.path.join(a.out, "neural_finetune.png"))
    neural_examples(os.path.join(a.out, "neural_examples.png"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
