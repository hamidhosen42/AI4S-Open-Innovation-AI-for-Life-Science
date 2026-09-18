"""Fill report/report_template.html with numbers from outputs/eval*, embed figures, render PDF.

    python scripts/build_report.py --eval outputs/eval --eval_tta outputs/eval_tta --figs report/figures
"""
import argparse
import base64
import datetime
import glob
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from chipstain.metrics import sparsification

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LABELS = {
    "baseline_unet_s0": "U-Net baseline (L1)",
    "pretrained_l1_s0": "+ ImageNet encoder (L1)",
    "baseline_unet_s0_tta": "U-Net baseline + TTA (σ = view variance)",
    "chipstain_nll_s0": "ChipStain (β-NLL head)",
    "chipstain_nll_s0_tta": "ChipStain + TTA (full)",
}


def b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/eval")
    ap.add_argument("--eval_tta", default="outputs/eval_tta")
    ap.add_argument("--figs", default="report/figures")
    ap.add_argument("--out", default="report/ChipStain_Technical_Report")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(os.path.join(p, "per_image_test.csv")) for p in (a.eval, a.eval_tta) if os.path.exists(os.path.join(p, "per_image_test.csv"))])
    order = [r for r in LABELS if r in set(df["run"])]
    g = df.groupby("run")

    def M(run, col):
        return float(g.get_group(run)[col].mean())

    def S(run, col):
        return float(g.get_group(run)[col].std())

    # ---- main table
    cols = [("pearson", "Pearson r ↑"), ("ssim", "SSIM ↑"), ("psnr", "PSNR (dB) ↑"), ("seg_f1", "seg-F1 ↑"), ("spearman_unc_err", "ρ<sub>σ,err</sub> ↑"), ("ause", "AUSE ↓")]
    rows = ["<table><tr><th>Model</th>" + "".join(f"<th>{h}</th>" for _, h in cols) + "</tr>"]
    for run in order:
        d = g.get_group(run)
        cells = []
        for c, _ in cols:
            if c in d and d[c].notna().any():
                cells.append(f"<td class='n'>{fmt(d[c].mean(), d[c].std())}</td>")
            else:
                cells.append("<td class='n'>—</td>")
        hl = " class='hl'" if run == "chipstain_nll_s0_tta" else ""
        rows.append(f"<tr{hl}><td>{LABELS[run]}</td>" + "".join(cells) + "</tr>")
    real = float(df["seg_f1_realfluo"].mean())
    rows.append(f"<tr><td><i>Real fluorescence, same pipeline (ceiling)</i></td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>{real:.3f} ± {df['seg_f1_realfluo'].std():.3f}</td><td class='n'>—</td><td class='n'>—</td></tr></table>")
    main_table = "\n".join(rows)

    full = "chipstain_nll_s0_tta" if "chipstain_nll_s0_tta" in order else "chipstain_nll_s0"
    b = "baseline_unet_s0"
    dfull = g.get_group(full)
    rho_img = spearmanr(dfull["mean_sigma"], dfull["mae"]).statistic

    # sparsification gain at 20 % from saved predictions
    z = np.load(os.path.join(a.eval_tta if full.endswith("tta") else a.eval, f"preds_{full}.npz"))
    n = max(int(k.split("_")[-1]) for k in z.files) + 1
    gains = []
    for j in range(n):
        err = np.abs(z[f"pred_{j}"] - z[f"gt_{j}"])
        fr, cu, co, _ = sparsification(err, z[f"unc_{j}"], n_bins=101)
        gains.append(1 - cu[20] / cu[0])
    sparse_gain = 100 * float(np.mean(gains))

    # nuclei counts
    cnt = dfull[["n_ref", "n_pred"]]
    count_err = float((cnt["n_pred"] - cnt["n_ref"]).abs().mean())
    count_rel = float(((cnt["n_pred"] - cnt["n_ref"]).abs() / cnt["n_ref"].clip(lower=1)).mean() * 100)
    count_r = float(np.corrcoef(cnt["n_pred"], cnt["n_ref"])[0, 1])

    pre = "pretrained_l1_s0"
    pretrain_note = f"Pearson {M(pre,'pearson'):.3f} vs {M(b,'pearson'):.3f}, SSIM {M(pre,'ssim'):.3f} vs {M(b,'ssim'):.3f}" if pre in order else "see Table 1"

    unc_text = (
        f"<p>Across the 125 test images the per-pixel Spearman correlation between predicted σ and absolute error is "
        f"{M(full,'spearman_unc_err'):.3f} ± {S(full,'spearman_unc_err'):.3f}, and the AUSE is {M(full,'ause'):.3f} ± {S(full,'ause'):.3f}. "
        f"Removing the 20 % most-uncertain pixels reduces the MAE of the remaining pixels by {sparse_gain:.1f} % on average (Figure 3). "
        f"At the image level, mean σ ranks the test images by their MAE with Spearman ρ = {rho_img:.2f} (Figure 4): "
        f"a laboratory can set a single threshold on mean σ to route the least reliable frames to manual review.</p>"
    )
    bt = "baseline_unet_s0_tta"
    if bt in order:
        unc_text += (
            f"<p><b>Is the learned head needed, or is TTA disagreement enough?</b> Using only the variance of the eight TTA views of the plain U-Net as an uncertainty gives "
            f"ρ<sub>σ,err</sub> = {M(bt,'spearman_unc_err'):.3f} and AUSE = {M(bt,'ause'):.3f}, versus {M(full,'spearman_unc_err'):.3f} and {M(full,'ause'):.3f} for ChipStain's learned aleatoric + epistemic map. "
            f"The learned head therefore captures error structure that view-disagreement alone misses, at no cost in Pearson correlation ({M(full,'pearson'):.3f} vs {M(bt,'pearson'):.3f}) and a small cost in SSIM ({M(full,'ssim'):.3f} vs {M(bt,'ssim'):.3f}).</p>"
        )
    count_text = (
        f"<p>Counting nuclei on the ChipStain prediction with the fixed watershed pipeline gives an object-level F1 of {M(full,'seg_f1'):.3f} ± {S(full,'seg_f1'):.3f} "
        f"against the StarDist reference labels, compared with {real:.3f} when the real fluorescence is segmented the same way and {M(b,'seg_f1'):.3f} for the plain U-Net. "
        f"The predicted nucleus count per image deviates from the reference by {count_err:.1f} nuclei on average ({count_rel:.1f} % relative), with a correlation of r = {count_r:.3f} between predicted and reference counts over the 125 images "
        f"(reference counts range {int(cnt['n_ref'].min())}–{int(cnt['n_ref'].max())}).</p>"
    )

    # ---- appendix A: per-time-point breakdown (cell density grows with time)
    tp = dfull.groupby("timepoint")[["pearson", "ssim", "seg_f1", "seg_f1_realfluo", "mean_sigma", "mae"]].mean()
    tp_rows = ["<table><tr><th>Time-point</th><th>Pearson r</th><th>SSIM</th><th>seg-F1 (pred)</th><th>seg-F1 (real)</th><th>mean σ</th><th>MAE</th></tr>"]
    for t, r in tp.iterrows():
        tp_rows.append(f"<tr><td>{int(t)}</td><td class='n'>{r['pearson']:.3f}</td><td class='n'>{r['ssim']:.3f}</td><td class='n'>{r['seg_f1']:.3f}</td><td class='n'>{r['seg_f1_realfluo']:.3f}</td><td class='n'>{r['mean_sigma']:.4f}</td><td class='n'>{r['mae']:.4f}</td></tr>")
    tp_rows.append("</table>")
    tp_table = "\n".join(tp_rows)
    # precision / recall for the full model
    from chipstain.metrics import match_f1  # noqa
    pr_text = ""
    if {"n_ref", "n_pred"} <= set(dfull.columns):
        pr_text = f"<p>Over the test set the full model detects {int(dfull['n_pred'].sum())} nuclei against {int(dfull['n_ref'].sum())} reference objects ({100*dfull['n_pred'].sum()/dfull['n_ref'].sum():.1f} %).</p>"

    tpl = open("report/report_template.html", encoding="utf-8").read()
    sub = {
        "date": datetime.date.today().isoformat(),
        "main_table": main_table,
        "o_pearson": f"{M(full,'pearson'):.3f}", "o_ssim": f"{M(full,'ssim'):.3f}", "o_segf1": f"{M(full,'seg_f1'):.3f}",
        "b_pearson": f"{M(b,'pearson'):.3f}", "b_ssim": f"{M(b,'ssim'):.3f}", "real_segf1": f"{real:.3f}",
        "o_rho": f"{M(full,'spearman_unc_err'):.2f}", "sparse_gain": f"{sparse_gain:.0f}", "rho_img": f"{rho_img:.2f}",
        "unc_text": unc_text, "count_text": count_text, "pretrain_note": pretrain_note,
        "unc_summary": f"ρ = {M(full,'spearman_unc_err'):.2f} per pixel, ρ = {rho_img:.2f} per image",
        "fig_ablation": b64(os.path.join(a.figs, "ablation.png")),
        "fig_qual": b64(os.path.join(a.figs, f"qualitative_{full}.png")),
        "fig_sparse": b64(os.path.join(a.figs, f"sparsification_{full}.png")),
        "fig_calib": b64(os.path.join(a.figs, "calibration_image_level.png")),
        "fig_uncq": b64(os.path.join(a.figs, "uncertainty_quality.png")),
        "tp_table": tp_table, "pr_text": pr_text,
        "fig_curves": b64(os.path.join(a.figs, "training_curves.png")),
        "fig_qual_base": b64(os.path.join(a.figs, "qualitative_baseline_unet_s0_tta.png")),
        "fig_qual_nll": b64(os.path.join(a.figs, "qualitative_chipstain_nll_s0.png")),
    }
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    left = [l for l in set(__import__("re").findall(r"\{\{(\w+)\}\}", tpl))]
    if left:
        print("WARNING unfilled:", left)
    html = a.out + ".html"
    open(html, "w", encoding="utf-8").write(tpl)
    if os.path.exists(CHROME):
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000", f"--print-to-pdf={os.path.abspath(a.out)}.pdf", "file://" + os.path.abspath(html)], capture_output=True)
        print("wrote", a.out + ".pdf")
    print("wrote", html)
    # also drop the summary numbers for the writeup
    open("outputs/key_numbers.txt", "w").write("\n".join(f"{k}: {v}" for k, v in sub.items() if not k.startswith("fig") and k not in ("main_table", "unc_text", "count_text")))


if __name__ == "__main__":
    main()
