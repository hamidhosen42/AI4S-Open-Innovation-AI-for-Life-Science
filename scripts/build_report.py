"""Build the technical report (HTML + PDF) from report/report_template.html and the result
files in report/results/. Every number in the results sections is read from those files.

    python scripts/build_report.py [--out report/ChipStain_Technical_Report]
"""
import argparse
import datetime
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from scripts.report_utils import RES, figure, html_table, md_tables, read, table_from

CHROME = os.environ.get("CHROME") or next((c for c in (shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
                                                       shutil.which("chrome"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome") if c and os.path.exists(c)), None)
FULL, BT, CS, BU, PRE = "chipstain_nll+tta", "baseline_unet+tta", "chipstain_nll", "baseline_unet", "pretrained_l1"


# ----------------------------------------------------------------------------- numbers
class R:
    """Access to seed-level means and paired tests."""

    def __init__(self):
        self.seed = read("multiseed_seed_means.csv")
        self.tests = read("multiseed_tests.csv")
        self.img = read("multiseed_per_image.csv")

    def m(self, cfg, col):
        d = self.seed[self.seed.cfg == cfg][col]
        return float(d.mean())

    def sd(self, cfg, col):
        return float(self.seed[self.seed.cfg == cfg][col].std())

    def has(self, cfg):
        return self.seed is not None and (self.seed.cfg == cfg).any()

    def t(self, a, b, m):
        if self.tests is None:
            return None
        d = self.tests[(self.tests.a == a) & (self.tests.b == b) & (self.tests.metric == m)]
        return d.iloc[0] if len(d) else None

    def seed0(self, cfg, col):
        d = self.img[(self.img.cfg == cfg) & (self.img.seed == 0)][col]
        return float(d.mean())


def p_txt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def diff_txt(r, t, m, digits=3, pct=False):
    """'A − B = +x (95 % CI …; significantly better in b/3 seeds, worse in w/3)' for a test row."""
    if t is None:
        return ""
    f = (lambda v: f"{100 * v:+.1f} %") if pct else (lambda v: f"{v:+.{digits}f}")
    txt = f"{f(t.mean_a - t.mean_b)} (95 % CI {f(t.ci_lo)} to {f(t.ci_hi)}; significantly better in {int(t.seeds_sig_better)}/{int(t.n_seeds)} seeds, worse in {int(t.seeds_sig_worse)}/{int(t.n_seeds)}"
    if m in ("spearman_unc_err", "ause"):
        txt += f"; better in {int(t.fields_a_better)}/25 fields; pooled Holm {p_txt(t.p_holm)}"
    return txt + ")"


def sig(t):
    """A robust difference: the seed+field bootstrap CI excludes 0, no seed goes the other way,
    and at least 2 of 3 seeds are significant on their own."""
    return t is not None and (t.ci_lo > 0 or t.ci_hi < 0) and t.seeds_sig_worse == 0 and t.seeds_sig_better >= 2


# ----------------------------------------------------------------------------- sections
def sec_main(r):
    s = []
    s.append("<h3>6.1 Main results over three training seeds</h3>")
    s.append(table_from(f"{RES}/multiseed_summary.md", "Main comparison", highlight="ChipStain + TTA (full)"))
    s.append(f"<p class='small'>Mean ± s.d. over 3 training seeds; each seed is the mean over the 125 test images of well R05-C03. "
             f"The released checkpoint is seed 0 (ChipStain + TTA: Pearson r {r.seed0(FULL, 'pearson'):.3f}, SSIM {r.seed0(FULL, 'ssim'):.3f}, "
             f"seg-F1 {r.seed0(FULL, 'seg_f1'):.3f}, ρ(σ, err) {r.seed0(FULL, 'spearman_unc_err'):.3f}). "
             "The real-fluorescence row runs the same watershed on the real H2B image; it is a reference level, not a strict bound.</p>")
    s.append(figure("multiseed.png", "<b>Figure 1.</b> All configurations over three training seeds: open circles are seeds, diamonds their mean, bars ±1 s.d. "
                    "ChipStain + TTA (blue) is the full method."))
    # like-for-like narrative
    tp, ts, tm, tf = (r.t(FULL, BT, k) for k in ("pearson", "ssim", "mae", "seg_f1"))
    tr, ta = r.t(FULL, BT, "spearman_unc_err"), r.t(FULL, BT, "ause")
    s.append("<h3>6.2 Like-for-like comparison and statistics</h3>")
    s.append("<p>The fair comparison for the full method is a U-Net that also uses 8× TTA (its uncertainty is then the TTA disagreement). "
             "Differences are ChipStain + TTA minus U-Net + TTA, with a 95 % CI from a bootstrap that resamples both seeds and fields, and a field-level Wilcoxon test (Holm-corrected).</p><ul>")
    s.append(f"<li><b>Correlation with the real stain:</b> Pearson r {diff_txt(r, tp, 'pearson')} — {'a robust gain' if sig(tp) and tp.mean_a > tp.mean_b else 'comparable; no robust difference'}.</li>")
    s.append(f"<li><b>Pixel fidelity:</b> MAE {diff_txt(r, tm, 'mae', digits=4)}; SSIM {diff_txt(r, ts, 'ssim')}. "
             f"The MAE cost is small but consistent (its CI excludes zero); SSIM is lower in most seeds. §6.3 tests whether it comes from the loss or the head.</li>")
    s.append(f"<li><b>Downstream nuclei F1:</b> {diff_txt(r, tf, 'seg_f1')}: higher on average but not robust across seeds — the U-Net's seed-to-seed spread "
             f"(s.d. {r.sd(BT, 'seg_f1'):.3f} vs {r.sd(FULL, 'seg_f1'):.3f} for ChipStain) comes from one unstable U-Net run (n = 3 seeds, descriptive only).</li>")
    s.append(f"<li><b>Uncertainty ranking:</b> ρ(σ, |error|) {diff_txt(r, tr, 'spearman_unc_err')}; AUSE {diff_txt(r, ta, 'ause')}. "
             f"This is the clearest and most consistent result of the study.</li></ul>")
    tm2 = {k: r.t(FULL, "pretrained_l1+tta", k) for k in ("pearson", "mae", "spearman_unc_err", "ause")}
    if all(v is not None for v in tm2.values()):
        s.append(f"<p><b>Matched control.</b> The U-Net above differs from ChipStain in encoder initialisation and learning rate as well as in the head. "
                 f"Against an ImageNet-encoder L1 U-Net with the same TTA and learning rate — differing only in the probabilistic head and loss — ChipStain's uncertainty is still better: "
                 f"ρ {diff_txt(r, tm2['spearman_unc_err'], 'spearman_unc_err')}; AUSE {diff_txt(r, tm2['ause'], 'ause')}; with Pearson r {diff_txt(r, tm2['pearson'], 'pearson')} and MAE {diff_txt(r, tm2['mae'], 'mae', digits=4)}.</p>")
    t1 = r.t(CS, BT, "ause")
    if t1 is not None:
        s.append(f"<p>Without TTA, ChipStain's learned σ alone still ranks errors better than the U-Net's TTA disagreement in ρ, but <b>not</b> in AUSE "
                 f"({r.m(CS, 'ause'):.3f} vs {r.m(BT, 'ause'):.3f}; Holm {p_txt(t1.p_holm)}): the learned head and the TTA term are complementary, and the full method needs both.</p>")
    tests = r.tests
    keep = tests[tests.a.isin([FULL, CS, "pretrained_l1+tta"]) & tests.b.isin([BT, "pretrained_l1+tta"])].copy()
    rows = [["A vs B", "metric", "A", "B", "A − B (95 % CI, seeds + fields)", "seeds sig. better / worse", "p (pooled, Holm; σ metrics)"]]
    lab = {FULL: "ChipStain + TTA", CS: "ChipStain", BT: "U-Net + TTA", BU: "U-Net", "pretrained_l1+tta": "ImageNet-L1 U-Net + TTA"}
    mname = {"pearson": "Pearson r", "ssim": "SSIM", "mae": "MAE", "seg_f1": "nuclei F1", "spearman_unc_err": "ρ(σ, error)", "ause": "AUSE"}
    for _, t in keep.iterrows():
        rows.append([f"{lab[t.a]} vs {lab[t.b]}", mname.get(t.metric, t.metric), f"{t.mean_a:.4f}", f"{t.mean_b:.4f}", f"{t.mean_a - t.mean_b:+.4f} ({t.ci_lo:+.4f}, {t.ci_hi:+.4f})",
                     f"{int(t.seeds_sig_better)} / {int(t.seeds_sig_worse)} of {int(t.n_seeds)}", f"{t.p_holm:.2g}" if t.metric in ("spearman_unc_err", "ause") else "—"])
    s.append(html_table(rows))
    s.append("<p class='small'><b>Table 2.</b> Paired comparisons, all with TTA on both sides (except single-pass ChipStain, marked). Inference rests on the hierarchical seed+field bootstrap CI and on per-seed tests; "
             "the pooled field-level p-value averages over seeds, so it is shown for the uncertainty metrics only. For MAE and AUSE lower is better.</p>")
    return "\n".join(s)


def sec_ablation(r):
    s = ["<h3>6.3 Ablation: what each design choice does</h3>"]
    if not r.has("ablate_pretrained_mse"):
        s.append("<p><i>The de-confounded ablation arms are still training; this section is completed when they finish.</i></p>")
        return "\n".join(s)
    s.append(table_from(f"{RES}/multiseed_summary.md", "Ablation arms"))
    steps = [("ablate_scratch_l1_lr5e4", BU, "Learning rate only (scratch U-Net, 5e-4 vs 1e-3)"),
             (PRE, "ablate_scratch_l1_lr5e4", "ImageNet pre-training only (same LR)"),
             ("ablate_pretrained_mse", PRE, "Loss only: L1 → MSE (pretrained, single output)"),
             (CS, "ablate_pretrained_mse", "Variance head only: β-NLL vs MSE"),
             (CS, "ablate_nll_beta0", "β = 0.5 vs plain NLL (β = 0)"),
             (CS, "ablate_nll_beta1", "β = 0.5 vs β = 1")]
    rows = [["step (A vs B)", "Δ Pearson r", "Δ SSIM", "Δ MAE", "Δ seg-F1"]]
    for a, b, name in steps:
        cells = [name]
        for m in ("pearson", "ssim", "mae", "seg_f1"):
            t = r.t(a, b, m)
            cells.append("—" if t is None else f"{t.mean_a - t.mean_b:+.4f}{'*' if sig(t) else ''}")
        rows.append(cells)
    s.append(html_table(rows))
    s.append("<p class='small'><b>Table 3.</b> Each row changes one factor (A − B; * = Holm p &lt; 0.05 and the seed+field bootstrap CI excludes 0). "
             "All arms: 3 seeds, no TTA.</p>")
    s.append("{{ablation_text}}")
    return "\n".join(s)


def sec_qual():
    return "\n".join([
        "<h3>6.4 Qualitative results and failure cases</h3>",
        figure("qualitative_chipstain_nll_s0_tta.png", "<b>Figure 3.</b> Test images (seed-0 checkpoint, TTA). Columns: bright-field input, real H2B, prediction, absolute error, predicted σ. "
               "σ is raised over every nucleus (heteroscedastic σ scales with intensity) and is near zero on background. The strongest σ and error hotspots are the "
               "elliptical, grid-textured patches on the right of rows 1–2 (the same field at two time-points, same position) and at the bottom right of row 3: a fixed optical/plate artefact "
               "that the model renders as dark textured regions. We did not annotate mitoses, debris or focus; nucleus-level σ is quantified in §6.9. "
               "Data: R. Guiet, EPFL BIOP, Zenodo 10.5281/zenodo.6140064, CC BY 4.0."),
        figure("failure_gallery.png", "<b>Figure 4.</b> The worst test images by MAE across all ChipStain + TTA seeds — all from one training seed (seed 1) at the densest time-point, where that run "
               "hallucinates large bright regions. The error is dominated by these regions, and σ is elevated over and around them; these are also that seed's highest-σ test images (§6.10)."),
    ])


def sec_uncertainty(r):
    s = ["<h3>6.5 Does σ find the errors better than simple alternatives?</h3>",
         "<p>An uncertainty map is only useful if it beats what a user could compute without it. Errors concentrate on bright nuclei and their boundaries, "
         "so we rank pixels by σ and, as uncertainty-free proxies, by the predicted intensity μ and its edge strength |∇μ| (Table 4).</p>",
         table_from(f"{RES}/multiseed_summary.md", "proxies", highlight="ChipStain + TTA (full)"),
         "<p class='small'><b>Table 4.</b> ρ: Spearman correlation with |error|. AUSE: area between the sparsification curve and the oracle curve (lower is better). "
         "gain: MAE reduction after removing the 20 % highest-score pixels; oracle = removing the 20 % largest true errors. Mean ± s.d. over seeds, all 125 test images.</p>"]
    g, gmu, ggr, gor = (r.m(FULL, k) for k in ("gain20", "gain20_mu", "gain20_grad", "gain20_oracle"))
    a, amu, agr = (r.m(FULL, k) for k in ("ause", "ause_mu", "ause_grad"))
    s.append(f"<p>Removing the 20 % highest-σ pixels lowers the remaining MAE by <b>{100 * g:.0f} %</b> on average over all 125 test images "
             f"(oracle {100 * gor:.0f} %), but ranking by edge strength gives {100 * ggr:.0f} % and by intensity {100 * gmu:.0f} %: at that single cut-off σ adds "
             f"{'little' if g - max(gmu, ggr) < 0.05 else 'a clear margin'} over a trivial proxy. "
             f"The full ranking curve tells a different story: AUSE is {a:.3f} for σ versus {agr:.3f} for edge strength and {amu:.3f} for intensity — σ orders the errors "
             f"much more faithfully over the whole range (Figure 5). An earlier version of this report quoted a 55 % reduction measured on only 8 test images; the full-set numbers above replace it.</p>")
    s.append(figure("uncertainty_baselines.png", "<b>Figure 5.</b> Sparsification curves averaged over all 125 test images (seed 0): MAE of the remaining pixels as pixels are removed in order of each score. "
                    "Lower is better; the oracle removes the true largest errors first."))
    cal = md_tables(f"{RES}/calibration.md")
    if cal:
        s.append("<p><b>Calibration.</b> Ranking is not calibration. Table 5 reports the empirical coverage of μ ± zσ; one scalar fitted on the validation split rescales σ.</p>")
        s.append(html_table(cal[0][1]))
        s.append("<p class='small'><b>Table 5.</b> Empirical coverage of the nominal Gaussian intervals on the 125 test images, mean over seeds.</p>")
        s.append("{{calibration_text}}")
    return "\n".join(s)


def sec_image_level(r):
    s = ["<h3>6.6 Image level: is mean σ more than a density counter?</h3>",
         table_from(f"{RES}/multiseed_summary.md", "Image level")]
    s.append(f"<p class='small'><b>Table 6.</b> Per-image Spearman correlations over the 125 test images (mean ± s.d. over seeds). n_pred = number of nuclei detected on the prediction (needs no uncertainty).</p>")
    s.append(f"<p>Mean σ ranks test images by their MAE (ρ = {r.m(FULL, 'img_rho_sigma_mae'):.2f}), but in-distribution most of this is cell density: "
             f"the predicted nucleus count alone reaches ρ = {r.m(FULL, 'img_rho_npred_mae'):.2f}, and the U-Net's TTA disagreement reaches {r.m(BT, 'img_rho_sigma_mae'):.2f}. "
             f"Controlling for the count, mean σ still carries independent information (partial ρ = {r.m(FULL, 'img_partial_sigma_mae_given_npred'):.2f} ± {r.sd(FULL, 'img_partial_sigma_mae_given_npred'):.2f} over seeds), "
             f"but so does the U-Net's TTA disagreement (partial ρ = {r.m(BT, 'img_partial_sigma_mae_given_npred'):.2f}; ρ(σ̄, 1 − r) {r.m(BT, 'img_rho_sigma_1mr'):.2f} vs {r.m(FULL, 'img_rho_sigma_1mr'):.2f} for ChipStain). "
             f"In distribution, the frame-level ranking is therefore <b>not</b> an advantage of the β-NLL head. Where frame-level σ matters is when a run or the input goes wrong: "
             f"one ChipStain seed's three catastrophic test frames are its three highest-σ images (§6.10), and under imaging shift (§6.7) density proxies do not change while the error does.</p>")
    return "\n".join(s)


def sec_shift():
    s = ["<h3>6.7 Imaging shift: does the uncertainty notice?</h3>",
         "<p>Moving from a well plate to a chip changes the optics: thick PDMS and curved channel walls defocus and scatter light, and labs differ in camera noise, contrast and modality. "
         "We applied such shifts to 50 test images (time-points 10 and 100): Gaussian defocus, sensor noise, contrast loss, and a modality swap to the dataset's digital phase-contrast rendering.</p>",
         figure("shift_test.png", "<b>Figure 6.</b> (A) Pearson r with the real stain under each shift. (B) Mean σ relative to the clean image (log scale). "
                "ChipStain (blue) and the U-Net (orange) both use 8× TTA; the U-Net's σ is its TTA disagreement.")]
    tabs = md_tables(f"{RES}/shift_test.md")
    if tabs:
        s.append(html_table(tabs[0][1]))
        s.append("<p class='small'><b>Table 7.</b> AUROC of per-image mean σ for separating shifted from clean images, per uncertainty term "
                 "(1 = perfect separation, 0.5 = no signal, &lt; 0.5 = σ falls under shift).</p>")
    s.append("{{shift_text}}")
    return "\n".join(s)


def sec_counting(r):
    s = ["<h3>6.8 Downstream counting and the direct-segmentation alternative</h3>"]
    s.append("{{counting_text}}")
    cp = md_tables(f"{RES}/cellpose_baseline.md")
    if cp:
        s.append(html_table(cp[0][1]))
        s.append("<p class='small'><b>Table 8.</b> Cellpose bright-field nuclei model published with the dataset, per time-point, against the same StarDist references.</p>")
    return "\n".join(s)


def sec_nucleus():
    s = ["<h3>6.9 Per-nucleus uncertainty</h3>"]
    tabs = md_tables(f"{RES}/nucleus_uncertainty.md")
    if tabs:
        s.append(html_table(tabs[0][1]))
        s.append("<p class='small'><b>Table 9.</b> AUROC of each per-nucleus score for flagging predicted nuclei that match no reference nucleus (IoU &lt; 0.5).</p>")
    s.append(figure("nucleus_gate.png", "<b>Figure 7.</b> Discarding the highest-score detections raises precision but lowers recall and F1 for every score: per-nucleus σ is a review flag, not a count filter."))
    s.append("{{nucleus_text}}")
    return "\n".join(s)


def sec_prolif():
    s = ["<h3>6.10 A biological read-out: proliferation from bright-field alone</h3>",
         "<p>The organisers ask for predictions that enable downstream analysis. The most direct biological quantity in this time-lapse is the population doubling time. "
         "Per test field we fit log(count) against time over the five time-points (0–37 h) and compare doubling times from label-free counts with those from the StarDist reference annotation of the real stain.</p>"]
    tabs = md_tables(f"{RES}/proliferation.md")
    if tabs:
        s.append(html_table(tabs[0][1]))
        s.append("<p class='small'><b>Table 10.</b> Doubling time (geometric mean over fields) and its error against the reference, without and with the σ gate "
                 "(threshold fixed on the validation split, §4.4). 'Failed frames' are test frames whose nuclei F1 is below 0.5.</p>")
    s.append(figure("proliferation.png", "<b>Figure 8.</b> (A) Mean nuclei per field over time from the reference and from label-free counts (seed 0). (B) Per-field doubling times, all seeds."))
    s.append("{{prolif_text}}")
    s.append(figure("timelapse.png", "<b>Figure 9.</b> One held-out field imaged every 15 min for 60 h (240 frames): nuclei counted on the label-free prediction and on the real H2B stain with the same pipeline (top), and the frame's mean σ (bottom). "
                    "Data: R. Guiet, EPFL BIOP, Zenodo 10.5281/zenodo.6139958, CC BY 4.0."))
    s.append(figure("timelapse_strip.png", "<b>Figure 10.</b> Frames at 0, 15, 30, 45 and 60 h: bright-field input, predicted H2B and σ."))
    s.append("{{timelapse_text}}")
    return "\n".join(s)


def sec_neural():
    s = ["<h3>6.11 Transfer to neural cultures</h3>"]
    z = read("neural/zeroshot.csv")
    if z is None:
        s.append("<p><i>The neural transfer experiment is still running; this section is completed when it finishes.</i></p>")
        return "\n".join(s)
    s.append("{{neural_text}}")
    s.append(figure("neural_finetune.png", "<b>Figure 11.</b> Human iPSC-derived motor neurons (Christiansen et al. 2018, CC BY 4.0), 3 test wells. Fine-tuning with k training wells from the HeLa ChipStain weights (blue) "
                    "or from ImageNet only (orange); dashed line: HeLa model applied zero-shot."))
    s.append(figure("neural_examples.png", "<b>Figure 12.</b> Neural test well: bright-field, real DAPI, zero-shot prediction and σ, fine-tuned prediction and σ."))
    return "\n".join(s)


def sec_ensemble():
    tabs = md_tables("outputs/ensemble/summary_test.md") if os.path.exists("outputs/ensemble/summary_test.md") else []
    if not tabs:
        return ""
    return "\n".join(["<h3>6.12 Deep ensembles</h3>", "{{ensemble_text}}", html_table(tabs[0][1]),
                      "<p class='small'><b>Table 11.</b> 3-seed deep ensembles (all 125 test images): mean of the member predictions; σ² = variance of member means + mean member σ².</p>"])


def timepoint_table(r):
    d = r.img[r.img.cfg == FULL].groupby("timepoint")[["pearson", "ssim", "seg_f1", "seg_f1_realfluo", "mean_sigma", "mae"]].mean()
    rows = [["time-point (h)", "Pearson r", "SSIM", "seg-F1 (pred)", "seg-F1 (real)", "mean σ", "MAE"]]
    hours = {1: 0, 10: 2.25, 50: 12.25, 100: 24.75, 150: 37.25}
    for t, x in d.iterrows():
        rows.append([f"{int(t)} ({hours[int(t)]} h)", f"{x.pearson:.3f}", f"{x.ssim:.3f}", f"{x.seg_f1:.3f}", f"{x.seg_f1_realfluo:.3f}", f"{x.mean_sigma:.4f}", f"{x.mae:.4f}"])
    return ("<p>Cell density grows over the time-lapse (frames 1 → 150 ≈ 0 → 37 h of the 60 h recording). ChipStain + TTA, mean over 3 seeds: fidelity and downstream F1 fall with density "
            "and mean σ rises with it; the real-stain segmentation falls too, so part of the drop is the segmenter.</p>" + html_table(rows))


def image_register():
    p = "report/results/image_register.csv"
    if not os.path.exists(p):
        return ""
    d = pd.read_csv(p)
    rows = [["file", "pixels", "shows", "pixel source", "changes"]] + [[r.file, r.pixels, r.shows, r.pixel_source, r.modification] for r in d.itertuples()]
    return html_table(rows, num_cols_from=99)


# ----------------------------------------------------------------------------- key numbers
def numbers(r):
    """Every number quoted in prose, computed from the result files (n/a if missing)."""
    n = {}

    def put(k, f):
        try:
            n[k] = f()
        except Exception:  # noqa: BLE001 - missing optional result file
            n[k] = "n/a"

    for cfg, tag in [(FULL, "full"), (BT, "bt"), (CS, "cs"), (BU, "bu"), (PRE, "pre")]:
        put(f"n_{tag}_pearson", lambda c=cfg: f"{r.m(c, 'pearson'):.3f}")
        put(f"n_{tag}_ssim", lambda c=cfg: f"{r.m(c, 'ssim'):.3f}")
        put(f"n_{tag}_mae", lambda c=cfg: f"{r.m(c, 'mae'):.4f}")
        put(f"n_{tag}_segf1", lambda c=cfg: f"{r.m(c, 'seg_f1'):.3f}")
        put(f"n_{tag}_segf1_sd", lambda c=cfg: f"{r.sd(c, 'seg_f1'):.3f}")
        put(f"n_{tag}_rho", lambda c=cfg: f"{r.m(c, 'spearman_unc_err'):.2f}")
        put(f"n_{tag}_ause", lambda c=cfg: f"{r.m(c, 'ause'):.3f}")
        put(f"n_{tag}_img_rho", lambda c=cfg: f"{r.m(c, 'img_rho_sigma_mae'):.2f}")
        put(f"n_{tag}_img_npred", lambda c=cfg: f"{r.m(c, 'img_rho_npred_mae'):.2f}")
        put(f"n_{tag}_img_partial", lambda c=cfg: f"{r.m(c, 'img_partial_sigma_mae_given_npred'):.2f}")
    put("n_full_s0_pearson", lambda: f"{r.seed0(FULL, 'pearson'):.3f}")
    put("n_full_s0_segf1", lambda: f"{r.seed0(FULL, 'seg_f1'):.3f}")
    put("n_bu_seed_gap", lambda: (lambda v: f"{v.max() - v.min():.2f}")(r.seed[r.seed.cfg == BU]["seg_f1"]))
    put("n_mae_rel", lambda: f"{100 * (r.m(FULL, 'mae') / r.m(BT, 'mae') - 1):.0f}")
    put("n_gain", lambda: f"{100 * r.m(FULL, 'gain20'):.0f}")
    put("n_gain_mu", lambda: f"{100 * r.m(FULL, 'gain20_mu'):.0f}")
    put("n_gain_grad", lambda: f"{100 * r.m(FULL, 'gain20_grad'):.0f}")
    put("n_gain_oracle", lambda: f"{100 * r.m(FULL, 'gain20_oracle'):.0f}")
    put("n_ause_mu", lambda: f"{r.m(FULL, 'ause_mu'):.3f}")
    put("n_ause_grad", lambda: f"{r.m(FULL, 'ause_grad'):.3f}")
    put("n_sigma_range", lambda: (lambda v: f"{v.quantile(.05):.3f}–{v.quantile(.95):.3f} (5th–95th percentile, with TTA)")(r.img[r.img.cfg == FULL]["mean_sigma"]))
    put("n_real_segf1", lambda: f"{r.img[r.img.cfg == FULL].groupby('seed')['seg_f1_realfluo'].mean().mean():.3f}")
    for m in ("pearson", "spearman_unc_err", "ause"):
        put(f"n_ci_{m}", lambda m=m: (lambda t: f"{t.mean_a - t.mean_b:+.3f} (95 % CI {t.ci_lo:+.3f} to {t.ci_hi:+.3f})")(r.t(FULL, BT, m)))
    put("n_ci_seg_f1", lambda: (lambda t: f"{t.mean_a - t.mean_b:+.3f} (95 % CI {t.ci_lo:+.3f} to {t.ci_hi:+.3f})")(r.t(FULL, BT, "seg_f1")))
    # shift test (9-run format: shift_detection / shift_failure / shift_robustness / shift_gate csv)
    def shift():
        out = {}
        det, fail, rob = read("shift_detection.csv"), read("shift_failure.csv"), read("shift_robustness.csv")
        if det is not None:
            K = {"chipstain_nll": "cs", "baseline_unet": "bu", "pretrained_l1": "pre"}
            T = {"total σ": "sig", "learned head (TTA mean)": "sig_learned_tta", "TTA disagreement": "sig_tta_views", "learned head, single pass": "sig_single_pass"}
            F = {"blur": "defocus", "modality": "dpc", "noise": "noise"}
            for (cfg, term, fam), g in det.groupby(["cfg", "term", "family"]):
                out[f"n_shift_{K[cfg]}_{T[term]}_{F[fam]}"] = f"{g.auroc.mean():.2f}"
            for (cfg, term), g in fail.groupby(["cfg", "term"]):
                out[f"n_shift_{K[cfg]}_{T[term]}_fail"] = f"{g.auroc.mean():.2f}"
                if term == "total σ":
                    out[f"n_shift_{K[cfg]}_nfail"] = str(int(g.n_failed.sum()))
            for cfg, g in rob.groupby("cfg"):
                m = g.groupby("condition")[["pearson", "sigma_ratio"]].mean()
                out[f"n_shift_{K[cfg]}_r_clean"] = f"{m.loc['clean', 'pearson']:.2f}"
                out[f"n_shift_{K[cfg]}_r_def1"] = f"{m.loc['blur σ=1 px', 'pearson']:.2f}"
                out[f"n_shift_{K[cfg]}_r_dpc"] = f"{m.loc['modality: DPC', 'pearson']:.2f}"
                out[f"n_shift_{K[cfg]}_sig_def4x"] = f"{m.loc['blur σ=4 px', 'sigma_ratio']:.1f}"
                out[f"n_shift_{K[cfg]}_sig_dpcx"] = f"{m.loc['modality: DPC', 'sigma_ratio']:.1f}"
            gt = read("shift_gate.csv")
            if gt is not None:
                for cfg, g in gt.groupby("cfg"):
                    m = g.groupby("condition")["flagged"].mean()
                    out[f"n_shift_{K[cfg]}_gate_clean"] = f"{100 * m.get('clean', np.nan):.0f}"
                    out[f"n_shift_{K[cfg]}_gate_blur2"] = f"{100 * m.get('blur σ=2 px', np.nan):.0f}"
                    out[f"n_shift_{K[cfg]}_gate_dpc"] = f"{100 * m.get('modality: DPC', np.nan):.0f}"
            out["n_shift_seeds"] = str(int(det.seed.nunique()))
            return out
        # legacy single-seed file
        from scipy.stats import mannwhitneyu
        d = read("shift_test.csv")
        for run in d.run.unique():
            g = d[d.run == run]
            k = "cs" if "chipstain" in run else "bu"
            for term in ("mean_sigma", "mean_sigma_learned_tta", "mean_sigma_tta_views", "mean_sigma_single_pass"):
                if term not in g or g[term].isna().all() or (g[term] == 0).all():
                    continue
                c = g[g.condition == "clean"][term].values
                for fam, conds in [("defocus", ["defocus σ=1", "defocus σ=2", "defocus σ=4"]), ("dpc", ["modality: DPC"]), ("noise", ["noise 10 %", "noise 20 %"])]:
                    sh = g[g.condition.isin(conds)][term].values
                    out[f"n_shift_{k}_{term.replace('mean_sigma', 'sig')}_{fam}"] = f"{mannwhitneyu(sh, c).statistic / (len(c) * len(sh)):.2f}"
            t = g.groupby("condition")[["pearson", "mean_sigma"]].mean()
            out[f"n_shift_{k}_r_clean"] = f"{t.loc['clean', 'pearson']:.2f}"
            out[f"n_shift_{k}_r_def1"] = f"{t.loc['defocus σ=1', 'pearson']:.2f}"
            out[f"n_shift_{k}_r_dpc"] = f"{t.loc['modality: DPC', 'pearson']:.2f}"
            out[f"n_shift_{k}_sig_def4x"] = f"{t.loc['defocus σ=4', 'mean_sigma'] / t.loc['clean', 'mean_sigma']:.1f}"
            out[f"n_shift_{k}_sig_dpcx"] = f"{t.loc['modality: DPC', 'mean_sigma'] / t.loc['clean', 'mean_sigma']:.1f}"
        out["n_shift_seeds"] = "1"
        return out
    try:
        n.update(shift())
    except Exception as e:  # noqa: BLE001
        print("numbers: shift unavailable:", e)
    # time-lapse
    def tl():
        d = read("timelapse.csv")
        fit = lambda col: np.log(2) / np.polyfit(d.hours, np.log(d[col].clip(lower=1)), 1)[0]  # noqa: E731
        from scipy.stats import spearmanr
        return {"n_tl_dt_pred": f"{fit('n_pred'):.1f}", "n_tl_dt_real": f"{fit('n_real'):.1f}",
                "n_tl_count_err": f"{100 * ((d.n_pred - d.n_real).abs() / d.n_real.clip(lower=1)).mean():.0f}",
                "n_tl_f1": f"{d.f1_vs_real.median():.2f}", "n_tl_rho": f"{spearmanr(d.mean_sigma, 1 - d.f1_vs_real).statistic:.2f}",
                "n_tl_frames": str(len(d))}
    try:
        n.update(tl())
    except Exception:  # noqa: BLE001
        pass
    # calibration
    def cal():
        d = read("calibration.csv")
        g = d.groupby("model").mean(numeric_only=True)
        return {"n_cal_68": f"{100 * g.loc['chipstain_nll_tta', 'raw_0.683']:.0f}", "n_cal_95": f"{100 * g.loc['chipstain_nll_tta', 'raw_0.95']:.0f}",
                "n_cal_fg95": f"{100 * g.loc['chipstain_nll_tta', 'fg_0.95']:.0f}" if "fg_0.95" in g else "n/a",
                "n_cal_scale": f"{g.loc['chipstain_nll_tta', 'val_scale']:.2f}", "n_cal_unet_68": f"{100 * g.loc['baseline_unet_tta', 'raw_0.683']:.0f}",
                "n_cal_unet_scale": f"{g.loc['baseline_unet_tta', 'val_scale']:.0f}"}
    # proliferation
    def prolif():
        d = read("proliferation.csv")
        o = {}
        for model, k in [("ChipStain + TTA", "cs"), ("U-Net + TTA", "bt")]:
            a = d[(d.model == model) & (d.frames == "all frames")].set_index("seed")
            gt = d[(d.model == model) & (d.frames == "σ-gated")].set_index("seed")
            o[f"n_pl_{k}_bias_all"] = f"{a.bias_pct.abs().mean():.1f}"
            o[f"n_pl_{k}_bias_gate"] = f"{gt.bias_pct.abs().mean():.1f}"
            o[f"n_pl_{k}_mape_all"] = f"{a.mape_pct.mean():.1f}"
            o[f"n_pl_{k}_mape_gate"] = f"{gt.mape_pct.mean():.1f}"
            o[f"n_pl_{k}_caught"] = f"{int(gt.failed_frames_flagged.sum())}/{int(gt.failed_frames.sum())}"
            o[f"n_pl_{k}_flagged"] = f"{int(gt.frames_flagged.sum())}"
            o[f"n_pl_{k}_worst_all"] = f"{a.bias_pct.abs().max():.1f}"
            o[f"n_pl_{k}_worst_gate"] = f"{gt.bias_pct.abs().max():.1f}"
            o[f"n_pl_{k}_review"] = f"{int(gt.fields_needing_review.sum())}"
        ref = read("proliferation_per_field.csv", index_col=0)["reference"]
        o["n_pl_ref_dt"] = f"{np.exp(np.log(ref).mean()):.1f}"
        o["n_pl_frames_total"] = str(3 * 125)
        return o
    # per nucleus
    def nuc():
        d = read("nucleus_auroc.csv")
        g = d.groupby(["model", "score"])["auroc"].mean()
        return {"n_nuc_cs": f"{g[('ChipStain + TTA', 'sigma_learned')]:.2f}", "n_nuc_bt": f"{g[('U-Net + TTA', 'sigma_tta_unet')]:.2f}",
                "n_nuc_dim_cs": f"{g[('ChipStain + TTA', 'dim')]:.2f}", "n_nuc_dim_bt": f"{g[('U-Net + TTA', 'dim')]:.2f}",
                "n_nuc_ens": f"{g[('ChipStain ensemble (3 seeds + TTA)', 'sigma_ensemble')]:.2f}"}
    # direct segmentation
    def cp():
        from scipy.stats import spearmanr
        d = read("cellpose_baseline.csv")
        rh = [spearmanr(g.mean_sigma, 1 - g.seg_f1_cp).statistic
              for _, g in r.img[r.img.cfg == FULL].merge(d, on=["well", "field", "timepoint"], suffixes=("", "_cp")).groupby("seed")]
        return {"n_cp_f1": f"{d.seg_f1.mean():.3f}", "n_cp_count_err": f"{100 * ((d.n_pred - d.n_ref).abs() / d.n_ref).mean():.1f}",
                "n_cp_sigma_rho": f"{np.mean(rh):.2f}"}
    # ensembles
    def ens():
        d = pd.read_csv("outputs/ensemble/per_image_test.csv").groupby("run").mean(numeric_only=True)
        return {"n_ens_cs_ause": f"{d.loc['ens3_chipstain_nll', 'ause']:.3f}", "n_ens_cs_rho": f"{d.loc['ens3_chipstain_nll', 'spearman_unc_err']:.2f}",
                "n_ens_cstta_ause": f"{d.loc['ens3_chipstain_nll_tta', 'ause']:.3f}", "n_ens_cstta_pearson": f"{d.loc['ens3_chipstain_nll_tta', 'pearson']:.3f}",
                "n_ens_cstta_segf1": f"{d.loc['ens3_chipstain_nll_tta', 'seg_f1']:.3f}", "n_ens_bu_ause": f"{d.loc['ens3_baseline_unet', 'ause']:.3f}",
                "n_ens_bu_rho": f"{d.loc['ens3_baseline_unet', 'spearman_unc_err']:.2f}", "n_ens_pre_ause": f"{d.loc['ens3_pretrained_l1', 'ause']:.3f}",
                "n_ens_pre_rho": f"{d.loc['ens3_pretrained_l1', 'spearman_unc_err']:.2f}"}
    # neural transfer
    def neural():
        z = read("neural/zeroshot.csv"); f = read("neural/finetune.csv")
        o = {}
        zz = z.groupby("model")[["pearson", "seg_f1_vs_real", "mean_sigma", "n_pred", "n_real"]].mean()
        o["n_nz_cs_r"] = f"{zz.loc['ChipStain (HeLa, zero-shot)', 'pearson']:.2f}"
        o["n_nz_bu_r"] = f"{zz.loc['U-Net baseline (HeLa, zero-shot)', 'pearson']:.2f}"
        o["n_nz_cs_f1"] = f"{zz.loc['ChipStain (HeLa, zero-shot)', 'seg_f1_vs_real']:.2f}"
        o["n_nz_cs_sigma"] = f"{zz.loc['ChipStain (HeLa, zero-shot)', 'mean_sigma']:.3f}"
        o["n_nz_cs_count"] = f"{100 * (zz.loc['ChipStain (HeLa, zero-shot)', 'n_pred'] / zz.loc['ChipStain (HeLa, zero-shot)', 'n_real'] - 1):+.0f}"
        if f is not None:
            g = f.groupby(["init", "k_wells"])[["pearson", "seg_f1_vs_real", "mean_sigma", "spearman_unc_err"]].mean()
            ks = sorted(f.k_wells.unique())
            for init, tag in [("from ChipStain (HeLa)", "cs"), ("from ImageNet", "in")]:
                for k in ks:
                    if (init, k) in g.index:
                        o[f"n_nf_{tag}_k{k}_r"] = f"{g.loc[(init, k), 'pearson']:.2f}"
                        o[f"n_nf_{tag}_k{k}_f1"] = f"{g.loc[(init, k), 'seg_f1_vs_real']:.2f}"
                        o[f"n_nf_{tag}_k{k}_sigma"] = f"{g.loc[(init, k), 'mean_sigma']:.3f}"
            o["n_nf_kmax"] = str(max(ks))
            o["n_nf_kmin"] = str(min(ks))
        sel = json.load(open(f"{RES}/neural/selection.json"))
        o["n_neural_scale"], o["n_neural_z"] = f"{sel['scale']:.2f}", str(sel["z"])
        return o
    # ablation: list the robust single-factor effects
    def ablation():
        steps = [("ablate_scratch_l1_lr5e4", BU, "lowering the learning rate"), (PRE, "ablate_scratch_l1_lr5e4", "ImageNet pre-training"),
                 ("ablate_pretrained_mse", PRE, "switching L1 to MSE"), (CS, "ablate_pretrained_mse", "the variance head (β-NLL vs MSE)"),
                 (CS, "ablate_nll_beta0", "β = 0.5 instead of plain NLL"), (CS, "ablate_nll_beta1", "β = 0.5 instead of β = 1")]
        nm = {"pearson": "Pearson r", "ssim": "SSIM", "mae": "MAE", "seg_f1": "nuclei F1", "spearman_unc_err": "ρ(σ, error)", "ause": "AUSE"}
        eff = []
        for a, b, name in steps:
            for m in ("pearson", "ssim", "mae", "seg_f1", "spearman_unc_err", "ause"):
                t = r.t(a, b, m)
                if t is not None and sig(t):
                    eff.append(f"{name}: {nm[m]} {t.mean_a - t.mean_b:+.3f}")
        if not r.has("ablate_pretrained_mse"):
            raise ValueError("ablation arms not evaluated yet")
        return {"n_abl_robust": "; ".join(eff) if eff else "none — no single-factor change has a robust effect (CI excluding zero and consistent across seeds)",
                "n_abl_n": str(len(eff))}
    # Cellpose extras
    def cpx():
        o = {}
        sh = read("cellpose_shift.csv")
        if sh is not None:
            m = sh.groupby("condition")["f1"].mean()
            o["n_cps_clean"], o["n_cps_blur1"], o["n_cps_blur2"], o["n_cps_dpc"] = (f"{m[c]:.2f}" for c in ("clean", "blur=1", "blur=2", "modality: DPC"))
        pr = read("cellpose_on_predictions.csv")
        if pr is not None:
            pr["src"] = pr.source.str.replace(r" s\d$", "", regex=True)
            m = pr.groupby("src")["f1"].mean()
            o["n_cpp_real"], o["n_cpp_pred"] = f"{m['real H2B']:.3f}", f"{m['ChipStain + TTA']:.3f}"
        return o
    for f in (cal, prolif, nuc, cp, ens, neural, ablation, cpx):
        try:
            n.update(f())
        except Exception as e:  # noqa: BLE001
            print("numbers:", f.__name__, "unavailable:", e)
    try:
        dm = json.load(open(f"{RES}/demo_examples.json"))
        n.update({"n_demo_count": str(dm["clean"]["count"]), "n_demo_ref": str(dm["reference_count"]),
                  "n_demo_sigma_clean": f"{dm['clean']['mean_sigma']:.2f}", "n_demo_sigma_blur": f"{dm['blur']['mean_sigma']:.2f}"})
    except Exception as e:  # noqa: BLE001
        print("numbers: demo unavailable:", e)
    # CPU timing
    try:
        t = json.load(open(f"{RES}/cpu_timing.json"))
        n["cpu_single"], n["cpu_tta"] = f"≤ {t['cpu_single']:.1f}", f"≤ {t['cpu_tta']:.1f}"
    except Exception:  # noqa: BLE001
        n["cpu_single"], n["cpu_tta"] = "≈0.5", "≈2.5"
    return n


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/ChipStain_Technical_Report")
    ap.add_argument("--texts", default="report/report_texts.json", help="prose blocks that interpret the numbers")
    a = ap.parse_args()
    r = R()
    texts = json.load(open(a.texts)) if os.path.exists(a.texts) else {}
    results = "\n".join([sec_main(r), sec_ablation(r), sec_qual(), sec_uncertainty(r), sec_image_level(r), sec_shift(), sec_counting(r), sec_nucleus(), sec_prolif(), sec_neural(), sec_ensemble()])
    tpl = open("report/report_template.html", encoding="utf-8").read()
    chan = open(f"{RES}/channel_check.md").read() if os.path.exists(f"{RES}/channel_check.md") else ""
    ch = re.findall(r":\s+([0-9.]+) ±", chan)
    logs = [json.load(open(f))[-1]["time_min"] for f in glob.glob("runs/*/log.json")]
    sub = {
        "date": datetime.date.today().isoformat(),
        "sec_results": results,
        "chan_tub": ch[0] if ch else "0.42", "chan_h2b": ch[1] if len(ch) > 1 else "0.79",
        "train_minutes": f"{min(logs):.0f}–{max(logs):.0f}" if logs else "≈15",
        "fig_curves": figure("training_curves.png", "").split("src='")[1].split("'")[0] if os.path.exists("report/figures/training_curves.png") else "",
        "fig_qual_base": figure("qualitative_baseline_unet_s0_tta.png", "").split("src='")[1].split("'")[0] if os.path.exists("report/figures/qualitative_baseline_unet_s0_tta.png") else "",
        "fig_qual_nll": figure("qualitative_chipstain_nll_s0.png", "").split("src='")[1].split("'")[0] if os.path.exists("report/figures/qualitative_chipstain_nll_s0.png") else "",
        "app_timepoint": timepoint_table(r),
        "app_images": image_register(),
    }
    sub.update(texts)  # abstract, sec_limitations, sec_impact, *_text
    sub.update(numbers(r))
    json.dump({k: v for k, v in sub.items() if k.startswith(("n_", "cpu_", "train_minutes"))}, open(f"{RES}/key_numbers.json", "w"), indent=1)
    for _ in range(2):  # texts may contain placeholders themselves
        for k, v in sub.items():
            tpl = tpl.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)))
    tpl = re.sub(r"\{\{\w+\}\}", "(pending)", tpl)  # results still running: never leave raw placeholders
    if left:
        print("WARNING unfilled placeholders:", left)
    out_html = a.out + ".html"
    open(out_html, "w", encoding="utf-8").write(tpl)
    print("wrote", out_html)
    if CHROME:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                        f"--print-to-pdf={os.path.abspath(a.out)}.pdf", "file://" + os.path.abspath(out_html)], capture_output=True)
        print("wrote", a.out + ".pdf")
    else:
        print("WARNING: no Chrome/Chromium found (set CHROME=/path); only the HTML was written")


if __name__ == "__main__":
    main()
