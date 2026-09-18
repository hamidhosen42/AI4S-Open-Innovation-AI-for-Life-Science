**Category: Model & Algorithm**

# ChipStain — uncertainty-aware label-free nuclear staining for organ-on-a-chip imaging

## Demo video

▶ **[VIDEO LINK — paste YouTube (unlisted) link here]** (≤ 5 min)

## Code repository

**https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science** — MIT licence. Includes `requirements.txt`, `pyproject.toml`, weights download script, `scripts/inference.py` (CPU-only, ~6 s per image), `scripts/train.py` / `evaluate.py` for full reproduction, a Gradio demo and a Kaggle notebook (`notebooks/kaggle_train.ipynb`).

## Project summary

Fluorescent nuclear stains are the starting point of almost every quantitative read-out in organ-on-a-chip (OoC) experiments — cell counts, viability, proliferation, morphology, dose–response — yet staining is phototoxic, terminates a live time-lapse and adds reagent and handling steps on small microfluidic devices. In-silico labeling (ISL) networks predict the stain from label-free bright-field images, but return one image with no indication of where it can be trusted; that missing trust signal is the main obstacle to using ISL for decisions.

ChipStain is a bright-field → H2B (nuclear) translation model that outputs, for every pixel, both the fluorescence intensity and its uncertainty. It is a U-Net with an ImageNet-pretrained ResNet-34 encoder and a heteroscedastic Gaussian head trained with the β-NLL loss; at inference, eight-fold dihedral test-time augmentation adds an epistemic term. On a held-out well of the public HeLa "Kyoto" dataset (125 images, CC BY 4.0), ChipStain reaches Pearson r = 0.779 and SSIM = 0.808 against the real stain (plain U-Net: 0.766 / 0.830), and nuclei segmented from its prediction match reference objects with F1 = 0.718 (0.765 when the real fluorescence is segmented the same way; 0.693 for the plain U-Net). The uncertainty is informative: per-pixel σ correlates with the absolute error (Spearman ρ = 0.44, versus 0.32 for TTA disagreement of a plain U-Net), discarding the 20 % most-uncertain pixels lowers the remaining error by 55 %, and the per-image mean σ ranks images by their accuracy with ρ = 0.88–0.95 — a single threshold that routes unreliable frames to human review. Inference is CPU-only with no external service; all code, weights and data links are public.

## Technical report

📄 **[REPORT PDF LINK — upload `report/ChipStain_Technical_Report.pdf` as a Kaggle attachment or to the GitHub release and paste the link here]** (14 pages incl. appendices)

### 1. Problem and application scenario
OoC devices are imaged for days by transmitted light; nuclear read-outs normally require a DNA stain or a fluorescent histone reporter. Staining costs: phototoxicity, end-point-only measurement, reagent/handling cost on chip, and a spent fluorescence channel. ISL removes these costs — it is one of the three impact areas named by the organisers — but adoption needs a trust signal. **ChipStain answers a narrower, more useful question: can we predict the stain *and* say, per pixel and per image, how much to trust it?** Target users: OoC / cell-culture labs and CROs running live or high-content assays; platform developers who need a quality gate before automated analysis.

### 2. Data (public, CC BY 4.0, no personal or clinical data)
*Automatic labelling of HeLa "Kyoto" cells* (R. Guiet, EPFL PTBIOP), Zenodo 10.5281/zenodo.6140064, derived from 10.5281/zenodo.6139958. 540 × 540 px 16-bit images from a PerkinElmer Operetta (20×), five time-points of a 60 h time-lapse. Input: bright-field. Target: mCherry-H2B channel. StarDist nuclei labels are used only as reference objects for the downstream metric. **Splits:** train = wells R05-C05/C07 fields 0–19 (200 images); validation = fields 20–24 of the same wells (50); **test = separate well R05-C03 (125)**. All time-points of a field stay on one side, so no near-duplicate leakage. Pre-processing: per-image percentile normalisation of the input; fixed global scaling of the target to [0, 1]. Full details: `DATA.md`.

### 3. Method
* **Backbone:** U-Net (`segmentation_models_pytorch`), ResNet-34 encoder initialised from ImageNet, 1-channel input, ≈24 M parameters.
* **Heteroscedastic head:** two output maps — mean μ and log-variance log σ² of a per-pixel Gaussian. Trained with **β-NLL** (Seitzer et al., ICLR 2022; β = 0.5), which multiplies the NLL by a detached (σ²)^β factor so that hard pixels are not abandoned early in training. σ is the model's aleatoric uncertainty.
* **Test-time augmentation:** 8 dihedral views; μ̄ = mean prediction; σ²_tot = mean predicted σ² (aleatoric) + variance of the 8 μ (epistemic proxy).
* **Ablation:** U-Net from scratch + L1 (standard ISL recipe) → + ImageNet encoder → + β-NLL head (ChipStain) → + TTA (full). Also compared: the plain U-Net's TTA view-variance used as an uncertainty, to show the learned head is needed.
* **Evaluation:** MAE / PSNR / SSIM / Pearson r; downstream nuclei segmentation (fixed Otsu + watershed pipeline) matched to StarDist references at IoU ≥ 0.5 → F1; uncertainty quality via Spearman ρ(σ, |error|), sparsification curves / AUSE, and image-level ρ(mean σ, MAE).

### 4. Results (test well R05-C03, n = 125, mean ± std over images)

| Model | Pearson r ↑ | SSIM ↑ | PSNR ↑ | seg-F1 ↑ | ρ(σ, err) ↑ | AUSE ↓ |
|---|---|---|---|---|---|---|
| U-Net baseline (L1) | 0.766 ± 0.060 | 0.830 ± 0.074 | 24.32 | 0.693 ± 0.118 | — | — |
| + ImageNet encoder (L1) | 0.756 ± 0.057 | 0.825 ± 0.081 | 24.25 | 0.667 ± 0.126 | — | — |
| U-Net baseline + TTA (σ = view variance) | 0.780 ± 0.057 | 0.834 ± 0.071 | 24.54 | 0.710 ± 0.114 | 0.315 ± 0.199 | 0.244 ± 0.080 |
| ChipStain (β-NLL head) | 0.774 ± 0.046 | 0.814 ± 0.082 | 24.32 | 0.711 ± 0.110 | 0.426 ± 0.153 | 0.242 ± 0.171 |
| **ChipStain + TTA (full)** | **0.779 ± 0.045** | 0.808 ± 0.085 | 24.42 | **0.718 ± 0.111** | **0.441 ± 0.173** | **0.201 ± 0.063** |
| Real fluorescence, same segmentation pipeline (ceiling) | — | — | — | 0.765 ± 0.086 | — | — |

* **Fidelity:** equal to the best L1 baseline in Pearson r (0.779 vs 0.780) and downstream F1 (0.718 vs 0.710), with a small SSIM cost (0.808 vs 0.834) — the uncertainty comes essentially for free.
* **Uncertainty is informative:** ρ(σ, |error|) = 0.44 vs 0.32 for TTA-disagreement alone; AUSE 0.20 vs 0.24. Removing the 20 % most-uncertain pixels reduces the remaining MAE by **55 %**. Per-image mean σ predicts per-image MAE with Spearman **ρ = 0.95** (single pass) / 0.88 (TTA).
* **Counting:** predicted nucleus counts deviate from the reference by 9.2 nuclei on average (7.3 % relative; r = 0.994 across images; counts range 28–538).
* **Density stress test (Appendix A of the report):** from sparse (t = 1) to dense (t = 150) fields, Pearson falls 0.81 → 0.72 and seg-F1 0.82 → 0.59, while mean σ rises 0.034 → 0.080 — the model flags its own hard cases without being told the time-point.
* Qualitatively, σ concentrates on exactly the regions where the prediction fails (debris, crowded/out-of-focus cells, mitotic figures) and stays low on well-focused nuclei and background.

### 5. Reliability and limitations
Single cell line and microscope (well-level hold-out tests well-to-well, not cross-system, generalisation; fine-tuning on ≥ 50 paired fields recommended for a new OoC system). Target is an H2B reporter, not a dye. σ is a triage signal, not a calibrated confidence interval — a heteroscedastic head can be confidently wrong far from the training distribution; TTA mitigates. The simple watershed segmenter caps F1 at 0.765 even on real fluorescence; the fair comparison is predicted-vs-real under the same pipeline. Pre-training did not help on this dataset under our schedule (reported honestly). Single seed (multi-seed script provided). No human/clinical/personal data; the main misuse risk is over-trusting a hallucinated prediction — the very thing the uncertainty output counters.

### 6. Impact
Every bright-field frame of an OoC time-lapse becomes a nuclear read-out without staining: counts and proliferation curves over days from the same chip, a freed fluorescence channel, lower reagent and handling cost. The uncertainty map makes this usable: mean-σ thresholds flag frames for review, per-pixel σ can be propagated to per-nucleus confidence and excluded from counts or dose–response curves, and σ maps point at imaging problems (focus, density, debris). CPU-only inference with no proprietary dependency lets the model sit inside an acquisition pipeline. A trustworthy label-free nuclear channel is a prerequisite for the organisers' goal of AI-driven OoC digital twins. Extensions: further channels (tubulin is in the dataset), 3-D, neural OoC cultures.

### 7. Reproduction
```bash
git clone https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science.git && cd AI4S-Open-Innovation-AI-for-Life-Science
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py            # Zenodo, ~757 MB
python scripts/download_weights.py         # -> weights/chipstain.pt
python scripts/inference.py --image demo/examples/example_bf_dense_t150.tif --weights weights/chipstain.pt --out outputs/pred --device cpu
bash scripts/run_all.sh 0                  # retrain all three configs (~40 min each)
python scripts/evaluate.py --runs runs/baseline_unet_s0 runs/pretrained_l1_s0 runs/chipstain_nll_s0 --out outputs/eval
python scripts/evaluate.py --runs runs/baseline_unet_s0 runs/chipstain_nll_s0 --tta --out outputs/eval_tta
python scripts/make_figures.py && python scripts/build_report.py
python demo/app.py --weights weights/chipstain.pt
```

### 8. Sources, licences, AI-tool disclosure
Data: Zenodo 10.5281/zenodo.6140064 / 6139958 (R. Guiet, EPFL PTBIOP), CC BY 4.0. Models/libraries: ResNet-34 ImageNet weights via segmentation_models_pytorch (MIT); PyTorch, NumPy, SciPy, scikit-image, tifffile, pandas, Matplotlib, PyYAML (BSD/MIT/PSF); Gradio (Apache-2.0). Method: β-NLL loss re-implemented from Seitzer et al. 2022. **AI tools:** Claude (Anthropic, via Claude Code) was used for code scaffolding, documentation drafting and rule analysis; all code was executed and verified by the team and every reported number comes from runs executed by the team. No LLM, foundation model, external API or paid service is used at inference. New work created for this competition; not based on prior work. See `AI_ASSISTANCE.md`.

## Optional demo
🖥 **[DEMO LINK — Hugging Face Space, paste here if deployed]** — if the Space is down, the same app runs locally with `python demo/app.py`; screenshots are in the report (Figure 2) and `outputs/pred/panel.png`.

## Team
Md. Hamid Hosen (team leader) — AI / machine learning. Single-member team; no biology/bioengineering member, so no cross-disciplinary bonus is claimed.
