**Category: Model & Algorithm**

# ChipStain — label-free nuclear staining that tells you where not to trust it

## Demo video

▶ **[VIDEO LINK — paste the YouTube link here]** (≤ 5 min)

## Code repository

**https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science** — MIT licence. `requirements.txt` + exact `requirements-lock.txt`, weights download script, `scripts/inference.py` (runs on a CPU: ≤ 0.5 s per 540×540 image, ≤ 2.4 s with 8× test-time augmentation), training / evaluation / analysis scripts that regenerate every number below, a Gradio demo and a Kaggle notebook.

## Project summary

Nuclear read-outs on an organ-on-a-chip (cell number, proliferation, viability) start from a nuclear stain, but fixation ends the experiment, live DNA dyes add phototoxic exposure, and reporter lines are impractical for primary or iPSC-derived tissue. In-silico labeling (ISL) predicts the stain from bright-field, but a point prediction gives no warning when it is wrong. ChipStain predicts the H2B nuclear channel from one bright-field image together with a per-pixel uncertainty σ (U-Net with an ImageNet ResNet-34 encoder, β-NLL heteroscedastic head, 8× test-time augmentation). We validate the uncertainty, not only the image, on a public HeLa time-lapse in a 96-well plate (held-out well, 125 images, 3 training seeds; no chip data were available to us). Against the same U-Net with TTA, ChipStain matches the correlation with the real stain (r 0.768 vs 0.764) at a small, consistent pixel-fidelity cost (MAE +10 %). Its σ ranks pixel errors better (Spearman ρ 0.47 vs 0.28; AUSE 0.177 vs 0.258; in every seed) and is approximately calibrated (nominal 95 % intervals cover 92 % of pixels). Under simulated blur and a phase-contrast input every model's prediction fails; whether σ warns is seed-dependent for all models (ChipStain: in 1 of 3 seeds — the released one), so σ is not a reliable drift alarm, whereas added noise is flagged by every model in every seed. A σ gate fixed on validation data cuts the error of label-free doubling times from 9.1 % to 4.1 %. On human iPSC-derived motor neurons, fine-tuning on a single well raises Pearson r from 0.59 (zero-shot) to 0.76. For nuclei counting alone, a direct bright-field segmenter is more accurate (Cellpose F1 0.831 vs 0.708); ChipStain's value is a fluorescence-like image plus a calibrated trust signal. Open code and weights; CPU inference.

## Technical report

📄 **Full report (PDF, 24 pages incl. appendices): https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science/blob/main/report/ChipStain_Technical_Report.pdf**

### 1. Problem
Nuclear read-outs (counts, proliferation, viability, morphology, dose–response) start from a nuclear stain, but on an organ-on-a-chip fixed DAPI ends the experiment, live DNA dyes are phototoxic and perturb the cell cycle, and reporter lines are impractical for primary or iPSC-derived (e.g. neural) tissue. In-silico labeling (ISL) predicts the stain from bright-field, but a point prediction gives no warning when it is wrong. **ChipStain asks: can we predict the nuclear stain and say — per pixel, per nucleus and per frame — how much to trust it, and does that signal beat simple alternatives?** Target users: OoC / cell-culture labs and CROs running live or high-content assays.

### 2. Data (public, CC BY 4.0, no personal or clinical data)
* **Main:** HeLa "Kyoto" (R. Guiet, EPFL BIOP; Zenodo [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)): 2-D cells in a 96-well imaging plate (CellCarrier Ultra 96) on a PerkinElmer Operetta, 20×; bright-field → mCherry-H2B. Train = wells R05-C05/C07 fields 0–19 (200 images), validation = fields 20–24 (50), **test = held-out well R05-C03 of the same plate and imaging session (125 images)**; all time-points of a field on one side. The automatic StarDist labels are used only as evaluation references, never for training (as the provider requests).
* **60 h time-lapse** of one held-out field (240 frames; Zenodo 10.5281/zenodo.6139958), evaluation only.
* **Neural cultures:** human iPSC-derived motor neurons, in-silico-labeling data Condition A (Christiansen et al., Cell 2018; CC BY 4.0), for a transfer test.
* **No organ-on-a-chip device data were available**: OoC is the target deployment; the shift test (§4) approximates some of the optical changes a chip introduces.

### 3. Method
U-Net (segmentation_models_pytorch) with an ImageNet ResNet-34 encoder; the head predicts the mean μ and log-variance of a per-pixel Gaussian, trained with **β-NLL** (Seitzer et al., ICLR 2022; β = 0.5). At inference, **8× dihedral test-time augmentation**: σ² = mean predicted σ² + variance across the 8 views ("TTA disagreement"). Baselines: L1 U-Net (scratch / ImageNet), the same U-Net with TTA (its σ = TTA disagreement), de-confounded ablation arms (learning rate, L1 vs MSE vs β-NLL, β ∈ {0, 0.5, 1}), 3-seed deep ensembles, and the dataset author's Cellpose model that segments nuclei directly from bright-field. **Prior art:** ISL already predicted per-pixel intensity distributions (Christiansen 2018) and CARE used probabilistic outputs and ensembles (Weigert 2018). Our contribution is not a new mechanism but a validated, open tool: σ tested against an equally strong TTA baseline and uncertainty-free proxies, at pixel / nucleus / frame level, under shift, on a biological read-out, and on neurons. Main results use 3 training seeds, with 95 % CIs from a bootstrap over seeds and fields and per-seed tests; other analyses state their seeds.

### 4. Results (test well, 125 images, mean ± s.d. over 3 seeds)

| Model | Pearson r ↑ | SSIM ↑ | MAE ↓ | PSNR ↑ | seg-F1 ↑ | ρ(σ, err) ↑ | AUSE ↓ | MAE drop, top-20 % σ removed ↑ |
|---|---|---|---|---|---|---|---|---|
| U-Net baseline (scratch, L1, lr 1e-3) | 0.749 ± 0.027 | 0.826 ± 0.003 | 0.0299 ± 0.0006 | 24.15 ± 0.24 | 0.648 ± 0.089 | — | — | — |
| + ImageNet encoder (L1, lr 5e-4) | 0.755 ± 0.005 | 0.823 ± 0.005 | 0.0296 ± 0.0003 | 24.25 ± 0.07 | 0.669 ± 0.021 | — | — | — |
| U-Net baseline + TTA (σ = view s.d.) | 0.764 ± 0.024 | 0.831 ± 0.003 | 0.0294 ± 0.0006 | 24.36 ± 0.25 | 0.665 ± 0.083 | 0.283 ± 0.071 | 0.258 ± 0.025 | 40.5 ± 3.2 % |
| + ImageNet encoder + TTA (σ = view s.d.) | 0.768 ± 0.006 | 0.827 ± 0.005 | 0.0291 ± 0.0003 | 24.47 ± 0.10 | 0.690 ± 0.019 | 0.317 ± 0.037 | 0.248 ± 0.011 | 42.9 ± 1.8 % |
| ChipStain (β-NLL, β = 0.5) | 0.757 ± 0.015 | 0.810 ± 0.018 | 0.0324 ± 0.0013 | 23.92 ± 0.36 | 0.697 ± 0.012 | 0.430 ± 0.006 | 0.303 ± 0.072 | 45.6 ± 1.6 % |
| ChipStain + TTA (full) | 0.768 ± 0.012 | 0.814 ± 0.022 | 0.0323 ± 0.0007 | 24.11 ± 0.32 | 0.708 ± 0.011 | 0.469 ± 0.027 | 0.177 ± 0.021 | 49.7 ± 4.2 % |
| Real fluorescence, same segmentation pipeline (reference level, not a bound) | — | — | — | — | 0.765 | — | — | — |

* **Fidelity (like-for-like vs U-Net + TTA):** Pearson r +0.004 (95 % CI -0.021 to +0.030) — no difference; MAE 10 % higher and SSIM lower (a small, consistent cost; see the ablation); nuclei F1 +0.043 (95 % CI -0.017 to +0.135) — higher on average but not robust across seeds.
* **Uncertainty (the main result):** ρ(σ, |error|) +0.186 (95 % CI +0.120 to +0.264); AUSE -0.081 (95 % CI -0.114 to -0.049); better in all 25 fields (seed-averaged) and significantly better in each seed. σ also beats uncertainty-free proxies over the whole sparsification curve (AUSE 0.177 vs 0.271 for edge strength, 0.308 for intensity); at a single 20 % cut-off its MAE reduction (50 %) is close to ranking by intensity (48 %).
* **Imaging shift:** under blur or a phase-contrast input every model's prediction fails, and whether σ warns depends on the training run for all models: ChipStain's σ rises in 1 of 3 seeds (including the released one) but falls in 2 of 3. σ is therefore **not** a reliable drift alarm; added noise, in contrast, is flagged by every model in every seed.
* **Calibration:** ChipStain's σ is approximately calibrated out of the box — the nominal 95 % interval covers 92 % of test pixels (validation-fitted scale ×0.98) — whereas the U-Nets' TTA disagreement needs ×21 rescaling: it can rank, not quantify.
* **Frame level:** mean σ ranks images by error (ρ 0.91), but in-distribution that is mostly cell density (predicted count alone: 0.89; the U-Net's TTA disagreement: 0.90). It matters when something goes wrong: one ChipStain seed's three catastrophic test frames are its three highest-σ images.
* **Per nucleus:** per-nucleus σ flags wrong detections only weakly (AUROC 0.63), no better than a dim-nucleus heuristic (0.66); we use it as a review aid, not a filter.
* **Counting vs direct segmentation:** counting on the prediction reaches nuclei F1 0.708; the dataset author's Cellpose model, which segments nuclei directly from bright-field, reaches 0.831. For counting alone, direct segmentation is better; ChipStain adds a fluorescence-like image and a calibrated σ.
* **Biology — proliferation:** label-free counts give population doubling times within 9.1 % (mean over seeds) of the real-stain reference; a σ gate fixed on validation data cuts this to 4.1 % (the same gate on the U-Net's TTA disagreement: 13.8 → 8.8 % — the gate helps both). On a 60 h, 240-frame recording: 24.9 h vs 23.9 h from the real stain.
* **Neural transfer:** on human iPSC-derived motor neurons (dish, not chip) the HeLa model reaches Pearson r 0.59 zero-shot (but undercounts nuclei by about half, and its σ stays in the normal range — it does not warn); fine-tuning on one well raises r to 0.76 (0.77 with 20 wells; one seed).
* **Ablation:** one factor at a time, 3 seeds each; robust effects: ImageNet pre-training (Pearson r +0.031, MAE -0.001, nuclei F1 +0.057); β = 0.5 instead of plain NLL (Pearson r +0.049, SSIM +0.030, MAE -0.002, nuclei F1 +0.084); β = 0.5 instead of β = 1 (SSIM +0.054, nuclei F1 +0.026, ρ(σ, error) +0.081); all other single-factor changes are within run-to-run variation.

### 5. Reliability and limitations
Validation is on 2-D HeLa cells in one well plate (one imaging session) and neurons in a dish — **no chip data**. ChipStain trades 10 % MAE for its uncertainty. Its σ is strongest at ranking pixel errors, calibration and catching catastrophic frames in distribution; under blur or a new modality it warns only in some training runs (as do the baselines), at the nucleus level it is no better than a simple heuristic, and a direct segmenter counts nuclei better. References are automatic StarDist labels. One of three seeds fails on two fields (σ flags them).

### 6. Impact
Once fine-tuned on a few paired images from a device: a quality-controlled, label-free nuclear channel for long OoC recordings — growth and viability curves without dye exposure or fixation, a freed fluorescence channel, and a σ gate that keeps catastrophic frames out of the analysis (in distribution; drift must still be checked with paired images). CPU inference, no proprietary dependencies.

### 7. Reproduction
```bash
git clone https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science.git && cd AI4S-Open-Innovation-AI-for-Life-Science
pip install -r requirements.txt && pip install -e .          # exact versions: requirements-lock.txt
python scripts/download_data.py && python scripts/download_weights.py
python scripts/inference.py --image demo/examples/example_bf_dense_t150.tif --weights weights/chipstain.pt --out outputs/pred --device cpu
bash scripts/run_all.sh 0 1 2 && bash scripts/run_ablations.sh 0 1 2      # ≈9–28 min per run on an Apple M5 (MPS)
# evaluation, statistics, analyses, figures and this report: see README "Reproduce everything"
python demo/app.py --weights weights/chipstain.pt
```

### 8. Sources, licences, AI-tool disclosure
Data: Zenodo 10.5281/zenodo.6140064 / 6139958 / 6140111 (R. Guiet, EPFL BIOP) and the in-silico-labeling data (Christiansen et al. 2018) — all CC BY 4.0. Encoder: ResNet-34 ImageNet weights (torchvision, BSD-3; via the Hugging Face Hub, training only). Libraries: PyTorch, segmentation_models_pytorch, timm, NumPy, SciPy, scikit-image, pandas, tifffile, Matplotlib, PyYAML, tqdm, Pillow, Gradio (BSD / MIT / Apache-2.0 / PSF / MPL-2.0). Method components re-implemented from Seitzer et al. 2022 (β-NLL), TTA uncertainty and AUSE (Ilg et al. 2018). **AI assistance: Claude Opus 5 and Claude Opus 5.5 (Anthropic), via Claude Code, were the primary implementation assistant** — they wrote most of the code, ran training and evaluation on the team's laptop under the team's direction, produced the figures and drafted the report, this Writeup and the video script. The team chose the problem, category and method, directed the work and takes full responsibility for the content (details: `AI_ASSISTANCE.md`). No language model or external API is used by the method itself. Released code and checkpoint: MIT (checkpoint trained on CC BY 4.0 data — attribute Guiet 2022). New work created for this competition; builds on the published methods cited. Every image used is listed with its source in `report/IMAGES.md`.

## Optional demo
🖥 **Local demo only (no hosted Space) — see below** — the same app runs locally with `python demo/app.py`; example outputs: `report/figures/demo_panel_dense_t150.png`, `report/figures/demo_panel_sparse_t010.png`.

## Team — Hack2Publish
* **Md. Hamid Hosen** (team leader; Kaggle [@hosen42](https://www.kaggle.com/hosen42)) — Computer Science and Engineering
* **Esfer Sami** (Kaggle [@esfersami50](https://www.kaggle.com/esfersami50)) — Computer Science and Engineering
* **Foysal** (Kaggle [@foysalemonshanto](https://www.kaggle.com/foysalemonshanto)) — Computer Science and Engineering

All three members are in Computer Science and Engineering; no member has a biology/bioengineering/clinical background, so no cross-disciplinary bonus is claimed.
