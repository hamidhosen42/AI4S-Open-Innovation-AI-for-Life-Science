# ChipStain — uncertainty-aware label-free nuclear staining

**Category: Model & Algorithm** · Entry for [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien) (AI + Organ-on-a-Chip, 5th Pazhou Algorithm Competition)

ChipStain predicts the **nuclear fluorescence channel (H2B / DNA) from a plain bright-field image**, and — unlike a standard in-silico-labeling U-Net — also outputs a **per-pixel uncertainty map** that tells the biologist *where not to trust the prediction*. Label-free nuclear readouts matter for organ-on-a-chip (OoC) work because fluorescent staining is phototoxic, ends a live time-lapse, and adds reagent and handling cost on small microfluidic devices.

<p align="center"><img src="report/figures/qualitative_chipstain_nll_s0_tta.png" width="900" alt="Bright-field → predicted nuclei with uncertainty"></p>

## What is new

| Component | Standard ISL U-Net | ChipStain |
|---|---|---|
| Output | fluorescence intensity | fluorescence **mean μ + log-variance** (heteroscedastic Gaussian) |
| Loss | L1 / L2 | **β-NLL** (Seitzer et al. 2022) — stable variance learning |
| Encoder | from scratch | ImageNet-pretrained ResNet-34 (`segmentation_models_pytorch`) |
| Test time | single pass | 8-fold dihedral TTA → aleatoric (mean σ²) **+ epistemic** (variance across views) |
| Validation | image metrics only | image metrics **+ downstream nuclei-segmentation F1 + uncertainty calibration** (Spearman ρ, sparsification / AUSE) |

## Results (held-out well R05-C03, 125 images, mean ± std over images)

See `outputs/eval/summary_test.md` after running the evaluation; the numbers reported in the technical report are reproduced by the commands below.

<!-- RESULTS_TABLE -->

## Quick start

```bash
git clone https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science.git
cd AI4S-Open-Innovation-AI-for-Life-Science
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 1. data (Zenodo, CC BY 4.0, ~757 MB)
python scripts/download_data.py

# 2. pretrained weights (or train your own, step 4)
python scripts/download_weights.py          # -> weights/chipstain.pt

# 3. predict on one bright-field image (CPU is fine, ~5 s with TTA)
python scripts/inference.py --image demo/examples/example_bf.tif --weights weights/chipstain.pt --out outputs/pred
#    -> outputs/pred/prediction.tif, uncertainty.tif, panel.png

# 4. reproduce training + evaluation (≈30 min per run on an Apple M-series / any GPU)
bash scripts/run_all.sh 0            # baseline, pretrained, ours (seed 0)
python scripts/evaluate.py --runs runs/baseline_unet_s0 runs/pretrained_l1_s0 runs/chipstain_nll_s0 --out outputs/eval
python scripts/evaluate.py --runs runs/chipstain_nll_s0 --tta --out outputs/eval
python scripts/make_figures.py

# 5. interactive demo
python demo/app.py --weights weights/chipstain.pt     # http://localhost:7860
```

## Repository layout

```
chipstain/            package: data.py, model.py, losses.py, metrics.py
scripts/              download_data.py, download_weights.py, train.py, evaluate.py, inference.py, make_figures.py, run_all.sh
configs/              baseline.yaml, pretrained.yaml, ours.yaml
demo/                 Gradio app + example image
notebooks/            Kaggle notebook version of train + evaluate
report/               technical report (PDF) and figures
docs/                 competition plan
DATA.md               data sources, licenses, splits, pre-processing
AI_ASSISTANCE.md      pre-trained models and AI tools disclosure
```

## Inputs / outputs

* **Input**: single-channel bright-field image (TIFF/PNG, any size ≥ 64 px; normalised per image, so camera offset/gain do not matter).
* **Output**: `prediction.tif` — nuclear fluorescence in normalised units [0, 1] (`(I − 600)/(20000 − 600)` of the 16-bit training data); `uncertainty.tif` — predicted σ in the same units; `panel.png` — side-by-side view.

## Data, licenses, disclosure

* Data: HeLa "Kyoto" paired bright-field / H2B dataset, Zenodo [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064), CC BY 4.0 — details in [DATA.md](DATA.md).
* Pre-trained encoder and AI-tool use: [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
* Code: MIT ([LICENSE](LICENSE)).

## Team

* Md. Hamid Hosen — AI/ML ([@hamidhosen42](https://github.com/hamidhosen42))

## Citation of the data

Romain Guiet (EPFL BioImaging & Optics Platform), *HeLa "Kyoto" cells under the scope* (2022), Zenodo, doi:10.5281/zenodo.6139958; and *Automatic labelling of HeLa "Kyoto" cells using Deep Learning tools* (2022), Zenodo, doi:10.5281/zenodo.6140064. Both CC BY 4.0.
