**Category: Model & Algorithm**

# ChipStain — label-free nuclear staining that tells you where not to trust it, towards organ-on-a-chip imaging

## Demo video

▶ **{{video_link}}** (≤ 5 min)

## Code repository

**[github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science](https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science)** — MIT licence. `requirements.txt` + exact `requirements-lock.txt`, weights download script, `scripts/inference.py` (runs on a CPU — {{cpu_single}} s per 540×540 image, {{cpu_tta}} s with 8× test-time augmentation on an {{cpu_hw}}; expect a few times longer on a 4-core laptop), training / evaluation / analysis scripts that regenerate every number below, a Gradio demo and a Kaggle notebook.

## Project summary

{{summary}}

## Technical report

📄 **[Full technical report (PDF, {{report_pages}} pages incl. appendices)]({{report_link}})**

### 1. Problem
Nuclear read-outs (counts, proliferation, viability, morphology, dose–response) start from a nuclear stain, but on an organ-on-a-chip fixed DAPI ends the experiment, live DNA dyes are phototoxic and perturb the cell cycle, and reporter lines are impractical for primary or iPSC-derived (e.g. neural) tissue. In-silico labeling (ISL) predicts the stain from bright-field, but a point prediction gives no warning when it is wrong. **ChipStain asks: can we predict the nuclear stain and say — per pixel, per nucleus and per frame — how much to trust it, and does that signal beat simple alternatives?** Target users: OoC / cell-culture labs and CROs running live or high-content assays.

### 2. Data (public, CC BY 4.0, no personal or clinical data)
* **Main:** HeLa "Kyoto" (R. Guiet, EPFL BIOP; Zenodo [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)): 2-D cells in a 96-well imaging plate (CellCarrier Ultra 96) on a PerkinElmer Operetta, 20×; bright-field → mCherry-H2B. Train = wells R05-C05/C07 fields 0–19 (200 images), validation = fields 20–24 (50), **test = held-out well R05-C03 of the same plate and imaging session (125 images)**; all time-points of a field on one side. The automatic StarDist labels are used only as evaluation references, never for training (as the provider requests).
* **60 h time-lapse** of one held-out field (240 frames; Zenodo 10.5281/zenodo.6139958), evaluation only.
* **Neural cultures:** human iPSC-derived motor neurons, in-silico-labeling data Condition A (Christiansen et al., Cell 2018; CC BY 4.0), for a transfer test.
* **Direct-segmentation competitor:** the dataset author's Cellpose bright-field nuclei model (Zenodo [10.5281/zenodo.6140111](https://doi.org/10.5281/zenodo.6140111), CC BY 4.0), evaluation only.
* **No organ-on-a-chip device data were available**: OoC is the target deployment; the shift test (§4) approximates some of the optical changes a chip introduces.

### 3. Method
U-Net (segmentation_models_pytorch) with an ImageNet ResNet-34 encoder; the head predicts the mean μ and log-variance of a per-pixel Gaussian, trained with **β-NLL** (Seitzer et al., ICLR 2022; β = 0.5). At inference, **8× dihedral test-time augmentation**: σ² = mean predicted σ² + variance across the 8 views ("TTA disagreement"). Baselines: L1 U-Net from scratch and with the ImageNet encoder, each with the same TTA (σ = TTA disagreement) — the **ImageNet L1 U-Net + TTA is the matched control** (same encoder, learning rate and TTA; only head and loss differ) — de-confounded ablation arms (learning rate, L1 vs MSE vs β-NLL, β ∈ {0, 0.5, 1}), 3-seed deep ensembles, and the dataset author's Cellpose model that segments nuclei directly from bright-field. **Prior art:** ISL already predicted per-pixel intensity distributions (Christiansen 2018) and CARE used probabilistic outputs and ensembles (Weigert 2018). Our contribution is not a new mechanism but a validated, open tool: σ tested against a matched TTA baseline and uncertainty-free proxies, at pixel / nucleus / frame level, under shift, on a biological read-out, and on neurons. Main results use 3 training seeds, with 95 % CIs from a bootstrap over seeds and fields and per-seed tests; other analyses state their seeds.

### 4. Results (test well, 125 images, mean ± s.d. over 3 seeds)

{{results_table}}

* **Fidelity (vs the matched ImageNet L1 U-Net + TTA):** Pearson r {{n_pmci_pearson}} — no difference; MAE {{n_pmci_mae}}, i.e. {{n_pm_mae_rel}} % higher, and SSIM lower — a small, consistent cost that the ablation traces to the squared-error loss family; nuclei F1 {{n_pmci_seg_f1}} — no robust difference.
* **Uncertainty (the main result, vs the matched control):** ρ(σ, |error|) {{n_pmci_spearman_unc_err}}; AUSE {{n_pmci_ause}}; better in all 25 fields (seed-averaged) and significantly better in each seed. Against the scratch U-Net + TTA the advantage is similar (ρ {{n_ci_spearman_unc_err}}). σ also beats uncertainty-free proxies over the whole sparsification curve (AUSE {{n_full_ause}} vs {{n_ause_grad}} for edge strength, {{n_ause_mu}} for intensity); at a single 20 % cut-off its MAE reduction ({{n_gain}} %) is close to ranking by intensity ({{n_gain_mu}} %).
* **Imaging shift:** {{shift_bullet}}
* **Calibration:** {{calibration_bullet}}
* **Frame level:** mean σ ranks images by error (ρ {{n_full_img_rho}}), but in-distribution that is mostly cell density (predicted count alone: {{n_full_img_npred}}; the U-Net's TTA disagreement: {{n_bt_img_rho}}). It matters when something goes wrong: one ChipStain seed's three catastrophic test frames are its three highest-σ images.
* **Per nucleus:** {{nucleus_bullet}}
* **Counting vs direct segmentation:** {{counting_bullet}}
* **Biology — proliferation:** {{prolif_bullet}}
* **Neural transfer:** {{neural_bullet}}
* **Ablation:** {{ablation_bullet}}

### 5. Reliability and limitations
{{limitations_short}}

### 6. Impact
{{impact_short}}

### 7. Reproduction
```bash
git clone https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science.git && cd AI4S-Open-Innovation-AI-for-Life-Science
pip install -r requirements.txt && pip install -e .          # exact versions: requirements-lock.txt
python scripts/download_data.py && python scripts/download_weights.py
python scripts/inference.py --image demo/examples/example_bf_dense_t150.tif --weights weights/chipstain.pt --out outputs/pred --device cpu
bash scripts/run_all.sh 0 1 2 && bash scripts/run_ablations.sh 0 1 2      # ≈{{train_minutes}} min per run on an Apple M5 (MPS)
# evaluation, statistics, analyses, figures and this report: see README "Reproduce everything"
python demo/app.py --weights weights/chipstain.pt
```

### 8. Sources, licences, AI-tool disclosure
Data: Zenodo 10.5281/zenodo.6140064 / 6139958 / 6140111 (R. Guiet, EPFL BIOP) and the in-silico-labeling data (Christiansen et al. 2018) — all CC BY 4.0. Encoder: torchvision ResNet-34 ImageNet weights (BSD-3 code; ImageNet terms may apply to the weights; fetched via the Hugging Face Hub for training only). Libraries: PyTorch, segmentation_models_pytorch, timm, NumPy, SciPy, scikit-image, pandas, tifffile, Matplotlib, PyYAML, tqdm, Pillow, Gradio (BSD / MIT / Apache-2.0 / PSF / MPL-2.0). Method components re-implemented from Seitzer et al. 2022 (β-NLL), TTA uncertainty and AUSE (Ilg et al. 2018). **AI assistance: Claude Opus 5 and Claude Opus 5.5 (Anthropic), via Claude Code, were the primary implementation assistant** — they wrote most of the code, ran training and evaluation on the team's laptop under the team's direction, produced the figures and drafted the report, this Writeup and the video script. The assistant analysed the rules and proposed options for the direction and dataset; the team chose the problem, category and method, directed the work and takes full responsibility for the content (details: technical report §10). No language model or external API is used by the method itself. Released code and checkpoint: MIT (checkpoint trained on CC BY 4.0 data — attribute Guiet 2022). New work created for this competition; builds on the published methods cited. Every image used is listed with its source in `report/IMAGES.md`.

## Optional demo
🖥 **{{demo_link}}** — the Gradio demo runs locally with `python demo/app.py --weights weights/chipstain.pt` (it shows the prediction, the σ map and a nuclei count, and warns when mean σ exceeds a fixed threshold); example outputs: `report/figures/demo_panel_dense_t150.png`, `report/figures/demo_panel_sparse_t010.png`.

## Team — Hack2Publish
* **Md. Hamid Hosen** (team leader; Kaggle [@hosen42](https://www.kaggle.com/hosen42)) — Computer Science and Engineering
* **Esfer Sami** (Kaggle [@esfersami50](https://www.kaggle.com/esfersami50)) — Computer Science and Engineering
* **Foysal** (Kaggle [@foysalemonshanto](https://www.kaggle.com/foysalemonshanto)) — Computer Science and Engineering
* **Kahakashan Ashraf** (Kaggle [@kahakashanashraf](https://www.kaggle.com/kahakashanashraf)) — Computer Science and Engineering

All four members are in Computer Science and Engineering; no member has a biology/bioengineering/clinical background, so no cross-disciplinary bonus is claimed.
