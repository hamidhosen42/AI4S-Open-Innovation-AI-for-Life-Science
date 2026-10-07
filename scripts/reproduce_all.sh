#!/usr/bin/env bash
# Regenerate every result file, figure and document of the submission, in order.
#   bash scripts/reproduce_all.sh             # everything (~6-8 h on an Apple M5; most of it training)
#   bash scripts/reproduce_all.sh --main-only # main configurations only (no ablations / neural / Cellpose)
#   bash scripts/reproduce_all.sh --quick     # no training: evaluate the released seed-0 checkpoint only
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-all}"
python scripts/download_data.py                                   # data/raw/hela_kyoto (md5-verified)

if [ "$MODE" = "--quick" ]; then
  python scripts/download_weights.py                              # weights/chipstain.pt (sha256-verified)
  mkdir -p runs/released_s0 && cp weights/chipstain.pt runs/released_s0/best.pt
  python scripts/evaluate.py --runs runs/released_s0 --tta --out outputs/quick   # -> outputs/quick/summary_test.md
  exit 0
fi

bash scripts/run_all.sh 0 1 2                                     # 9 training runs, ~9-19 min each
S="{0,1,2}"
eval python scripts/evaluate.py --runs runs/{baseline_unet,pretrained_l1,chipstain_nll}_s$S --cache outputs/cache --out outputs/multiseed
eval python scripts/evaluate.py --runs runs/{baseline_unet,pretrained_l1,chipstain_nll}_s$S --tta --cache outputs/cache --out outputs/multiseed_tta
eval python scripts/evaluate.py --runs runs/{baseline_unet,pretrained_l1,chipstain_nll}_s$S --split val --tta --cache outputs/cache_val --out outputs/multiseed_val
DIRS="outputs/multiseed outputs/multiseed_tta"

if [ "$MODE" != "--main-only" ]; then
  bash scripts/run_ablations.sh 0 1 2                             # 12 ablation runs
  eval python scripts/evaluate.py --runs runs/ablate_{scratch_l1_lr5e4,pretrained_mse,nll_beta0,nll_beta1}_s$S --out outputs/ablate
  eval python scripts/evaluate.py --runs runs/ablate_{nll_beta0,nll_beta1}_s$S --tta --out outputs/ablate_tta
  DIRS="$DIRS outputs/ablate outputs/ablate_tta"
fi

python scripts/multiseed_summary.py --dirs $DIRS --out report/results   # tables + paired tests
python scripts/ensemble_eval.py --cache outputs/cache --out outputs/ensemble
python scripts/nucleus_uncertainty.py --cache outputs/cache --out report/results
python scripts/calibration.py --cache outputs/cache --cache_val outputs/cache_val --out report/results
python scripts/proliferation.py --test report/results/multiseed_per_image.csv --val outputs/multiseed_val/per_image_val.csv --out report/results
python scripts/check_channels.py

if [ "$MODE" != "--main-only" ]; then
  eval python scripts/shift_test.py --runs runs/{baseline_unet,pretrained_l1,chipstain_nll}_s$S --out report/results
  python scripts/input_qc.py --out report/results                # input-level drift check (needs shift_test.csv)
  mkdir -p data/raw/hela_timelapse
  [ -f data/raw/hela_timelapse/R05-C03-F0.tif ] || curl -L -o data/raw/hela_timelapse/R05-C03-F0.tif "https://zenodo.org/records/6139958/files/20210904_TL2%20-%20R05-C03-F0.tif?download=1"
  python scripts/timelapse.py --weights runs/chipstain_nll_s0/best.pt --out report/results
  python scripts/download_isl_neurons.py && python scripts/neural_transfer.py --out report/results/neural
  # direct-segmentation competitor (separate environment, see scripts/cellpose_baseline.py):
  #   python3 -m venv outputs/venv_cellpose && outputs/venv_cellpose/bin/pip install "cellpose<4" tifffile pandas scikit-image
  #   outputs/venv_cellpose/bin/python scripts/cellpose_baseline.py --model <nuclei_from_bf model from Zenodo 10.5281/zenodo.6140111>
  #   outputs/venv_cellpose/bin/python scripts/cellpose_extra.py --mode shift --bf_model <same model>
  #   outputs/venv_cellpose/bin/python scripts/cellpose_extra.py --mode predicted
fi

python scripts/make_figures.py && python scripts/make_figures_extra.py && python scripts/image_register.py
python scripts/build_report.py && python scripts/build_writeup.py && python scripts/build_presentation.py
echo "done: report/ChipStain_Technical_Report.pdf, README.md, writeup/kaggle_writeup.md"
