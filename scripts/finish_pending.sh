#!/usr/bin/env bash
# One-off: wait for the long runs started on 2026-10-02, run the remaining analyses, then rebuild everything.
cd "$(dirname "$0")/.."
wait_for() { while ! grep -q "$2" "$1" 2>/dev/null; do sleep 30; done; }
wait_for outputs/multiseed_val.log VAL_DONE
wait_for outputs/calibration.log CAL_DONE
wait_for outputs/prolif.log PROLIF_DONE
wait_for outputs/shift.log SHIFT_DONE
while pgrep -f run_ablations.sh >/dev/null; do sleep 60; done
python scripts/evaluate.py --runs runs/ablate_{scratch_l1_lr5e4,pretrained_mse,nll_beta0,nll_beta1}_s{0,1,2} --out outputs/ablate > outputs/ablate_eval.log 2>&1
python scripts/evaluate.py --runs runs/ablate_{nll_beta0,nll_beta1}_s{0,1,2} --tta --out outputs/ablate_tta >> outputs/ablate_eval.log 2>&1
python scripts/multiseed_summary.py --dirs outputs/multiseed outputs/multiseed_tta outputs/ablate outputs/ablate_tta --out report/results > outputs/summary.log 2>&1
python scripts/ensemble_eval.py --cache outputs/cache --out outputs/ensemble > outputs/ensemble.log 2>&1
python scripts/proliferation.py --test report/results/multiseed_per_image.csv --val outputs/multiseed_val/per_image_val.csv --out report/results > outputs/prolif.log 2>&1
python scripts/check_channels.py > /dev/null 2>&1
while pgrep -f neural_transfer.py >/dev/null; do sleep 60; done
while pgrep -f cellpose_extra.py >/dev/null; do sleep 60; done
bash scripts/build_all.sh > outputs/build_all.log 2>&1
echo FINISHED >> outputs/build_all.log
