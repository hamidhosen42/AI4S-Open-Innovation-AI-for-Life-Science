#!/usr/bin/env bash
# Extra ablation arms (3 seeds each). Usage: bash scripts/run_ablations.sh 0 1 2
set -e
cd "$(dirname "$0")/.."
for seed in "$@"; do
  for cfg in ablate_scratch_l1_lr5e4 ablate_pretrained_mse ablate_nll_beta0 ablate_nll_beta1; do
    echo "=== $cfg seed $seed ==="
    python scripts/train.py --config configs/$cfg.yaml --seed $seed
  done
done
