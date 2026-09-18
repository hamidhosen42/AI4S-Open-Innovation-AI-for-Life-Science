#!/usr/bin/env bash
# Train all configs for the given seeds sequentially. Usage: bash scripts/run_all.sh 0 1 2
set -e
cd "$(dirname "$0")/.."
for seed in "$@"; do
  for cfg in baseline pretrained ours; do
    echo "=== $cfg seed $seed ==="
    python scripts/train.py --config configs/$cfg.yaml --seed $seed
  done
done
