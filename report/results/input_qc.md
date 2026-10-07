## Shift detection (AUROC, shifted vs clean test images; mean ± s.d. over 3 ChipStain seeds)

| check | blur | noise | modality (DPC) | failure AUROC (r < 0.5 vs r > 0.7) |
|---|---|---|---|---|
| focus (variance of Laplacian) | 1.00 ± 0.00 | 0.84 ± 0.00 | 1.00 ± 0.00 | 0.95 ± 0.01 |
| encoder features (Mahalanobis) | 1.00 ± 0.00 | 0.98 ± 0.03 | 0.99 ± 0.01 | 0.99 ± 0.01 |
| total σ (for comparison) | 0.34 ± 0.57 | 1.00 ± 0.00 | 0.33 ± 0.58 | 0.56 ± 0.38 |

## Validation-calibrated gate (95th percentile on clean validation images): fraction of images flagged, mean over seeds (min–max)

| check | clean | blur σ=1 px | blur σ=2 px | blur σ=4 px | noise 10 % | noise 20 % | modality: DPC |
|---|---|---|---|---|---|---|---|
| focus | 16 % | 100 % | 100 % | 100 % | 32 % | 100 % | 100 % |
| encoder features | 9 % (0–22) | 100 % | 100 % | 100 % | 67 % (0–100) | 67 % (0–100) | 100 % |
| either | 23 % (16–34) | 100 % | 100 % | 100 % | 77 % (32–100) | 100 % | 100 % |
| focus or σ gate | 17 % (16–20) | 100 % | 100 % | 100 % | 98 % (94–100) | 100 % | 100 % |
