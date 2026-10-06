Reference (StarDist on real H2B): population doubling time 23.3 h (geometric mean of 25 fields).
Gate thresholds (relative mean σ, 95th percentile on validation): ChipStain + TTA 1.55, ImageNet-L1 U-Net + TTA 1.59, Scratch U-Net + TTA 1.57

| model | seed | estimate | fields used | doubling time (h) | mean signed bias | mean abs. error | collapsed fields | fields to review | frames flagged | failed frames (F1 < 0.5) caught |
|---|---|---|---|---|---|---|---|---|---|---|
| Scratch U-Net + TTA | 0 | all frames | 25 | 24.9 | +7.7 % | 11.3 % | 0 | — | — | — |
| Scratch U-Net + TTA | 0 | all frames, same fields as gated | 19 | 24.3 | +4.3 % | 9.0 % | — | — | — | — |
| Scratch U-Net + TTA | 0 | σ-gated | 19 | 24.3 | +4.1 % | 8.9 % | — | 6 | 26 | 4/6 |
| Scratch U-Net + TTA | 1 | all frames | 25 | 24.7 | +6.8 % | 10.7 % | 0 | — | — | — |
| Scratch U-Net + TTA | 1 | all frames, same fields as gated | 21 | 24.2 | +4.2 % | 8.8 % | — | — | — | — |
| Scratch U-Net + TTA | 1 | σ-gated | 21 | 24.0 | +3.6 % | 8.5 % | — | 4 | 22 | 4/4 |
| Scratch U-Net + TTA | 2 | all frames | 25 | 29.0 | +26.7 % | 28.6 % | 0 | — | — | — |
| Scratch U-Net + TTA | 2 | all frames, same fields as gated | 20 | 27.6 | +19.4 % | 21.6 % | — | — | — | — |
| Scratch U-Net + TTA | 2 | σ-gated | 20 | 27.5 | +18.7 % | 21.0 % | — | 5 | 24 | 8/46 |
| ChipStain + TTA | 0 | all frames | 25 | 23.8 | +3.2 % | 9.2 % | 0 | — | — | — |
| ChipStain + TTA | 0 | all frames, same fields as gated | 22 | 23.5 | +1.3 % | 8.1 % | — | — | — | — |
| ChipStain + TTA | 0 | σ-gated | 22 | 23.5 | +1.5 % | 8.6 % | — | 3 | 15 | 0/3 |
| ChipStain + TTA | 1 | all frames | 23 | 26.1 | +15.1 % | 19.6 % | 2 | — | — | — |
| ChipStain + TTA | 1 | all frames, same fields as gated | 20 | 25.0 | +7.9 % | 13.1 % | — | — | — | — |
| ChipStain + TTA | 1 | σ-gated | 20 | 24.8 | +7.1 % | 12.9 % | — | 5 | 29 | 8/9 |
| ChipStain + TTA | 2 | all frames | 25 | 25.1 | +9.1 % | 14.0 % | 0 | — | — | — |
| ChipStain + TTA | 2 | all frames, same fields as gated | 23 | 24.9 | +8.2 % | 13.6 % | — | — | — | — |
| ChipStain + TTA | 2 | σ-gated | 23 | 23.9 | +3.7 % | 10.0 % | — | 2 | 16 | 1/4 |
| ImageNet-L1 U-Net + TTA | 0 | all frames | 25 | 25.2 | +9.0 % | 12.4 % | 0 | — | — | — |
| ImageNet-L1 U-Net + TTA | 0 | all frames, same fields as gated | 20 | 24.8 | +6.9 % | 11.1 % | — | — | — | — |
| ImageNet-L1 U-Net + TTA | 0 | σ-gated | 20 | 24.8 | +7.0 % | 11.2 % | — | 5 | 28 | 4/12 |
| ImageNet-L1 U-Net + TTA | 1 | all frames | 25 | 25.2 | +8.8 % | 12.0 % | 0 | — | — | — |
| ImageNet-L1 U-Net + TTA | 1 | all frames, same fields as gated | 18 | 25.4 | +8.9 % | 12.6 % | — | — | — | — |
| ImageNet-L1 U-Net + TTA | 1 | σ-gated | 18 | 24.7 | +5.9 % | 9.8 % | — | 7 | 29 | 4/12 |
| ImageNet-L1 U-Net + TTA | 2 | all frames | 25 | 26.0 | +12.8 % | 15.9 % | 0 | — | — | — |
| ImageNet-L1 U-Net + TTA | 2 | all frames, same fields as gated | 22 | 25.3 | +9.1 % | 12.7 % | — | — | — | — |
| ImageNet-L1 U-Net + TTA | 2 | σ-gated | 22 | 25.1 | +8.5 % | 12.1 % | — | 3 | 14 | 3/19 |
