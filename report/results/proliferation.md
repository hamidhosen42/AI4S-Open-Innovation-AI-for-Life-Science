Reference (StarDist on real H2B): population doubling time 23.3 h (geometric mean of 25 fields).
Gate thresholds (relative mean σ, 95th percentile on validation): ChipStain + TTA 1.55, U-Net + TTA 1.57

| model | seed | frames used | doubling time (h) | mean bias | mean abs. error | fields to review | frames flagged | failed frames (F1 < 0.5) caught |
|---|---|---|---|---|---|---|---|---|
| U-Net + TTA | 0 | all frames | 24.9 | +7.7 % | 11.3 % | 0 | 0 | —/6 |
| U-Net + TTA | 0 | σ-gated | 24.3 | +4.1 % | 8.9 % | 6 | 26 | 4/6 |
| U-Net + TTA | 1 | all frames | 24.7 | +6.8 % | 10.7 % | 0 | 0 | —/4 |
| U-Net + TTA | 1 | σ-gated | 24.0 | +3.6 % | 8.5 % | 4 | 22 | 4/4 |
| U-Net + TTA | 2 | all frames | 29.0 | +26.7 % | 28.6 % | 0 | 0 | —/46 |
| U-Net + TTA | 2 | σ-gated | 27.5 | +18.7 % | 21.0 % | 5 | 24 | 8/46 |
| ChipStain + TTA | 0 | all frames | 23.8 | +3.2 % | 9.2 % | 0 | 0 | —/3 |
| ChipStain + TTA | 0 | σ-gated | 23.5 | +1.5 % | 8.6 % | 3 | 15 | 0/3 |
| ChipStain + TTA | 1 | all frames | 26.1 | +15.1 % | 19.6 % | 2 | 0 | —/9 |
| ChipStain + TTA | 1 | σ-gated | 24.8 | +7.1 % | 12.9 % | 5 | 29 | 8/9 |
| ChipStain + TTA | 2 | all frames | 25.1 | +9.1 % | 14.0 % | 0 | 0 | —/4 |
| ChipStain + TTA | 2 | σ-gated | 23.9 | +3.7 % | 10.0 % | 2 | 16 | 1/4 |
