## Does each uncertainty term separate shifted from clean images? (AUROC of per-image mean σ; 0.5 = no signal)

| model | σ term | defocus σ=1 | defocus σ=2 | defocus σ=4 | noise 5 % | noise 10 % | noise 20 % | contrast ×0.5 | contrast ×0.25 | modality: DPC |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net + TTA (seed 0) | total (TTA) | 0.30 | 0.08 | 0.00 | 0.77 | 1.00 | 1.00 | 0.51 | 0.58 | 0.62 |
| U-Net + TTA (seed 0) | TTA view s.d. | 0.30 | 0.08 | 0.00 | 0.77 | 1.00 | 1.00 | 0.51 | 0.58 | 0.62 |
| ChipStain + TTA (seed 0) | total (TTA) | 1.00 | 1.00 | 1.00 | 0.46 | 1.00 | 1.00 | 0.39 | 0.39 | 1.00 |
| ChipStain + TTA (seed 0) | learned head (TTA mean) | 0.83 | 0.93 | 0.92 | 0.58 | 1.00 | 1.00 | 0.46 | 0.48 | 0.97 |
| ChipStain + TTA (seed 0) | TTA view s.d. | 1.00 | 1.00 | 1.00 | 0.31 | 0.89 | 0.93 | 0.33 | 0.26 | 1.00 |
| ChipStain + TTA (seed 0) | learned head, single pass | 0.66 | 0.83 | 0.79 | 0.63 | 1.00 | 1.00 | 0.51 | 0.56 | 0.90 |

### baseline_unet_s0
Spearman ρ(mean σ, MAE) over all 500 (image, condition) pairs: 0.725

| condition | mean σ | σ / clean | MAE | Pearson r |
|---|---|---|---|---|
| clean | 0.0059 | 1.00× | 0.0283 | 0.784 |
| defocus σ=1 | 0.0045 | 0.76× | 0.0317 | 0.449 |
| defocus σ=2 | 0.0020 | 0.33× | 0.0322 | 0.318 |
| defocus σ=4 | 0.0006 | 0.11× | 0.0332 | 0.189 |
| noise 5 % | 0.0098 | 1.67× | 0.0327 | 0.750 |
| noise 10 % | 0.0348 | 5.91× | 0.0715 | 0.496 |
| noise 20 % | 0.0468 | 7.96× | 0.1046 | 0.321 |
| contrast ×0.5 | 0.0060 | 1.02× | 0.0291 | 0.783 |
| contrast ×0.25 | 0.0067 | 1.15× | 0.0306 | 0.778 |
| modality: DPC | 0.0065 | 1.11× | 0.0303 | 0.644 |

### chipstain_nll_s0
Spearman ρ(mean σ, MAE) over all 500 (image, condition) pairs: 0.957

| condition | mean σ | σ / clean | MAE | Pearson r |
|---|---|---|---|---|
| clean | 0.0476 | 1.00× | 0.0300 | 0.782 |
| defocus σ=1 | 0.2572 | 5.40× | 0.1108 | 0.123 |
| defocus σ=2 | 0.4048 | 8.50× | 0.2297 | 0.078 |
| defocus σ=4 | 0.4510 | 9.46× | 0.2795 | 0.094 |
| noise 5 % | 0.0443 | 0.93× | 0.0329 | 0.778 |
| noise 10 % | 0.1477 | 3.10× | 0.1179 | 0.496 |
| noise 20 % | 0.1977 | 4.15× | 0.1893 | 0.321 |
| contrast ×0.5 | 0.0413 | 0.87× | 0.0295 | 0.787 |
| contrast ×0.25 | 0.0403 | 0.85× | 0.0309 | 0.784 |
| modality: DPC | 0.5131 | 10.77× | 0.2683 | 0.081 |

