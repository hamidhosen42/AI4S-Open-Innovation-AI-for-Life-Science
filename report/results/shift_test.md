## Failure detection (AUROC of per-image mean σ: failed images r < 0.5 vs successful r > 0.7, all shifted conditions; mean ± s.d. over seeds)

| model | σ term | failure AUROC | failed images | within-condition ρ(σ, MAE) |
|---|---|---|---|---|
| U-Net + TTA | total σ | 0.48 ± 0.10 | 640 | 0.42 ± 0.27 |
| U-Net + TTA | TTA disagreement | 0.48 ± 0.10 | 640 | 0.42 ± 0.27 |
| ChipStain + TTA | total σ | nan | 768 | 0.69 ± 0.03 |
| ChipStain + TTA | learned head (TTA mean) | nan | 768 | 0.43 ± 0.28 |
| ChipStain + TTA | TTA disagreement | nan | 768 | 0.51 ± 0.17 |
| ChipStain + TTA | learned head, single pass | nan | 768 | 0.42 ± 0.30 |
| + ImageNet encoder + TTA | total σ | 0.58 ± 0.51 | 695 | 0.51 ± 0.33 |
| + ImageNet encoder + TTA | TTA disagreement | 0.58 ± 0.51 | 695 | 0.51 ± 0.33 |

## Shift detection (AUROC of per-image mean σ, shifted vs clean; 0.5 = no signal, < 0.5 = σ falls under shift)

| model | σ term | blur | noise | modality (DPC) |
|---|---|---|---|---|
| U-Net + TTA | total σ | 0.43 ± 0.41 | 1.00 ± 0.00 | 0.85 ± 0.20 |
| U-Net + TTA | TTA disagreement | 0.43 ± 0.41 | 1.00 ± 0.00 | 0.85 ± 0.20 |
| ChipStain + TTA | total σ | 0.34 ± 0.57 | 1.00 ± 0.00 | 0.33 ± 0.58 |
| ChipStain + TTA | learned head (TTA mean) | 0.30 ± 0.51 | 1.00 ± 0.00 | 0.32 ± 0.56 |
| ChipStain + TTA | TTA disagreement | 0.35 ± 0.57 | 0.93 ± 0.04 | 0.39 ± 0.53 |
| ChipStain + TTA | learned head, single pass | 0.26 ± 0.43 | 1.00 ± 0.00 | 0.30 ± 0.52 |
| + ImageNet encoder + TTA | total σ | 0.71 ± 0.49 | 0.98 ± 0.01 | 0.42 ± 0.47 |
| + ImageNet encoder + TTA | TTA disagreement | 0.71 ± 0.49 | 0.98 ± 0.01 | 0.42 ± 0.47 |

## Accuracy and σ per condition (mean over seeds)

| model | condition | Pearson r | MAE | mean σ / clean |
|---|---|---|---|---|
| U-Net + TTA | clean | 0.769 | 0.0282 | 1.00× |
| U-Net + TTA | blur σ=1 px | 0.441 | 0.0318 | 1.35× |
| U-Net + TTA | blur σ=2 px | 0.268 | 0.0338 | 1.16× |
| U-Net + TTA | blur σ=4 px | 0.135 | 0.0352 | 1.03× |
| U-Net + TTA | noise 10 % | 0.540 | 0.0612 | 5.00× |
| U-Net + TTA | noise 20 % | 0.376 | 0.0850 | 6.70× |
| U-Net + TTA | modality: DPC | 0.560 | 0.0411 | 6.23× |
| ChipStain + TTA | clean | 0.772 | 0.0305 | 1.00× |
| ChipStain + TTA | blur σ=1 px | 0.321 | 0.0584 | 2.04× |
| ChipStain + TTA | blur σ=2 px | 0.179 | 0.0996 | 3.00× |
| ChipStain + TTA | blur σ=4 px | 0.095 | 0.1168 | 3.30× |
| ChipStain + TTA | noise 10 % | 0.478 | 0.1188 | 3.39× |
| ChipStain + TTA | noise 20 % | 0.309 | 0.1785 | 4.27× |
| ChipStain + TTA | modality: DPC | 0.368 | 0.1124 | 3.84× |
| + ImageNet encoder + TTA | clean | 0.771 | 0.0277 | 1.00× |
| + ImageNet encoder + TTA | blur σ=1 px | 0.235 | 0.0566 | 373.30× |
| + ImageNet encoder + TTA | blur σ=2 px | 0.093 | 0.0768 | 1553.06× |
| + ImageNet encoder + TTA | blur σ=4 px | 0.027 | 0.0753 | 2280.20× |
| + ImageNet encoder + TTA | noise 10 % | 0.620 | 0.0507 | 2.78× |
| + ImageNet encoder + TTA | noise 20 % | 0.349 | 0.0955 | 5.51× |
| + ImageNet encoder + TTA | modality: DPC | 0.443 | 0.0353 | 1.36× |

## Validation-calibrated gate (threshold = 95th percentile of mean σ on clean validation images): fraction of images flagged

| model | clean | blur σ=1 px | blur σ=2 px | blur σ=4 px | noise 10 % | noise 20 % | modality: DPC |
|---|---|---|---|---|---|---|---|
| U-Net + TTA | 1 % | 17 % | 20 % | 23 % | 100 % | 100 % | 43 % |
| ChipStain + TTA | 1 % | 32 % | 33 % | 33 % | 97 % | 100 % | 33 % |
| + ImageNet encoder + TTA | 4 % | 62 % | 66 % | 67 % | 59 % | 67 % | 31 % |
