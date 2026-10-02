| Model | Pearson r ↑ | SSIM ↑ | PSNR ↑ | seg-F1 ↑ | ρ(σ, err) ↑ | AUSE ↓ | image ρ(σ̄, MAE) ↑ |
|---|---|---|---|---|---|---|---|
| U-Net baseline (L1) | 0.749 ± 0.027 | 0.826 ± 0.003 | 24.15 ± 0.24 | 0.648 ± 0.089 | — | — | — |
| + ImageNet encoder (L1) | 0.755 ± 0.005 | 0.823 ± 0.005 | 24.25 ± 0.07 | 0.669 ± 0.021 | — | — | — |
| U-Net baseline + TTA (σ = view variance) | 0.764 ± 0.024 | 0.831 ± 0.003 | 24.36 ± 0.25 | 0.665 ± 0.083 | 0.283 ± 0.071 | 0.258 ± 0.025 | 0.903 ± 0.029 |
| ChipStain (β-NLL head) | 0.757 ± 0.015 | 0.810 ± 0.018 | 23.92 ± 0.36 | 0.697 ± 0.012 | 0.430 ± 0.006 | 0.303 ± 0.072 | 0.948 ± 0.022 |
| ChipStain + TTA (full) | 0.768 ± 0.012 | 0.814 ± 0.022 | 24.11 ± 0.32 | 0.708 ± 0.011 | 0.469 ± 0.027 | 0.177 ± 0.021 | 0.908 ± 0.067 |
| Real fluorescence, same segmentation pipeline (ceiling) | — | — | — | 0.765 | — | — | — |

Mean ± std over 3 seeds (each seed: mean over the 125 test images of well R05-C03).

Paired tests (Wilcoxon signed-rank, field-level n = 25, Holm-corrected):

| A vs B | metric | A | B | A better in fields | A better in seeds | p (Holm) |
|---|---|---|---|---|---|---|
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view variance) | spearman_unc_err | 0.469 | 0.283 | 25/25 | 3/3 | 7.7e-07 |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view variance) | ause | 0.177 | 0.258 | 25/25 | 3/3 | 7.7e-07 |
| ChipStain (β-NLL head) vs U-Net baseline + TTA (σ = view variance) | spearman_unc_err | 0.430 | 0.283 | 25/25 | 3/3 | 7.7e-07 |
| ChipStain (β-NLL head) vs U-Net baseline + TTA (σ = view variance) | ause | 0.303 | 0.258 | 12/25 | 1/3 | 0.57 |
| ChipStain + TTA (full) vs U-Net baseline (L1) | pearson | 0.768 | 0.749 | 23/25 | 2/3 | 0.00045 |
| ChipStain + TTA (full) vs U-Net baseline (L1) | ssim | 0.814 | 0.826 | 2/25 | 1/3 | 4.2e-06 |
| ChipStain + TTA (full) vs U-Net baseline (L1) | seg_f1 | 0.708 | 0.648 | 24/25 | 2/3 | 7.5e-06 |
| ChipStain (β-NLL head) vs U-Net baseline (L1) | pearson | 0.757 | 0.749 | 19/25 | 2/3 | 0.041 |
| ChipStain (β-NLL head) vs U-Net baseline (L1) | ssim | 0.810 | 0.826 | 4/25 | 1/3 | 0.00045 |
| ChipStain (β-NLL head) vs U-Net baseline (L1) | seg_f1 | 0.697 | 0.648 | 23/25 | 2/3 | 9.1e-06 |
| + ImageNet encoder (L1) vs U-Net baseline (L1) | pearson | 0.755 | 0.749 | 18/25 | 1/3 | 0.015 |
| + ImageNet encoder (L1) vs U-Net baseline (L1) | ssim | 0.823 | 0.826 | 12/25 | 1/3 | 0.57 |
| + ImageNet encoder (L1) vs U-Net baseline (L1) | seg_f1 | 0.669 | 0.648 | 21/25 | 1/3 | 3.7e-05 |
