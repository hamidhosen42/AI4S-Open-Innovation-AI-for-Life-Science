## Main comparison (mean ± sd over 3 seeds; each seed = mean over the 125 test images of well R05-C03)

| Model | Pearson r ↑ | SSIM ↑ | MAE ↓ | PSNR ↑ | seg-F1 ↑ | ρ(σ, err) ↑ | AUSE ↓ | MAE drop, top-20 % σ removed ↑ |
|---|---|---|---|---|---|---|---|---|
| U-Net baseline (scratch, L1, lr 1e-3) | 0.749 ± 0.027 | 0.826 ± 0.003 | 0.0299 ± 0.0006 | 24.15 ± 0.24 | 0.648 ± 0.089 | — | — | — |
| + ImageNet encoder (L1, lr 5e-4) | 0.755 ± 0.005 | 0.823 ± 0.005 | 0.0296 ± 0.0003 | 24.25 ± 0.07 | 0.669 ± 0.021 | — | — | — |
| U-Net baseline + TTA (σ = view s.d.) | 0.764 ± 0.024 | 0.831 ± 0.003 | 0.0294 ± 0.0006 | 24.36 ± 0.25 | 0.665 ± 0.083 | 0.283 ± 0.071 | 0.258 ± 0.025 | 40.5 ± 3.2 % |
| + ImageNet encoder + TTA (σ = view s.d.) | 0.768 ± 0.006 | 0.827 ± 0.005 | 0.0291 ± 0.0003 | 24.47 ± 0.10 | 0.690 ± 0.019 | 0.317 ± 0.037 | 0.248 ± 0.011 | 42.9 ± 1.8 % |
| ChipStain (β-NLL, β = 0.5) | 0.757 ± 0.015 | 0.810 ± 0.018 | 0.0324 ± 0.0013 | 23.92 ± 0.36 | 0.697 ± 0.012 | 0.430 ± 0.006 | 0.303 ± 0.072 | 45.6 ± 1.6 % |
| ChipStain + TTA (full) | 0.768 ± 0.012 | 0.814 ± 0.022 | 0.0323 ± 0.0007 | 24.11 ± 0.32 | 0.708 ± 0.011 | 0.469 ± 0.027 | 0.177 ± 0.021 | 49.7 ± 4.2 % |
| Real fluorescence, same segmentation pipeline (reference level, not a bound) | — | — | — | — | 0.765 | — | — | — |

## Does σ beat uncertainty-free proxies at ranking pixel errors?

Proxies are computed from the model's own prediction: intensity = predicted μ, edges = Sobel edge strength of μ.

| Model | ρ: σ | ρ: intensity | ρ: edges | AUSE: σ | AUSE: intensity | AUSE: edges | gain: σ | gain: intensity | gain: edges | gain: oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net baseline + TTA (σ = view s.d.) | 0.283 ± 0.071 | 0.182 ± 0.025 | 0.297 ± 0.045 | 0.258 ± 0.025 | 0.310 ± 0.009 | 0.256 ± 0.017 | 40.5 ± 3.2 % | 39.2 ± 1.8 % | 40.0 ± 2.5 % | 55.1 ± 2.5 % |
| + ImageNet encoder + TTA (σ = view s.d.) | 0.317 ± 0.037 | 0.185 ± 0.033 | 0.297 ± 0.032 | 0.248 ± 0.011 | 0.379 ± 0.121 | 0.324 ± 0.124 | 42.9 ± 1.8 % | 39.4 ± 3.7 % | 39.5 ± 3.3 % | 56.6 ± 1.6 % |
| ChipStain (β-NLL, β = 0.5) | 0.430 ± 0.006 | 0.306 ± 0.020 | 0.395 ± 0.017 | 0.303 ± 0.072 | 0.296 ± 0.043 | 0.280 ± 0.087 | 45.6 ± 1.6 % | 47.8 ± 4.6 % | 44.3 ± 1.7 % | 59.3 ± 3.4 % |
| ChipStain + TTA (full) | 0.469 ± 0.027 | 0.320 ± 0.041 | 0.414 ± 0.030 | 0.177 ± 0.021 | 0.308 ± 0.104 | 0.271 ± 0.060 | 49.7 ± 4.2 % | 47.7 ± 5.4 % | 43.4 ± 3.0 % | 58.5 ± 3.7 % |

## Image level: σ versus cell density

| Model | ρ(mean σ, MAE) | ρ(n_pred, MAE) | partial ρ(σ, MAE) given n_pred | ρ(mean σ, 1−F1) | ρ(n_pred, 1−F1) | ρ(mean σ, 1−r) |
|---|---|---|---|---|---|---|
| U-Net baseline + TTA (σ = view s.d.) | 0.903 ± 0.029 | 0.872 ± 0.021 | 0.515 ± 0.108 | 0.406 ± 0.044 | 0.292 ± 0.060 | 0.648 ± 0.022 |
| + ImageNet encoder + TTA (σ = view s.d.) | 0.771 ± 0.275 | 0.896 ± 0.022 | 0.462 ± 0.218 | 0.310 ± 0.081 | 0.276 ± 0.035 | 0.586 ± 0.049 |
| ChipStain (β-NLL, β = 0.5) | 0.948 ± 0.022 | 0.907 ± 0.031 | 0.596 ± 0.233 | 0.392 ± 0.108 | 0.308 ± 0.037 | 0.622 ± 0.072 |
| ChipStain + TTA (full) | 0.908 ± 0.067 | 0.891 ± 0.028 | 0.628 ± 0.290 | 0.426 ± 0.135 | 0.303 ± 0.034 | 0.650 ± 0.089 |

## Paired tests

Inference rests on the 95 % CI of A − B from a hierarchical bootstrap that resamples seeds (per configuration) and fields, and on per-seed tests (field-level Wilcoxon within each seed pair, p < 0.05): 'seeds better / worse' counts the seed pairs in which A is significantly better / worse. The pooled field-level Wilcoxon averages over seeds and only measures consistency within this single well (one plate, one imaging session); it is shown, Holm-corrected over all tests, for the uncertainty metrics only.

| A vs B | metric | A | B | A − B (95 % CI, seeds + fields) | seeds sig. better / worse | fields A better (seed-avg) | p (pooled, Holm) |
|---|---|---|---|---|---|---|---|
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | pearson | 0.7681 | 0.7644 | +0.0037 (-0.0211, +0.0304) | 1 / 0 of 3 | 18/25 | — |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | ssim | 0.8137 | 0.8310 | -0.0173 (-0.0361, +0.0061) | 0 / 2 of 3 | 0/25 | — |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | mae | 0.0323 | 0.0294 | +0.0029 (+0.0009, +0.0054) | 0 / 2 of 3 | 6/25 | — |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | psnr | 24.1148 | 24.3614 | -0.2466 (-0.7972, +0.1756) | 0 / 0 of 3 | 11/25 | — |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | seg_f1 | 0.7082 | 0.6651 | +0.0431 (-0.0171, +0.1351) | 1 / 0 of 3 | 24/25 | — |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | spearman_unc_err | 0.4689 | 0.2832 | +0.1857 (+0.1201, +0.2641) | 3 / 0 of 3 | 25/25 | 2.4e-06 |
| ChipStain + TTA (full) vs U-Net baseline + TTA (σ = view s.d.) | ause | 0.1775 | 0.2582 | -0.0807 (-0.1136, -0.0493) | 3 / 0 of 3 | 25/25 | 2.4e-06 |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | pearson | 0.7681 | 0.7678 | +0.0003 (-0.0194, +0.0133) | 2 / 0 of 3 | 15/25 | — |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | ssim | 0.8137 | 0.8271 | -0.0134 (-0.0323, +0.0103) | 0 / 2 of 3 | 3/25 | — |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | mae | 0.0323 | 0.0291 | +0.0032 (+0.0012, +0.0056) | 0 / 2 of 3 | 3/25 | — |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | psnr | 24.1148 | 24.4676 | -0.3528 (-0.9085, -0.0132) | 0 / 0 of 3 | 7/25 | — |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | seg_f1 | 0.7082 | 0.6898 | +0.0184 (-0.0092, +0.0419) | 2 / 0 of 3 | 20/25 | — |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | spearman_unc_err | 0.4689 | 0.3165 | +0.1523 (+0.1057, +0.2010) | 3 / 0 of 3 | 25/25 | 2.4e-06 |
| ChipStain + TTA (full) vs + ImageNet encoder + TTA (σ = view s.d.) | ause | 0.1775 | 0.2477 | -0.0702 (-0.0947, -0.0453) | 3 / 0 of 3 | 25/25 | 2.4e-06 |
| + ImageNet encoder + TTA (σ = view s.d.) vs U-Net baseline + TTA (σ = view s.d.) | spearman_unc_err | 0.3165 | 0.2832 | +0.0334 (-0.0389, +0.1162) | 2 / 0 of 3 | 17/25 | 0.24 |
| + ImageNet encoder + TTA (σ = view s.d.) vs U-Net baseline + TTA (σ = view s.d.) | ause | 0.2477 | 0.2582 | -0.0105 (-0.0402, +0.0153) | 2 / 0 of 3 | 17/25 | 0.46 |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder + TTA (σ = view s.d.) | spearman_unc_err | 0.4302 | 0.3165 | +0.1137 (+0.0725, +0.1566) | 3 / 0 of 3 | 24/25 | 4.1e-06 |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder + TTA (σ = view s.d.) | ause | 0.3029 | 0.2477 | +0.0552 (-0.0213, +0.1935) | 0 / 0 of 3 | 12/25 | 1 |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline + TTA (σ = view s.d.) | spearman_unc_err | 0.4302 | 0.2832 | +0.1470 (+0.0900, +0.2271) | 3 / 0 of 3 | 25/25 | 2.4e-06 |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline + TTA (σ = view s.d.) | ause | 0.3029 | 0.2582 | +0.0447 (-0.0344, +0.1813) | 0 / 0 of 3 | 12/25 | 1 |
| ChipStain + TTA (full) vs U-Net baseline (scratch, L1, lr 1e-3) | pearson | 0.7681 | 0.7491 | +0.0190 (-0.0079, +0.0490) | 2 / 0 of 3 | 23/25 | — |
| ChipStain + TTA (full) vs U-Net baseline (scratch, L1, lr 1e-3) | ssim | 0.8137 | 0.8262 | -0.0125 (-0.0315, +0.0108) | 1 / 2 of 3 | 2/25 | — |
| ChipStain + TTA (full) vs U-Net baseline (scratch, L1, lr 1e-3) | mae | 0.0323 | 0.0299 | +0.0025 (+0.0005, +0.0048) | 0 / 2 of 3 | 7/25 | — |
| ChipStain + TTA (full) vs U-Net baseline (scratch, L1, lr 1e-3) | psnr | 24.1148 | 24.1467 | -0.0318 (-0.5867, +0.3785) | 1 / 0 of 3 | 15/25 | — |
| ChipStain + TTA (full) vs U-Net baseline (scratch, L1, lr 1e-3) | seg_f1 | 0.7082 | 0.6479 | +0.0603 (-0.0043, +0.1594) | 2 / 0 of 3 | 24/25 | — |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline (scratch, L1, lr 1e-3) | pearson | 0.7571 | 0.7491 | +0.0080 (-0.0212, +0.0393) | 2 / 0 of 3 | 19/25 | — |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline (scratch, L1, lr 1e-3) | ssim | 0.8098 | 0.8262 | -0.0164 (-0.0360, +0.0012) | 0 / 2 of 3 | 4/25 | — |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline (scratch, L1, lr 1e-3) | mae | 0.0324 | 0.0299 | +0.0025 (+0.0004, +0.0052) | 0 / 2 of 3 | 9/25 | — |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline (scratch, L1, lr 1e-3) | psnr | 23.9240 | 24.1467 | -0.2227 (-0.7694, +0.2373) | 0 / 0 of 3 | 12/25 | — |
| ChipStain (β-NLL, β = 0.5) vs U-Net baseline (scratch, L1, lr 1e-3) | seg_f1 | 0.6973 | 0.6479 | +0.0494 (-0.0154, +0.1486) | 2 / 0 of 3 | 23/25 | — |
| + ImageNet encoder (L1, lr 5e-4) vs U-Net baseline (scratch, L1, lr 1e-3) | pearson | 0.7545 | 0.7491 | +0.0054 (-0.0141, +0.0357) | 1 / 1 of 3 | 18/25 | — |
| + ImageNet encoder (L1, lr 5e-4) vs U-Net baseline (scratch, L1, lr 1e-3) | ssim | 0.8228 | 0.8262 | -0.0034 (-0.0117, +0.0024) | 0 / 1 of 3 | 12/25 | — |
| + ImageNet encoder (L1, lr 5e-4) vs U-Net baseline (scratch, L1, lr 1e-3) | mae | 0.0296 | 0.0299 | -0.0003 (-0.0011, +0.0005) | 1 / 0 of 3 | 16/25 | — |
| + ImageNet encoder (L1, lr 5e-4) vs U-Net baseline (scratch, L1, lr 1e-3) | psnr | 24.2512 | 24.1467 | +0.1046 (-0.1124, +0.3725) | 1 / 0 of 3 | 21/25 | — |
| + ImageNet encoder (L1, lr 5e-4) vs U-Net baseline (scratch, L1, lr 1e-3) | seg_f1 | 0.6690 | 0.6479 | +0.0211 (-0.0454, +0.1203) | 1 / 2 of 3 | 21/25 | — |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder (L1, lr 5e-4) | pearson | 0.7571 | 0.7545 | +0.0025 (-0.0195, +0.0194) | 2 / 0 of 3 | 16/25 | — |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder (L1, lr 5e-4) | ssim | 0.8098 | 0.8228 | -0.0130 (-0.0331, +0.0062) | 0 / 2 of 3 | 9/25 | — |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder (L1, lr 5e-4) | mae | 0.0324 | 0.0296 | +0.0028 (+0.0007, +0.0054) | 0 / 2 of 3 | 10/25 | — |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder (L1, lr 5e-4) | psnr | 23.9240 | 24.2512 | -0.3272 (-0.8786, +0.0703) | 0 / 0 of 3 | 8/25 | — |
| ChipStain (β-NLL, β = 0.5) vs + ImageNet encoder (L1, lr 5e-4) | seg_f1 | 0.6973 | 0.6690 | +0.0283 (+0.0001, +0.0538) | 2 / 0 of 3 | 21/25 | — |
