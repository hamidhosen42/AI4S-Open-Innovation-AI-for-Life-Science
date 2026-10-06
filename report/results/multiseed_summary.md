## Main comparison (mean ± sd over 3 seeds; each seed = mean over the 125 test images of well R05-C03)

| Model | Pearson r ↑ | SSIM ↑ | MAE ↓ | PSNR ↑ | seg-F1 ↑ | ρ(σ, err) ↑ | AUSE ↓ | MAE drop, top-20 % σ removed ↑ |
|---|---|---|---|---|---|---|---|---|
| Scratch U-Net (L1, lr 1e-3) | 0.749 ± 0.027 | 0.826 ± 0.003 | 0.0299 ± 0.0006 | 24.15 ± 0.24 | 0.648 ± 0.089 | — | — | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) | 0.755 ± 0.005 | 0.823 ± 0.005 | 0.0296 ± 0.0003 | 24.25 ± 0.07 | 0.669 ± 0.021 | — | — | — |
| Scratch U-Net + TTA (σ = view s.d.) | 0.764 ± 0.024 | 0.831 ± 0.003 | 0.0294 ± 0.0006 | 24.36 ± 0.25 | 0.665 ± 0.083 | 0.283 ± 0.071 | 0.258 ± 0.025 | 40.5 ± 3.2 % |
| ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | 0.768 ± 0.006 | 0.827 ± 0.005 | 0.0291 ± 0.0003 | 24.47 ± 0.10 | 0.690 ± 0.019 | 0.317 ± 0.037 | 0.248 ± 0.011 | 42.9 ± 1.8 % |
| ChipStain (β-NLL, β = 0.5) | 0.757 ± 0.015 | 0.810 ± 0.018 | 0.0324 ± 0.0013 | 23.92 ± 0.36 | 0.697 ± 0.012 | 0.430 ± 0.006 | 0.303 ± 0.072 | 45.6 ± 1.6 % |
| ChipStain + TTA (full) | 0.768 ± 0.012 | 0.814 ± 0.022 | 0.0323 ± 0.0007 | 24.11 ± 0.32 | 0.708 ± 0.011 | 0.469 ± 0.027 | 0.177 ± 0.021 | 49.7 ± 4.2 % |
| Real fluorescence, same segmentation pipeline (reference level, not a bound) | — | — | — | — | 0.765 | — | — | — |

## Ablation arms (no TTA)

| Model | Pearson r ↑ | SSIM ↑ | MAE ↓ | seg-F1 ↑ | ρ(σ, err) ↑ | AUSE ↓ |
|---|---|---|---|---|---|---|
| Scratch U-Net (L1, lr 1e-3) | 0.749 ± 0.027 | 0.826 ± 0.003 | 0.0299 ± 0.0006 | 0.648 ± 0.089 | — | — |
| Scratch U-Net (L1, lr 5e-4) | 0.724 ± 0.013 | 0.815 ± 0.002 | 0.0310 ± 0.0000 | 0.612 ± 0.035 | — | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) | 0.755 ± 0.005 | 0.823 ± 0.005 | 0.0296 ± 0.0003 | 0.669 ± 0.021 | — | — |
| ImageNet U-Net (MSE, lr 5e-4) | 0.759 ± 0.042 | 0.799 ± 0.008 | 0.0327 ± 0.0028 | 0.685 ± 0.031 | — | — |
| ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | 0.768 ± 0.006 | 0.827 ± 0.005 | 0.0291 ± 0.0003 | 0.690 ± 0.019 | 0.317 ± 0.037 | 0.248 ± 0.011 |
| ChipStain head, β = 0 (plain NLL) | 0.708 ± 0.009 | 0.780 ± 0.008 | 0.0348 ± 0.0009 | 0.614 ± 0.019 | 0.517 ± 0.032 | 0.173 ± 0.022 |
| ChipStain head, β = 0, + TTA | 0.731 ± 0.017 | 0.798 ± 0.002 | 0.0342 ± 0.0017 | 0.640 ± 0.014 | 0.520 ± 0.013 | 0.160 ± 0.005 |
| ChipStain head, β = 1 | 0.746 ± 0.013 | 0.755 ± 0.029 | 0.0345 ± 0.0017 | 0.671 ± 0.011 | 0.350 ± 0.074 | 0.315 ± 0.028 |
| ChipStain head, β = 1, + TTA | 0.768 ± 0.006 | 0.793 ± 0.013 | 0.0326 ± 0.0011 | 0.695 ± 0.007 | 0.384 ± 0.076 | 0.245 ± 0.036 |
| ChipStain (β-NLL, β = 0.5) | 0.757 ± 0.015 | 0.810 ± 0.018 | 0.0324 ± 0.0013 | 0.697 ± 0.012 | 0.430 ± 0.006 | 0.303 ± 0.072 |

## Does σ beat uncertainty-free proxies at ranking pixel errors?

Proxies are computed from the model's own prediction: intensity = predicted μ, edges = Sobel edge strength of μ.

| Model | ρ: σ | ρ: intensity | ρ: edges | AUSE: σ | AUSE: intensity | AUSE: edges | gain: σ | gain: intensity | gain: edges | gain: oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| Scratch U-Net + TTA (σ = view s.d.) | 0.283 ± 0.071 | 0.182 ± 0.025 | 0.297 ± 0.045 | 0.258 ± 0.025 | 0.310 ± 0.009 | 0.256 ± 0.017 | 40.5 ± 3.2 % | 39.2 ± 1.8 % | 40.0 ± 2.5 % | 55.1 ± 2.5 % |
| ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | 0.317 ± 0.037 | 0.185 ± 0.033 | 0.297 ± 0.032 | 0.248 ± 0.011 | 0.379 ± 0.121 | 0.324 ± 0.124 | 42.9 ± 1.8 % | 39.4 ± 3.7 % | 39.5 ± 3.3 % | 56.6 ± 1.6 % |
| ChipStain head, β = 0 (plain NLL) | 0.517 ± 0.032 | 0.293 ± 0.048 | 0.447 ± 0.027 | 0.173 ± 0.022 | 0.431 ± 0.107 | 0.277 ± 0.120 | 50.2 ± 3.8 % | 45.1 ± 3.1 % | 44.2 ± 1.8 % | 59.5 ± 3.1 % |
| ChipStain head, β = 0, + TTA | 0.520 ± 0.013 | 0.295 ± 0.074 | 0.441 ± 0.038 | 0.160 ± 0.005 | 0.344 ± 0.116 | 0.274 ± 0.119 | 48.6 ± 3.5 % | 44.4 ± 3.7 % | 42.5 ± 2.2 % | 57.6 ± 2.8 % |
| ChipStain head, β = 1 | 0.350 ± 0.074 | 0.484 ± 0.036 | 0.374 ± 0.011 | 0.315 ± 0.028 | 0.303 ± 0.039 | 0.283 ± 0.036 | 41.8 ± 1.4 % | 44.0 ± 0.9 % | 39.4 ± 2.2 % | 54.5 ± 0.6 % |
| ChipStain head, β = 1, + TTA | 0.384 ± 0.076 | 0.410 ± 0.007 | 0.374 ± 0.010 | 0.245 ± 0.036 | 0.341 ± 0.081 | 0.329 ± 0.074 | 45.2 ± 1.3 % | 42.8 ± 1.2 % | 39.6 ± 1.5 % | 55.1 ± 1.0 % |
| ChipStain (β-NLL, β = 0.5) | 0.430 ± 0.006 | 0.306 ± 0.020 | 0.395 ± 0.017 | 0.303 ± 0.072 | 0.296 ± 0.043 | 0.280 ± 0.087 | 45.6 ± 1.6 % | 47.8 ± 4.6 % | 44.3 ± 1.7 % | 59.3 ± 3.4 % |
| ChipStain + TTA (full) | 0.469 ± 0.027 | 0.320 ± 0.041 | 0.414 ± 0.030 | 0.177 ± 0.021 | 0.308 ± 0.104 | 0.271 ± 0.060 | 49.7 ± 4.2 % | 47.7 ± 5.4 % | 43.4 ± 3.0 % | 58.5 ± 3.7 % |

## Image level: σ versus cell density

| Model | ρ(mean σ, MAE) | ρ(n_pred, MAE) | partial ρ(σ, MAE) given n_pred | ρ(mean σ, 1−F1) | ρ(n_pred, 1−F1) | ρ(mean σ, 1−r) |
|---|---|---|---|---|---|---|
| Scratch U-Net + TTA (σ = view s.d.) | 0.903 ± 0.029 | 0.872 ± 0.021 | 0.515 ± 0.108 | 0.406 ± 0.044 | 0.292 ± 0.060 | 0.648 ± 0.022 |
| ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | 0.771 ± 0.275 | 0.896 ± 0.022 | 0.462 ± 0.218 | 0.310 ± 0.081 | 0.276 ± 0.035 | 0.586 ± 0.049 |
| ChipStain head, β = 0 (plain NLL) | 0.945 ± 0.016 | 0.939 ± 0.015 | 0.373 ± 0.045 | 0.317 ± 0.097 | 0.279 ± 0.104 | 0.491 ± 0.162 |
| ChipStain head, β = 0, + TTA | 0.936 ± 0.016 | 0.929 ± 0.013 | 0.358 ± 0.115 | 0.374 ± 0.067 | 0.331 ± 0.078 | 0.575 ± 0.087 |
| ChipStain head, β = 1 | 0.942 ± 0.022 | 0.940 ± 0.015 | 0.374 ± 0.188 | 0.379 ± 0.085 | 0.337 ± 0.043 | 0.481 ± 0.183 |
| ChipStain head, β = 1, + TTA | 0.882 ± 0.040 | 0.933 ± 0.014 | 0.308 ± 0.190 | 0.333 ± 0.081 | 0.321 ± 0.033 | 0.569 ± 0.074 |
| ChipStain (β-NLL, β = 0.5) | 0.948 ± 0.022 | 0.907 ± 0.031 | 0.596 ± 0.233 | 0.392 ± 0.108 | 0.308 ± 0.037 | 0.622 ± 0.072 |
| ChipStain + TTA (full) | 0.908 ± 0.067 | 0.891 ± 0.028 | 0.628 ± 0.290 | 0.426 ± 0.135 | 0.303 ± 0.034 | 0.650 ± 0.089 |

## Paired tests

Inference rests on the 95 % CI of A − B from a hierarchical bootstrap that resamples seeds (per configuration) and fields, and on per-seed tests (field-level Wilcoxon within each seed pair, p < 0.05): 'seeds better / worse' counts the seed pairs in which A is significantly better / worse. The pooled field-level Wilcoxon averages over seeds and only measures consistency within this single well (one plate, one imaging session); it is shown, Holm-corrected over all tests, for the uncertainty metrics only.

| A vs B | metric | A | B | A − B (95 % CI, seeds + fields) | seeds sig. better / worse | fields A better (seed-avg) | p (pooled, Holm) |
|---|---|---|---|---|---|---|---|
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | pearson | 0.7681 | 0.7644 | +0.0037 (-0.0211, +0.0304) | 1 / 0 of 3 | 18/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | ssim | 0.8137 | 0.8310 | -0.0173 (-0.0361, +0.0061) | 0 / 2 of 3 | 0/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | mae | 0.0323 | 0.0294 | +0.0029 (+0.0009, +0.0054) | 0 / 2 of 3 | 6/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | psnr | 24.1148 | 24.3614 | -0.2466 (-0.7972, +0.1756) | 0 / 0 of 3 | 11/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | seg_f1 | 0.7082 | 0.6651 | +0.0431 (-0.0171, +0.1351) | 1 / 0 of 3 | 24/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | spearman_unc_err | 0.4689 | 0.2832 | +0.1857 (+0.1201, +0.2641) | 3 / 0 of 3 | 25/25 | 4.4e-06 |
| ChipStain + TTA (full) vs Scratch U-Net + TTA (σ = view s.d.) | ause | 0.1775 | 0.2582 | -0.0807 (-0.1136, -0.0493) | 3 / 0 of 3 | 25/25 | 4.4e-06 |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | pearson | 0.7681 | 0.7678 | +0.0003 (-0.0194, +0.0133) | 2 / 0 of 3 | 15/25 | — |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | ssim | 0.8137 | 0.8271 | -0.0134 (-0.0323, +0.0103) | 0 / 2 of 3 | 3/25 | — |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | mae | 0.0323 | 0.0291 | +0.0032 (+0.0012, +0.0056) | 0 / 2 of 3 | 3/25 | — |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | psnr | 24.1148 | 24.4676 | -0.3528 (-0.9085, -0.0132) | 0 / 0 of 3 | 7/25 | — |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | seg_f1 | 0.7082 | 0.6898 | +0.0184 (-0.0092, +0.0419) | 2 / 0 of 3 | 20/25 | — |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | spearman_unc_err | 0.4689 | 0.3165 | +0.1523 (+0.1057, +0.2010) | 3 / 0 of 3 | 25/25 | 4.4e-06 |
| ChipStain + TTA (full) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | ause | 0.1775 | 0.2477 | -0.0702 (-0.0947, -0.0453) | 3 / 0 of 3 | 25/25 | 4.4e-06 |
| ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control vs Scratch U-Net + TTA (σ = view s.d.) | spearman_unc_err | 0.3165 | 0.2832 | +0.0334 (-0.0389, +0.1162) | 2 / 0 of 3 | 17/25 | 0.43 |
| ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control vs Scratch U-Net + TTA (σ = view s.d.) | ause | 0.2477 | 0.2582 | -0.0105 (-0.0402, +0.0153) | 2 / 0 of 3 | 17/25 | 0.84 |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | spearman_unc_err | 0.4302 | 0.3165 | +0.1137 (+0.0725, +0.1566) | 3 / 0 of 3 | 24/25 | 6.8e-06 |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net + TTA (σ = view s.d.) — matched control | ause | 0.3029 | 0.2477 | +0.0552 (-0.0213, +0.1935) | 0 / 0 of 3 | 12/25 | 1 |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net + TTA (σ = view s.d.) | spearman_unc_err | 0.4302 | 0.2832 | +0.1470 (+0.0900, +0.2271) | 3 / 0 of 3 | 25/25 | 4.4e-06 |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net + TTA (σ = view s.d.) | ause | 0.3029 | 0.2582 | +0.0447 (-0.0344, +0.1813) | 0 / 0 of 3 | 12/25 | 1 |
| ChipStain + TTA (full) vs Scratch U-Net (L1, lr 1e-3) | pearson | 0.7681 | 0.7491 | +0.0190 (-0.0079, +0.0490) | 2 / 0 of 3 | 23/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net (L1, lr 1e-3) | ssim | 0.8137 | 0.8262 | -0.0125 (-0.0315, +0.0108) | 1 / 2 of 3 | 2/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net (L1, lr 1e-3) | mae | 0.0323 | 0.0299 | +0.0025 (+0.0005, +0.0048) | 0 / 2 of 3 | 7/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net (L1, lr 1e-3) | psnr | 24.1148 | 24.1467 | -0.0318 (-0.5867, +0.3785) | 1 / 0 of 3 | 15/25 | — |
| ChipStain + TTA (full) vs Scratch U-Net (L1, lr 1e-3) | seg_f1 | 0.7082 | 0.6479 | +0.0603 (-0.0043, +0.1594) | 2 / 0 of 3 | 24/25 | — |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net (L1, lr 1e-3) | pearson | 0.7571 | 0.7491 | +0.0080 (-0.0212, +0.0393) | 2 / 0 of 3 | 19/25 | — |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net (L1, lr 1e-3) | ssim | 0.8098 | 0.8262 | -0.0164 (-0.0360, +0.0012) | 0 / 2 of 3 | 4/25 | — |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net (L1, lr 1e-3) | mae | 0.0324 | 0.0299 | +0.0025 (+0.0004, +0.0052) | 0 / 2 of 3 | 9/25 | — |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net (L1, lr 1e-3) | psnr | 23.9240 | 24.1467 | -0.2227 (-0.7694, +0.2373) | 0 / 0 of 3 | 12/25 | — |
| ChipStain (β-NLL, β = 0.5) vs Scratch U-Net (L1, lr 1e-3) | seg_f1 | 0.6973 | 0.6479 | +0.0494 (-0.0154, +0.1486) | 2 / 0 of 3 | 23/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | pearson | 0.7545 | 0.7491 | +0.0054 (-0.0141, +0.0357) | 1 / 1 of 3 | 18/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | ssim | 0.8228 | 0.8262 | -0.0034 (-0.0117, +0.0024) | 0 / 1 of 3 | 12/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | mae | 0.0296 | 0.0299 | -0.0003 (-0.0011, +0.0005) | 1 / 0 of 3 | 16/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | psnr | 24.2512 | 24.1467 | +0.1046 (-0.1124, +0.3725) | 1 / 0 of 3 | 21/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | seg_f1 | 0.6690 | 0.6479 | +0.0211 (-0.0454, +0.1203) | 1 / 2 of 3 | 21/25 | — |
| Scratch U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | pearson | 0.7240 | 0.7491 | -0.0251 (-0.0499, +0.0043) | 0 / 3 of 3 | 0/25 | — |
| Scratch U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | ssim | 0.8148 | 0.8262 | -0.0114 (-0.0166, -0.0074) | 0 / 3 of 3 | 2/25 | — |
| Scratch U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | mae | 0.0310 | 0.0299 | +0.0011 (+0.0003, +0.0018) | 0 / 2 of 3 | 4/25 | — |
| Scratch U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | psnr | 23.8157 | 24.1467 | -0.3310 (-0.5891, -0.0516) | 0 / 3 of 3 | 1/25 | — |
| Scratch U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 1e-3) | seg_f1 | 0.6120 | 0.6479 | -0.0359 (-0.1121, +0.0603) | 1 / 2 of 3 | 0/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 5e-4) | pearson | 0.7545 | 0.7240 | +0.0306 (+0.0179, +0.0467) | 3 / 0 of 3 | 25/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 5e-4) | ssim | 0.8228 | 0.8148 | +0.0080 (-0.0009, +0.0141) | 2 / 0 of 3 | 17/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 5e-4) | mae | 0.0296 | 0.0310 | -0.0014 (-0.0020, -0.0008) | 3 / 0 of 3 | 21/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 5e-4) | psnr | 24.2512 | 23.8157 | +0.4355 (+0.2895, +0.6248) | 3 / 0 of 3 | 25/25 | — |
| ImageNet-L1 U-Net (L1, lr 5e-4) vs Scratch U-Net (L1, lr 5e-4) | seg_f1 | 0.6690 | 0.6120 | +0.0571 (+0.0190, +0.1001) | 3 / 0 of 3 | 25/25 | — |
| ImageNet U-Net (MSE, lr 5e-4) vs ImageNet-L1 U-Net (L1, lr 5e-4) | pearson | 0.7590 | 0.7545 | +0.0044 (-0.0453, +0.0331) | 2 / 0 of 3 | 19/25 | — |
| ImageNet U-Net (MSE, lr 5e-4) vs ImageNet-L1 U-Net (L1, lr 5e-4) | ssim | 0.7988 | 0.8228 | -0.0240 (-0.0362, -0.0145) | 0 / 3 of 3 | 0/25 | — |
| ImageNet U-Net (MSE, lr 5e-4) vs ImageNet-L1 U-Net (L1, lr 5e-4) | mae | 0.0327 | 0.0296 | +0.0031 (+0.0011, +0.0068) | 0 / 3 of 3 | 0/25 | — |
| ImageNet U-Net (MSE, lr 5e-4) vs ImageNet-L1 U-Net (L1, lr 5e-4) | psnr | 23.9293 | 24.2512 | -0.3219 (-1.4103, +0.3344) | 2 / 1 of 3 | 10/25 | — |
| ImageNet U-Net (MSE, lr 5e-4) vs ImageNet-L1 U-Net (L1, lr 5e-4) | seg_f1 | 0.6854 | 0.6690 | +0.0164 (-0.0270, +0.0499) | 2 / 0 of 3 | 17/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet U-Net (MSE, lr 5e-4) | pearson | 0.7571 | 0.7590 | -0.0019 (-0.0409, +0.0470) | 1 / 2 of 3 | 11/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet U-Net (MSE, lr 5e-4) | ssim | 0.8098 | 0.7988 | +0.0110 (-0.0099, +0.0318) | 2 / 1 of 3 | 18/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet U-Net (MSE, lr 5e-4) | mae | 0.0324 | 0.0327 | -0.0003 (-0.0042, +0.0032) | 1 / 1 of 3 | 15/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet U-Net (MSE, lr 5e-4) | psnr | 23.9240 | 23.9293 | -0.0053 (-0.9328, +1.0752) | 1 / 2 of 3 | 11/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet U-Net (MSE, lr 5e-4) | seg_f1 | 0.6973 | 0.6854 | +0.0119 (-0.0193, +0.0518) | 1 / 0 of 3 | 19/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net (L1, lr 5e-4) | pearson | 0.7571 | 0.7545 | +0.0025 (-0.0195, +0.0194) | 2 / 0 of 3 | 16/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net (L1, lr 5e-4) | ssim | 0.8098 | 0.8228 | -0.0130 (-0.0331, +0.0062) | 0 / 2 of 3 | 9/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net (L1, lr 5e-4) | mae | 0.0324 | 0.0296 | +0.0028 (+0.0007, +0.0054) | 0 / 2 of 3 | 10/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net (L1, lr 5e-4) | psnr | 23.9240 | 24.2512 | -0.3272 (-0.8786, +0.0703) | 0 / 0 of 3 | 8/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ImageNet-L1 U-Net (L1, lr 5e-4) | seg_f1 | 0.6973 | 0.6690 | +0.0283 (+0.0001, +0.0538) | 2 / 0 of 3 | 21/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | pearson | 0.7571 | 0.7080 | +0.0491 (+0.0258, +0.0682) | 3 / 0 of 3 | 23/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | ssim | 0.8098 | 0.7799 | +0.0298 (+0.0090, +0.0480) | 3 / 0 of 3 | 25/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | mae | 0.0324 | 0.0348 | -0.0024 (-0.0043, -0.0001) | 3 / 0 of 3 | 22/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | psnr | 23.9240 | 23.4647 | +0.4593 (-0.1086, +0.8770) | 2 / 0 of 3 | 22/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | seg_f1 | 0.6973 | 0.6136 | +0.0837 (+0.0577, +0.1105) | 3 / 0 of 3 | 24/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | spearman_unc_err | 0.4302 | 0.5170 | -0.0868 (-0.1194, -0.0507) | 0 / 3 of 3 | 0/25 | 4.4e-06 |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 0 (plain NLL) | ause | 0.3029 | 0.1727 | +0.1302 (+0.0524, +0.2622) | 0 / 3 of 3 | 0/25 | 4.4e-06 |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | pearson | 0.7571 | 0.7465 | +0.0106 (-0.0141, +0.0334) | 2 / 0 of 3 | 19/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | ssim | 0.8098 | 0.7554 | +0.0544 (+0.0241, +0.0887) | 3 / 0 of 3 | 25/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | mae | 0.0324 | 0.0345 | -0.0021 (-0.0046, +0.0007) | 3 / 0 of 3 | 20/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | psnr | 23.9240 | 23.8608 | +0.0633 (-0.5231, +0.5395) | 2 / 0 of 3 | 17/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | seg_f1 | 0.6973 | 0.6712 | +0.0261 (+0.0034, +0.0457) | 3 / 0 of 3 | 22/25 | — |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | spearman_unc_err | 0.4302 | 0.3496 | +0.0806 (+0.0066, +0.1516) | 2 / 0 of 3 | 23/25 | 2.2e-05 |
| ChipStain (β-NLL, β = 0.5) vs ChipStain head, β = 1 | ause | 0.3029 | 0.3148 | -0.0119 (-0.1086, +0.1254) | 2 / 0 of 3 | 15/25 | 1 |
