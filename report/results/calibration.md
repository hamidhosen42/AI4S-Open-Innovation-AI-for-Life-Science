Empirical coverage of μ ± z·σ on the 125 test images (every 3rd pixel), mean over seeds. 'recalibrated' multiplies σ by one scalar fitted on the validation split.

| model | σ-scale (val) | nominal 50 % | nominal 68.3 % | nominal 80 % | nominal 90 % | nominal 95 % |
|---|---|---|---|---|---|---|
| U-Net + TTA (view s.d.), raw | — | 4.9 % | 7.4 % | 9.6 % | 12.4 % | 14.8 % |
| U-Net + TTA (view s.d.), recalibrated | ×21.27 | 66.7 % | 79.5 % | 86.1 % | 91.4 % | 94.2 % |
| U-Net + TTA (view s.d.), raw, nuclei pixels only | — | 11.7 % | 17.1 % | 21.5 % | 26.8 % | 31.2 % |
| ChipStain, single pass, raw | — | 45.6 % | 63.3 % | 74.4 % | 84.1 % | 89.4 % |
| ChipStain, single pass, raw, nuclei pixels only | — | 52.6 % | 69.9 % | 79.7 % | 87.0 % | 90.6 % |
| ChipStain + TTA, raw | — | 52.4 % | 69.6 % | 79.4 % | 87.5 % | 91.9 % |
| ChipStain + TTA, recalibrated | ×0.98 | 51.3 % | 68.6 % | 78.5 % | 86.8 % | 91.3 % |
| ChipStain + TTA, raw, nuclei pixels only | — | 56.1 % | 73.9 % | 83.6 % | 90.5 % | 93.7 % |
| + ImageNet encoder + TTA (view s.d.), raw | — | 6.0 % | 8.8 % | 11.2 % | 14.2 % | 16.6 % |
| + ImageNet encoder + TTA (view s.d.), recalibrated | ×18.45 | 62.7 % | 75.8 % | 83.0 % | 89.1 % | 92.5 % |
| + ImageNet encoder + TTA (view s.d.), raw, nuclei pixels only | — | 11.7 % | 17.2 % | 21.7 % | 27.0 % | 31.4 % |
