Empirical coverage of μ ± z·σ on the 125 test images (every 3rd pixel), mean over seeds. 'recalibrated' multiplies σ by one scalar fitted on the validation split.

| model | σ-scale (val) | nominal 50 % | nominal 68.3 % | nominal 80 % | nominal 90 % | nominal 95 % |
|---|---|---|---|---|---|---|
| U-Net + TTA (view s.d.), raw | — | 4.9 % | 7.4 % | 9.6 % | 12.4 % | 14.8 % |
| U-Net + TTA (view s.d.), recalibrated | ×21.27 | 66.7 % | 79.5 % | 86.1 % | 91.4 % | 94.2 % |
| ChipStain, single pass, raw | — | 45.6 % | 63.3 % | 74.4 % | 84.1 % | 89.4 % |
| ChipStain + TTA, raw | — | 52.4 % | 69.6 % | 79.4 % | 87.5 % | 91.9 % |
| ChipStain + TTA, recalibrated | ×0.98 | 51.3 % | 68.6 % | 78.5 % | 86.8 % | 91.3 % |
