Reference nuclei: 15866. AUROC of each per-nucleus score for flagging detections that match no reference nucleus (IoU < 0.5); mean ± sd over seeds.

Pooled AUROC ranks nuclei across all images; within-image AUROC is the mean over images of the AUROC computed inside each image (removes differences in σ level between sparse and dense frames). _rel = σ/μ; _imgnorm = σ divided by the median σ of the image's nuclei.

| model | score | pooled AUROC | within-image AUROC | seeds |
|---|---|---|---|---|
| ChipStain + TTA | dim | 0.655 ± 0.031 | 0.650 | 3 |
| ChipStain + TTA | sigma_learned | 0.627 ± 0.024 | 0.710 | 3 |
| ChipStain + TTA | sigma_learned_imgnorm | 0.665 ± 0.032 | 0.710 | 3 |
| ChipStain + TTA | sigma_learned_rel | 0.769 ± 0.043 | 0.778 | 3 |
| ChipStain + TTA | small | 0.546 ± 0.039 | 0.524 | 3 |
| ChipStain ensemble (3 seeds + TTA) | dim | 0.653 | 0.640 | 1 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble | 0.620 | 0.719 | 1 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_imgnorm | 0.666 | 0.719 | 1 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_rel | 0.776 | 0.783 | 1 |
| ChipStain ensemble (3 seeds + TTA) | small | 0.528 | 0.509 | 1 |
| U-Net + TTA | dim | 0.743 ± 0.058 | 0.721 | 3 |
| U-Net + TTA | sigma_tta_unet | 0.721 ± 0.011 | 0.732 | 3 |
| U-Net + TTA | sigma_tta_unet_imgnorm | 0.721 ± 0.023 | 0.732 | 3 |
| U-Net + TTA | sigma_tta_unet_rel | 0.773 ± 0.004 | 0.759 | 3 |
| U-Net + TTA | small | 0.682 ± 0.066 | 0.672 | 3 |

Quality gate (discard the highest-score detections; mean over seeds):

| model | score | discarded | precision | recall | F1 |
|---|---|---|---|---|---|
| ChipStain + TTA | dim | 0% | 0.703 | 0.640 | 0.670 |
| ChipStain + TTA | dim | 5% | 0.732 | 0.633 | 0.679 |
| ChipStain + TTA | dim | 10% | 0.748 | 0.613 | 0.673 |
| ChipStain + TTA | dim | 20% | 0.765 | 0.558 | 0.645 |
| ChipStain + TTA | dim | 30% | 0.777 | 0.495 | 0.604 |
| ChipStain + TTA | sigma_learned | 0% | 0.703 | 0.640 | 0.670 |
| ChipStain + TTA | sigma_learned | 5% | 0.721 | 0.624 | 0.668 |
| ChipStain + TTA | sigma_learned | 10% | 0.734 | 0.602 | 0.661 |
| ChipStain + TTA | sigma_learned | 20% | 0.753 | 0.549 | 0.634 |
| ChipStain + TTA | sigma_learned | 30% | 0.761 | 0.485 | 0.592 |
| ChipStain + TTA | sigma_learned_rel | 0% | 0.703 | 0.640 | 0.670 |
| ChipStain + TTA | sigma_learned_rel | 5% | 0.733 | 0.634 | 0.679 |
| ChipStain + TTA | sigma_learned_rel | 10% | 0.755 | 0.619 | 0.680 |
| ChipStain + TTA | sigma_learned_rel | 20% | 0.792 | 0.577 | 0.667 |
| ChipStain + TTA | sigma_learned_rel | 30% | 0.822 | 0.524 | 0.640 |
| ChipStain ensemble (3 seeds + TTA) | dim | 0% | 0.711 | 0.640 | 0.674 |
| ChipStain ensemble (3 seeds + TTA) | dim | 5% | 0.738 | 0.631 | 0.681 |
| ChipStain ensemble (3 seeds + TTA) | dim | 10% | 0.754 | 0.611 | 0.675 |
| ChipStain ensemble (3 seeds + TTA) | dim | 20% | 0.772 | 0.556 | 0.646 |
| ChipStain ensemble (3 seeds + TTA) | dim | 30% | 0.782 | 0.493 | 0.605 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble | 0% | 0.711 | 0.640 | 0.674 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble | 5% | 0.726 | 0.621 | 0.669 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble | 10% | 0.740 | 0.600 | 0.663 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble | 20% | 0.761 | 0.548 | 0.637 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble | 30% | 0.767 | 0.483 | 0.593 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_rel | 0% | 0.711 | 0.640 | 0.674 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_rel | 5% | 0.737 | 0.631 | 0.680 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_rel | 10% | 0.761 | 0.616 | 0.681 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_rel | 20% | 0.796 | 0.573 | 0.667 |
| ChipStain ensemble (3 seeds + TTA) | sigma_ensemble_rel | 30% | 0.828 | 0.522 | 0.640 |
| U-Net + TTA | dim | 0% | 0.671 | 0.585 | 0.625 |
| U-Net + TTA | dim | 5% | 0.703 | 0.582 | 0.637 |
| U-Net + TTA | dim | 10% | 0.728 | 0.571 | 0.639 |
| U-Net + TTA | dim | 20% | 0.762 | 0.530 | 0.625 |
| U-Net + TTA | dim | 30% | 0.786 | 0.478 | 0.594 |
| U-Net + TTA | sigma_tta_unet | 0% | 0.671 | 0.585 | 0.625 |
| U-Net + TTA | sigma_tta_unet | 5% | 0.697 | 0.577 | 0.631 |
| U-Net + TTA | sigma_tta_unet | 10% | 0.718 | 0.563 | 0.631 |
| U-Net + TTA | sigma_tta_unet | 20% | 0.749 | 0.522 | 0.615 |
| U-Net + TTA | sigma_tta_unet | 30% | 0.772 | 0.471 | 0.585 |
| U-Net + TTA | sigma_tta_unet_rel | 0% | 0.671 | 0.585 | 0.625 |
| U-Net + TTA | sigma_tta_unet_rel | 5% | 0.701 | 0.580 | 0.635 |
| U-Net + TTA | sigma_tta_unet_rel | 10% | 0.726 | 0.570 | 0.638 |
| U-Net + TTA | sigma_tta_unet_rel | 20% | 0.766 | 0.534 | 0.629 |
| U-Net + TTA | sigma_tta_unet_rel | 30% | 0.797 | 0.485 | 0.603 |
