# Data

ChipStain uses only public, openly licensed data. No clinical, personal or restricted data is involved.

## Training / evaluation set

| | |
|---|---|
| Name | *Automatic labelling of HeLa "Kyoto" cells using Deep Learning tools* — `training_dataset.zip` |
| DOI | [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064) |
| Derived from | *HeLa "Kyoto" cells under the scope*, [10.5281/zenodo.6139958](https://doi.org/10.5281/zenodo.6139958) |
| License | CC BY 4.0 |
| Cells | HeLa "Kyoto" expressing EGFP-α-tubulin and mCherry-H2B |
| Microscope | PerkinElmer Operetta, 20× / NA 0.8, Andor Zyla 5.5 |
| Images | 540 × 540 px, 16-bit TIFF, single z-plane |
| Channels per field | bright-field (`_bf`), digital phase contrast (`_dpc`, `_sqrdpc`), 2-channel fluorescence (`_fluo`: [0] tubulin, [1] H2B), StarDist nuclei labels (`_nuclei`), Cellpose cell labels (`_cyto`) |
| Fields | train: wells R05-C05 and R05-C07, 25 fields × 5 time-points = 250 · test: well R05-C03, 125 |

We use `_bf` as input and `_fluo[1]` (mCherry-H2B) as the target. The `_nuclei` StarDist labels are used **only** as reference objects for the downstream segmentation metric — never for training.

## Splits (no leakage)

* **Test** — the separate well R05-C03 (as shipped in the archive; 125 images).
* **Validation** — fields 20–24 of each training well, all time-points (50 images).
* **Train** — remaining fields of the training wells (200 images).

All time-points of a field stay on the same side of a split, so near-duplicate frames cannot leak.

## Pre-processing

* Input: per-image robust normalisation using the 1st/99th percentiles, mapped to roughly [−2, 2].
* Target: fixed global scaling `(x − 600) / (20000 − 600)`, clipped to [0, 1] (percentiles of the training H2B channel). All reported image metrics are computed in this normalised space.
* Training crops 256 × 256 with random 90° rotations and flips; evaluation on the full 540 × 540 image (reflect-padded to 544).

## Download

```bash
python scripts/download_data.py          # ~757 MB -> data/raw/hela_kyoto/{train,test}
```

## Compliance statement

The dataset is a cell-line imaging dataset released under CC BY 4.0; it contains no human-subject, clinical or personal data. Attribution to the original authors is given in the technical report and the README.
