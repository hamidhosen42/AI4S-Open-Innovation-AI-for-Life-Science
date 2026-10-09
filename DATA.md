# Data

ChipStain uses only public, openly licensed data. No clinical, personal or restricted data is involved.

## Training / evaluation set

| | |
|---|---|
| Name | *Automatic labelling of HeLa "Kyoto" cells using Deep Learning tools* — `training_dataset.zip` |
| DOI | [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064) |
| Derived from | *HeLa "Kyoto" cells under the scope*, [10.5281/zenodo.6139958](https://doi.org/10.5281/zenodo.6139958) |
| Author | Romain Guiet, EPFL BioImaging & Optics Platform (PTBIOP) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Culture format | 2-D monolayer in a CellCarrier Ultra 96-well imaging plate (PerkinElmer) — **not** a microfluidic / organ-on-a-chip device |
| Cells | HeLa "Kyoto" expressing EGFP-α-tubulin and mCherry-H2B |
| Microscope | PerkinElmer Operetta, 20× / NA 0.8, Andor Zyla 5.5 |
| Acquisition details | see the source record [10.5281/zenodo.6139958](https://doi.org/10.5281/zenodo.6139958) (pixel size, exposure, time-lapse interval) |
| Images | 540 × 540 px, 16-bit TIFF, single z-plane |
| Channels per field | bright-field (`_bf`), digital phase contrast (`_dpc`, `_sqrdpc`), 2-channel fluorescence (`_fluo`: [0] tubulin, [1] H2B), StarDist nuclei labels (`_nuclei`), Cellpose cell labels (`_cyto`) |
| Fields | train: wells R05-C05 and R05-C07, 25 fields × 5 time-points = 250 · test: well R05-C03, 125 |

We use `_bf` as input and `_fluo[1]` (mCherry-H2B) as the target. The `_nuclei` StarDist labels are used **only** as reference objects for the downstream segmentation metric — never for training or model selection.

**Provider's usage note.** The dataset description says the fluorescence channels are included to ease review of the automatically generated labels and "should not be reused with our labels during training", and that the automatic labelling "will not produce 100% accurate labels". We comply: we train only bright-field → fluorescence, never fluorescence → labels, and we never train on the labels. Because the reference labels are automatic StarDist output and not manual ground truth, our segmentation F1 and count errors measure agreement with StarDist run on the real H2B stain.

## Splits (no leakage)

* **Test** — the held-out well R05-C03 (as shipped in the archive; 125 images). It comes from the **same plate and imaging session** as the training wells (all files prefixed `20210904_TL2`), so results measure well-to-well generalisation, not generalisation across experiments, days or microscopes.
* **Validation** — fields 20–24 of each training well, all time-points (50 images).
* **Train** — remaining fields of the training wells (200 images).

All time-points of a field stay on the same side of a split, so near-duplicate frames cannot leak.

## Pre-processing

* Input: per-image robust normalisation using the 1st/99th percentiles, mapped to roughly [−2, 2].
* Target: fixed global scaling `(x − 600) / (20000 − 600)`, clipped to [0, 1] (percentiles of the training H2B channel). All reported image metrics are computed in this normalised space.
* Training crops 256 × 256 with random 90° rotations and flips; evaluation on the full 540 × 540 image (reflect-padded to 544).
* Figures show percentile-normalised, colour-mapped renderings of the images (see `report/IMAGES.md`).

## Download

```bash
python scripts/download_data.py          # ~757 MB -> data/raw/hela_kyoto/{train,test}
```

## Additional evaluation data

* **60 h time-lapse of one held-out field** — `20210904_TL2 - R05-C03-F0.tif` from [10.5281/zenodo.6139958](https://doi.org/10.5281/zenodo.6139958) (CC BY 4.0): 240 frames every 15 min, same well as the test set (evaluation only).
* **Neural transfer data** — Christiansen et al., *In silico labeling*, Cell 2018, Condition A (human iPSC-derived motor neurons; bright-field z-stack + widefield Hoechst (channel named DAPI_WIDEFIELD)), `gs://in-silico-labeling/paper_data/{train,test}_single_channel_images/Rubin/`, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Download: `python scripts/download_isl_neurons.py`.

## Compliance and ethics statement

All data are public and released under CC BY 4.0; attribution is given in the report, the README and `report/IMAGES.md`. HeLa is an established human-derived cell line (taken from Henrietta Lacks in 1951 without her consent); the images contain no genetic, clinical or personal information. The neuron data are iPSC-derived cultures published openly by the original authors. No clinical, personal or restricted data are used.
