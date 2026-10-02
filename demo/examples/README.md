Example images from the held-out test well (R05-C03, field 3) of the HeLa "Kyoto" dataset
(Romain Guiet, EPFL BIOP; Zenodo [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064);
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

* `*_bf_*.tif` — bright-field inputs for the demo; unmodified copies of the original files.
* `*_h2b_real_*.tif` — the matching real mCherry-H2B images: channel 1 extracted from the original 2-channel
  `_fluo.tif`, pixel values unchanged. Provided only for visual comparison — not demo inputs.

Reference nuclei (StarDist on the real H2B, from the dataset): 64 at t010 and 180 at t150.
* `example_bf_dense_t150_blur1px.tif` — the dense bright-field example with a Gaussian blur of σ = 1 px (a crude
  defocus proxy, made by this project) to demonstrate the uncertainty response; derived from the CC BY 4.0 original.
