# AI-assistance and third-party disclosure

Required by the challenge rules (§4 External Models, Software and AI Tools; §6 disclosure of
material third-party work). Everything below is a material part of the project.

## AI assistant used to build the project

| Tool | Provider | How it was used |
|---|---|---|
| Claude Opus 5 and Claude Opus 5.5, through the Claude Code CLI | Anthropic | **Primary implementation assistant.** It analysed the competition rules, proposed the project direction and dataset, wrote most of the code (the `chipstain/` package, training / evaluation / inference / analysis scripts, the Gradio demo and the Kaggle notebook), launched the training and evaluation runs on the team's laptop under the team's direction, produced the figures, and drafted the technical report, the Kaggle Writeup and the video script. It also ran an automated multi-agent audit of the submission against the competition rules. |

The team (Hack2Publish) chose the problem, the category and the method from the options proposed, directed the
work, reviewed the code and outputs, and takes full responsibility for every statement and number in the
submission. All reported numbers are produced by the scripts in this repository from the saved runs; the result
files are in `report/results/`. Commits made with the assistant carry a `Co-Authored-By` trailer in the git history.

No large language model, foundation model, external API or paid service is used **by the method**: training and
inference run offline with the open-source components listed below.

## Pre-trained weights inside the method

| Component | Source | Licence | Role |
|---|---|---|---|
| ResNet-34 ImageNet-1k encoder weights | torchvision original, re-hosted on the Hugging Face Hub as `smp-hub/resnet34.imagenet`; downloaded automatically by segmentation_models_pytorch 0.5 when training the pre-trained configurations (internet needed for training only; inference needs no download) | BSD-3-Clause (torchvision); ImageNet dataset terms apply to the original training images | encoder initialisation |

## Datasets

| Dataset | Licence | Use |
|---|---|---|
| HeLa "Kyoto" automatic-labelling set, R. Guiet (EPFL BIOP), Zenodo [10.5281/zenodo.6140064](https://doi.org/10.5281/zenodo.6140064) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | training, validation, test |
| HeLa "Kyoto" time-lapse, R. Guiet (EPFL BIOP), Zenodo [10.5281/zenodo.6139958](https://doi.org/10.5281/zenodo.6139958) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 60 h read-out of one held-out field (evaluation only) |
| In-silico-labeling data, Condition A (human iPSC-derived motor neurons), Christiansen et al., Cell 2018, `gs://in-silico-labeling` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | neural transfer test (zero-shot and fine-tuning) |

## Third-party software

PyTorch (BSD-3), segmentation-models-pytorch (MIT), timm (Apache-2.0), scikit-image (BSD-3), SciPy (BSD-3),
NumPy (BSD-3), pandas (BSD-3), tifffile (BSD-3), Matplotlib (PSF-based), PyYAML (MIT), tqdm (MPL-2.0 / MIT),
Pillow (HPND), Gradio (Apache-2.0).

## Licence of what we release

Code and the released checkpoint `chipstain.pt`: MIT. The checkpoint was trained on CC BY 4.0 data, so reuse
must attribute Guiet (2022). Figures that show dataset images are CC BY 4.0 derivatives (see `report/IMAGES.md`).
This is new work created for the competition; it is not based on prior work by the team, and it builds on the
published methods cited in the technical report.
