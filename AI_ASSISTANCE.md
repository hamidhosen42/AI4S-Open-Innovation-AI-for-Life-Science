# AI-assistance disclosure

In line with the challenge rules (§4 External Models, Software, and AI Tools), we disclose all AI tools and pre-trained models that are a material part of this project.

## Pre-trained models used inside the method

| Component | Source | License | Role |
|---|---|---|---|
| ResNet-34 ImageNet encoder | `segmentation_models_pytorch` (timm / torchvision weights) | MIT (code); ImageNet weights: research use | Encoder initialisation for the U-Net |

No large language model, foundation model or external API is used at inference time. The model runs offline on CPU.

## AI tools used during development

| Tool | Use |
|---|---|
| Claude (Anthropic, Claude Code) | Code scaffolding, drafting of documentation and report text, competition-rule analysis. All generated code was reviewed, executed and verified by the team; all reported numbers come from runs executed by the team. |

## Third-party software

PyTorch (BSD-3), segmentation-models-pytorch (MIT), scikit-image (BSD-3), SciPy (BSD-3), NumPy (BSD-3), tifffile (BSD-3), Gradio (Apache-2.0), Matplotlib (PSF-based), pandas (BSD-3), PyYAML (MIT).
