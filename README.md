# AI4S Open Innovation: AI for Life Science

**AI + Organ-on-a-Chip: Open Innovation Challenge for In Vitro Life Systems**
Kaggle entry for the 5th Pazhou Algorithm Competition.

- Competition: https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien
- Official brief: https://www.aicompetition-pz.com/topic_detail/26
- Submission deadline: **10 October 2026, 15:59 UTC**

## About the challenge

Organ-on-a-chip (OoC) systems combine microfluidics, cell biology and sensing to
rebuild human organ structure and function in vitro. They generate complex
multimodal data — microscopy images and video, electrophysiology, omics — and the
challenge asks how AI can turn OoC from an observation tool into a predictive
platform.

The challenge is open-ended: no fixed task, dataset or method. Suggested directions
include image/video/3D segmentation and phenotyping, drug-effect and toxicity
prediction, AI agents for experiment design, digital twins / IVIVT / PBPK-PD models,
data standardisation and knowledge bases, and workflow automation.

### Judging criteria

| Dimension | Weight |
| --- | --- |
| Technical innovation | 30% |
| Work completion & effect | 25% |
| Practical value | 20% |
| Solution completeness | 15% |
| Interpretability & credibility | 10% |

### Required deliverables (single Kaggle Writeup)

1. Demo video (≤ 5 min, public, shows real functionality)
2. Public code repository (this repo)
3. Technical report (~15–20 pages)
4. Optional public interactive demo

## Data

The challenge uses an open data format — no official dataset is provided.
See [data/data_guidelines.md](data/data_guidelines.md) for the organiser's list of
example resources (BBBC, IDR, RxRx1, JUMP-Cell Painting, CytoData, CellNet) and the
compliance requirements. All data sources, licences and processing steps used in
this project will be documented in the technical report.

## Repository layout

```
.
├── data/                 # data guidelines and (gitignored) raw downloads
│   └── data_guidelines.md
├── README.md
└── .gitignore
```

## Setup

```bash
git clone https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science.git
cd AI4S-Open-Innovation-AI-for-Life-Science
```

Environment, entry point, and reproduction steps will be added as the project
develops.

## Team

- Md. Hamid Hosen ([@hamidhosen42](https://github.com/hamidhosen42))

## Compliance

Secrets (API tokens, `kaggle.json`) are excluded via `.gitignore` and must never be
committed. Use of open-source models, third-party libraries, and AI assistants will
be disclosed and attributed in the technical report as required by the rules.
