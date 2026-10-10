#!/usr/bin/env bash
# Rebuild every deliverable from the current result files, in dependency order:
# figures -> demo numbers -> image register -> report (HTML+PDF, key numbers) -> README + Writeup
# -> presentation (single HTML deck) + video script. Run after ANY change to results or texts.
#   bash scripts/build_all.sh            # everything
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_figures.py
python scripts/make_figures_extra.py
python scripts/demo_numbers.py
python scripts/make_gallery.py                 # 16:9 copies for the Kaggle media gallery
python scripts/image_register.py
python scripts/build_report.py
python scripts/build_writeup.py
python scripts/build_presentation.py
command -v pbcopy >/dev/null && pbcopy < writeup/kaggle_writeup.md
echo "Rebuilt: report/ChipStain_Technical_Report.pdf, README.md, writeup/kaggle_writeup.md, presentation/ChipStain_presentation.html, presentation/VIDEO_SCRIPT.md"
