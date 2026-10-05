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
cp report/figures/qualitative_chipstain_nll_s0_tta.png writeup/assets/gallery_1_qualitative.png
cp report/figures/multiseed.png writeup/assets/gallery_2_multiseed.png
cp report/figures/shift_test.png writeup/assets/gallery_3_shift_test.png
cp report/figures/timelapse.png writeup/assets/gallery_4_timelapse.png
cp report/figures/demo_panel_dense_t150.png writeup/assets/gallery_5_demo_panel.png
cp report/figures/proliferation.png writeup/assets/gallery_6_proliferation.png
[ -f report/figures/neural_examples.png ] && cp report/figures/neural_examples.png writeup/assets/gallery_7_neural.png
python scripts/image_register.py
python scripts/build_report.py
python scripts/build_writeup.py
python scripts/build_presentation.py
if [ -f tools_local/build_video.py ] && [ "${1:-}" != "--no-video" ]; then python tools_local/build_video.py; fi
command -v pbcopy >/dev/null && pbcopy < writeup/kaggle_writeup.md
# optional local helper (not part of the repository): push the text into the Kaggle draft
if [ -f tools_local/kaggle_sync_writeup.py ]; then { PYTHONPATH=. python tools_local/kaggle_sync_writeup.py || echo "Kaggle sync skipped - the text is on the clipboard"; }; fi
echo "Rebuilt: report/ChipStain_Technical_Report.pdf, README.md, writeup/kaggle_writeup.md, presentation/ChipStain_presentation.html, presentation/VIDEO_SCRIPT.md"
