"""Which fluorescence channel is the nuclear one? Correlation of each channel of *_fluo.tif
with the binarised StarDist nuclei labels over the training split.

    python scripts/check_channels.py   # -> report/results/channel_check.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import tifffile

from chipstain.data import make_splits


def main():
    train, _, _ = make_splits("data/raw/hela_kyoto")
    r = {0: [], 1: []}
    for s in train:
        fl = tifffile.imread(s.base + "_fluo.tif").astype(np.float32)
        m = (tifffile.imread(s.base + "_nuclei.tif") > 0).astype(np.float32)
        for c in (0, 1):
            r[c].append(np.corrcoef(fl[c].ravel(), m.ravel())[0, 1])
    lines = [f"Pearson correlation with binarised nuclei labels over {len(train)} training images (mean ± sd):",
             f"* channel 0 (EGFP-α-tubulin): {np.mean(r[0]):.3f} ± {np.std(r[0]):.3f}",
             f"* channel 1 (mCherry-H2B):    {np.mean(r[1]):.3f} ± {np.std(r[1]):.3f}"]
    os.makedirs("report/results", exist_ok=True)
    open("report/results/channel_check.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
