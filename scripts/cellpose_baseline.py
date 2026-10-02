"""Direct bright-field nuclei segmentation baseline: the Cellpose models published with the
dataset (Zenodo 10.5281/zenodo.6140111, CC BY 4.0; trained by the dataset author on this
dataset's training split to segment nuclei from bright-field). Answers "why predict the stain
instead of segmenting nuclei directly?".

Runs in its own environment (Cellpose 3.x):
    python3 -m venv outputs/venv_cellpose && outputs/venv_cellpose/bin/pip install "cellpose<4" tifffile pandas scikit-image
    outputs/venv_cellpose/bin/python scripts/cellpose_baseline.py --model <path to bf nuclei model> --out report/results
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import tifffile
from cellpose import models

from chipstain.metrics import match_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_dir", default="data/raw/hela_kyoto/test")
    ap.add_argument("--diameter", type=float, default=0, help="0 = use the model's own mean diameter")
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    model = models.CellposeModel(pretrained_model=a.model, gpu=False)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.test_dir, "*_bf.tif"))):
        m = re.search(r"(R\d+-C\d+)-F(\d+)-(\d+)_bf\.tif$", f)
        bf = tifffile.imread(f).astype(np.float32)
        masks = model.eval(bf, diameter=(a.diameter or None), channels=[0, 0])[0]
        ref = tifffile.imread(f.replace("_bf.tif", "_nuclei.tif")).astype(np.int32)
        r = match_f1(masks.astype(np.int32), ref)
        rows.append({"well": m.group(1), "field": int(m.group(2)), "timepoint": int(m.group(3)), "seg_f1": r["f1"], "seg_precision": r["precision"],
                     "seg_recall": r["recall"], "n_pred": r["n_pred"], "n_ref": r["n_gt"]})
        print(os.path.basename(f), round(r["f1"], 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "cellpose_baseline.csv"), index=False)
    t = df.groupby("timepoint")[["seg_f1", "n_pred", "n_ref"]].mean()
    md = [f"Cellpose bright-field nuclei model `{os.path.basename(a.model)}` on the 125 test images:",
          f"F1 {df.seg_f1.mean():.3f} ± {df.seg_f1.std():.3f} (precision {df.seg_precision.mean():.3f}, recall {df.seg_recall.mean():.3f}); "
          f"mean |count error| {100 * ((df.n_pred - df.n_ref).abs() / df.n_ref).mean():.1f} %.", "",
          "| time-point | F1 | n_pred | n_ref |", "|---|---|---|---|"] + [f"| {i} | {r.seg_f1:.3f} | {r.n_pred:.0f} | {r.n_ref:.0f} |" for i, r in t.iterrows()]
    open(os.path.join(a.out, "cellpose_baseline.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
