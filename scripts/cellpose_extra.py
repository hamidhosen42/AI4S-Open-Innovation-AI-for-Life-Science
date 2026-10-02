"""Two Cellpose experiments (run in the Cellpose environment, see scripts/cellpose_baseline.py).

  --mode shift      the dataset author's bright-field nuclei model under the same simulated shifts as
                    scripts/shift_test.py (50 test images, time-points 10 and 100): does direct
                    segmentation degrade, and does anything warn the user? (Cellpose returns no uncertainty.)
  --mode predicted  Cellpose's generic fluorescence 'nuclei' model applied to (a) the real H2B and
                    (b) ChipStain + TTA predictions (cached, 3 seeds): F1 against the StarDist references.
                    Tests how much of the downstream gap is the simple watershed rather than the prediction.
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
from scipy import ndimage as ndi

from chipstain.metrics import match_f1

TEST = "data/raw/hela_kyoto/test"


def test_files():
    out = []
    for f in sorted(glob.glob(os.path.join(TEST, "*_bf.tif"))):
        m = re.search(r"(R\d+-C\d+)-F(\d+)-(\d+)_bf\.tif$", f)
        out.append((f, m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def corrupt(bf, dpc, name, rng):
    if name == "clean":
        return bf
    if name.startswith("blur"):
        return ndi.gaussian_filter(bf, float(name.split("=")[1]))
    if name.startswith("noise"):
        lo, hi = np.percentile(bf, [1, 99])
        return bf + rng.normal(0, float(name.split("=")[1]) * (hi - lo), bf.shape)
    if name == "modality: DPC":
        return dpc
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shift", "predicted"], required=True)
    ap.add_argument("--bf_model", default=None)
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    files = test_files()
    rows = []
    if a.mode == "shift":
        model = models.CellposeModel(pretrained_model=a.bf_model, gpu=False)
        conds = ["clean", "blur=1", "blur=2", "blur=4", "noise=0.05", "noise=0.1", "noise=0.2", "modality: DPC"]
        for i, (f, well, field, tp) in enumerate([x for x in files if x[3] in (10, 100)]):
            bf = tifffile.imread(f).astype(np.float32)
            dpc = tifffile.imread(f.replace("_bf.tif", "_dpc.tif")).astype(np.float32)
            ref = tifffile.imread(f.replace("_bf.tif", "_nuclei.tif")).astype(np.int32)
            for c in conds:
                x = corrupt(bf, dpc, c, np.random.default_rng(1000 * i + len(c)))
                masks = model.eval(x, channels=[0, 0])[0]
                r = match_f1(masks.astype(np.int32), ref)
                rows.append({"image": i, "field": field, "timepoint": tp, "condition": c, "f1": r["f1"], "n_pred": r["n_pred"], "n_ref": r["n_gt"]})
            print(i, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, "cellpose_shift.csv"), index=False)
        t = df.groupby("condition", sort=False)[["f1"]].mean().join(df.assign(err=(df.n_pred - df.n_ref).abs() / df.n_ref).groupby("condition", sort=False)["err"].mean())
        md = ["Cellpose bright-field nuclei model under simulated shifts (50 test images): nuclei F1 vs StarDist references and mean relative count error.", "",
              "| condition | F1 | count error |", "|---|---|---|"] + [f"| {c} | {r.f1:.3f} | {100 * r.err:.0f} % |" for c, r in t.iterrows()]
    else:
        model = models.Cellpose(gpu=False, model_type="nuclei")
        refs = [tifffile.imread(f.replace("_bf.tif", "_nuclei.tif")).astype(np.int32) for f, *_ in files]
        sources = {"real H2B": [tifffile.imread(f.replace("_bf.tif", "_fluo.tif"))[1].astype(np.float32) for f, *_ in files]}
        for s in (0, 1, 2):
            p = os.path.join(a.cache, f"chipstain_nll_s{s}_tta.npz")
            if os.path.exists(p):
                sources[f"ChipStain + TTA s{s}"] = [m.astype(np.float32) for m in np.load(p)["mu"]]
        for name, imgs in sources.items():
            for i, (img, ref) in enumerate(zip(imgs, refs)):
                masks = model.eval(np.clip(img, 0, None), diameter=None, channels=[0, 0])[0]
                r = match_f1(masks.astype(np.int32), ref)
                rows.append({"source": name, "image": i, "timepoint": files[i][3], "f1": r["f1"], "n_pred": r["n_pred"], "n_ref": r["n_gt"]})
            print(name, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, "cellpose_on_predictions.csv"), index=False)
        df["src"] = df["source"].str.replace(r" s\d$", "", regex=True)
        per = df.groupby(["src", "source"])["f1"].mean().groupby("src").agg(["mean", "std"])
        md = ["Cellpose generic fluorescence 'nuclei' model (automatic diameter) applied to the real H2B and to ChipStain + TTA predictions; "
              "F1 against the StarDist references, mean over 125 test images (± s.d. over seeds).", "", "| segmented image | nuclei F1 |", "|---|---|"]
        md += [f"| {k} | {r['mean']:.3f}" + (f" ± {r['std']:.3f}" if not np.isnan(r["std"]) else "") + " |" for k, r in per.iterrows()]
    open(os.path.join(a.out, f"cellpose_{a.mode}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
