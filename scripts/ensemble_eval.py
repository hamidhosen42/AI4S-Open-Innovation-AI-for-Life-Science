"""Deep-ensemble evaluation from cached predictions (no new inference).

    python scripts/ensemble_eval.py --cache outputs/cache --out outputs/ensemble

For each configuration, the three seeds are combined: mu = mean of member means;
variance = variance of member means (ensemble disagreement) + mean member sigma^2
(learned aleatoric and/or TTA terms, when the member has them). Scored with exactly the
same per-image metrics as scripts/evaluate.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm

from chipstain.data import PairDataset, load_mask, make_splits
from scripts.evaluate import score_image, summarise

GROUPS = [("chipstain_nll", "_tta"), ("chipstain_nll", ""), ("baseline_unet", "_tta"), ("baseline_unet", ""), ("pretrained_l1", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="outputs/ensemble")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _, _, test = make_splits(a.data)
    ds = PairDataset(test, crop=None, augment=False, cache=False)
    gts = [ds[i][1][0, :540, :540].numpy() for i in range(len(ds))]
    refs = [load_mask(s) for s in test]
    rows = []
    for cfg, t in GROUPS:
        files = [os.path.join(a.cache, f"{cfg}_s{s}{t}.npz") for s in a.seeds]
        if not all(os.path.exists(f) for f in files):
            print("skip", cfg + t, "(missing cache)")
            continue
        zs = [np.load(f) for f in files]
        has_sigma = bool(zs[0]["has_sigma"])
        mus = np.stack([z["mu"] for z in zs]).astype(np.float32)          # (S, N, H, W)
        sig = np.stack([z["sigma"] for z in zs]).astype(np.float32) if has_sigma else None
        tag = f"ens{len(a.seeds)}_{cfg}{t}"
        for i in tqdm(range(mus.shape[1]), desc=tag):
            mu = mus[:, i].mean(0)
            var = mus[:, i].var(0) + ((sig[:, i] ** 2).mean(0) if sig is not None else 0.0)
            m, _ = score_image(mu, gts[i], np.sqrt(var), refs[i])
            m.update({"run": tag, "well": test[i].well, "field": test[i].field, "timepoint": test[i].timepoint})
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_image_test.csv"), index=False)
    summarise(df, a.out, "test")


if __name__ == "__main__":
    main()
