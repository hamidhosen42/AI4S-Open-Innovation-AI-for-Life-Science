"""60 h label-free nuclear read-out of one held-out field (240 frames, every 15 min).

    curl -L -o "data/raw/hela_timelapse/R05-C03-F0.tif" "https://zenodo.org/records/6139958/files/20210904_TL2%20-%20R05-C03-F0.tif?download=1"
    python scripts/timelapse.py --weights runs/chipstain_nll_s0/best.pt --out report/results

Source: Zenodo 10.5281/zenodo.6139958 (R. Guiet, EPFL BIOP, CC BY 4.0), field F0 of test well R05-C03.
The channel order is verified by bit-matching frames 1/10/50/100/150 against the test files.
Every bright-field frame is predicted (8x TTA); nuclei are counted with the same watershed on the
prediction and on the real H2B frame; mean sigma is recorded per frame.
Note: frames 1, 10, 50, 100 and 150 of this field are also in the test set; the other 235 are not.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from chipstain.data import normalize_target
from chipstain.metrics import image_metrics, match_f1, segment_nuclei
from scripts.evaluate import get_device
from scripts.inference import load_model, run


def find_layout(stack, test_dir):
    """Return (frames, bf_channel, h2b_channel) by matching against the test-set files of field F0."""
    ref = {}
    for t in (1, 10, 50, 100, 150):
        f = glob.glob(os.path.join(test_dir, f"*R05-C03-F0-{t:03d}_bf.tif"))
        if f:
            ref[t] = (tifffile.imread(f[0]), tifffile.imread(f[0].replace("_bf.tif", "_fluo.tif"))[1])
    if stack.ndim == 4:                      # (T, C, H, W)
        frames = stack
    elif stack.ndim == 3:                    # (T*C, H, W)
        for c in (5, 4, 3, 6):
            if stack.shape[0] % c == 0:
                frames = stack.reshape(-1, c, *stack.shape[1:])
                break
    else:
        raise ValueError(f"unexpected stack shape {stack.shape}")
    t0, (bf0, h2b0) = next(iter(ref.items()))
    bf_c = next(c for c in range(frames.shape[1]) if np.array_equal(frames[t0 - 1, c], bf0))
    h2b_c = next(c for c in range(frames.shape[1]) if np.array_equal(frames[t0 - 1, c], h2b0))
    ok = all(np.array_equal(frames[t - 1, bf_c], b) and np.array_equal(frames[t - 1, h2b_c], h) for t, (b, h) in ref.items())
    return frames, bf_c, h2b_c, ok, sorted(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", default="data/raw/hela_timelapse/R05-C03-F0.tif")
    ap.add_argument("--weights", default="runs/chipstain_nll_s0/best.pt")
    ap.add_argument("--test_dir", default="data/raw/hela_kyoto/test")
    ap.add_argument("--out", default="report/results")
    ap.add_argument("--save_frames", type=int, nargs="+", default=[1, 61, 121, 181, 240])
    a = ap.parse_args()
    stack = tifffile.imread(a.tif)
    frames, bf_c, h2b_c, ok, matched = find_layout(stack, a.test_dir)
    print(f"stack {stack.shape} -> frames {frames.shape}; bf channel {bf_c}, H2B channel {h2b_c}; bit-match with test files {matched}: {ok}")
    device = get_device()
    model, _ = load_model(a.weights, device)
    rows, keep = [], {}
    for t in tqdm(range(frames.shape[0]), desc="frames"):
        bf = frames[t, bf_c].astype(np.float32)
        real = normalize_target(frames[t, h2b_c])
        pred, sigma = run(model, bf, device, tta=True)
        seg_p, seg_r = segment_nuclei(pred), segment_nuclei(real)
        f = match_f1(seg_p, seg_r)
        m = image_metrics(pred, real)
        rows.append({"frame": t + 1, "hours": t * 0.25, "n_pred": int(seg_p.max()), "n_real": int(seg_r.max()), "f1_vs_real": f["f1"],
                     "mean_sigma": float(sigma.mean()), "pearson": m["pearson"], "mae": m["mae"], "in_test_set": (t + 1) in (1, 10, 50, 100, 150)})
        if (t + 1) in a.save_frames:
            keep[t + 1] = (bf, real, pred, sigma)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "timelapse.csv"), index=False)
    np.savez_compressed(os.path.join("outputs", "timelapse_frames.npz"), **{f"{k}_{t}": v for t, arrs in keep.items() for k, v in zip(["bf", "real", "pred", "sigma"], arrs)})

    def dt(col, lo=0, hi=None):
        d = df[(df.hours >= lo) & (df.hours <= (hi if hi is not None else df.hours.max()))]
        b = np.polyfit(d.hours, np.log(d[col].clip(lower=1)), 1)[0]
        return np.log(2) / b
    lines = [f"Field R05-C03-F0, {len(df)} frames over {df.hours.max():.2f} h (bit-matched layout: {ok}).",
             f"Doubling time over the whole time-lapse: predicted {dt('n_pred'):.1f} h vs real-stain count {dt('n_real'):.1f} h (same watershed).",
             f"Count agreement: mean |n_pred − n_real| / n_real = {100 * ((df.n_pred - df.n_real).abs() / df.n_real.clip(lower=1)).mean():.1f} %; Pearson r(n_pred, n_real) = {np.corrcoef(df.n_pred, df.n_real)[0, 1]:.3f}.",
             f"Per-frame object F1 (prediction vs real-stain segmentation): median {df.f1_vs_real.median():.3f} (IQR {df.f1_vs_real.quantile(.25):.3f}–{df.f1_vs_real.quantile(.75):.3f}).",
             f"Mean σ per frame: median {df.mean_sigma.median():.4f}, range {df.mean_sigma.min():.4f}–{df.mean_sigma.max():.4f}; Spearman ρ(mean σ, 1 − F1) = {df[['mean_sigma']].assign(e=1 - df.f1_vs_real).corr(method='spearman').iloc[0, 1]:.3f}."]
    open(os.path.join(a.out, "timelapse.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
