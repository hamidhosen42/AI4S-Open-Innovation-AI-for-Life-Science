"""Input-level drift check: can a cheap test on the bright-field INPUT catch the shifts that sigma misses?

    python scripts/input_qc.py --runs runs/chipstain_nll_s{0,1,2} --out report/results

Same 50 test images (time-points 10 and 100) and the same simulated shifts as scripts/shift_test.py
(identical corruption and random seeds). Two checks, both fitted without any shifted image:
  * focus      log variance of the Laplacian of the normalised input (model-free). Two-sided: blur lowers it,
               noise raises it. Score = |log VoL - median on training images|.
  * features   Mahalanobis distance of the ChipStain encoder's deepest feature map (global average pool,
               512-d; Ledoit-Wolf covariance fitted on the 200 training images), one per training seed.
  * either     an image is flagged if either check flags it.
  * focus or σ  the focus check combined with the validation-calibrated σ gate of shift_test.py (σ catches noise,
               focus catches blur and modality change).
Thresholds = 95th percentile of the score on the 50 clean VALIDATION images (same rule as the sigma gate in
shift_test.py). Reported: shift-detection AUROC (shifted vs clean test images), fraction of images flagged per
condition, and failure-detection AUROC (failed r < 0.5 vs successful r > 0.7, clean and shifted pooled, using the
per-image Pearson r from shift_test.csv) next to the same numbers for total sigma.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from chipstain.data import load_pair, make_splits, normalize_input, pad_to_multiple
from scripts.evaluate import get_device, load_run
from scripts.shift_test import CONDITIONS, FAMILIES, auroc, corrupt


def focus(x):
    return float(np.log(ndi.laplace(x.astype(np.float64)).var()))


@torch.no_grad()
def features(model, x, device):
    xp, _ = pad_to_multiple(x)
    f = model.net.encoder(torch.from_numpy(xp)[None, None].to(device))[-1]
    return f.mean(dim=(-2, -1))[0].cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"runs/chipstain_nll_s{s}" for s in (0, 1, 2)])
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--val", default="outputs/multiseed_val/per_image_val.csv")
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    device = get_device()
    train, val, test = make_splits(a.data)
    samples = [s for s in test if s.timepoint in (10, 100)]
    clean_in = lambda ss: [normalize_input(load_pair(s)[0].astype(np.float32)) for s in ss]  # noqa: E731
    tr_x, va_x = clean_in(train), clean_in(val)

    # shifted test inputs, exactly as in shift_test.py
    test_x = []
    for i, s in enumerate(samples):
        bf, _ = load_pair(s)
        dpc = tifffile.imread(s.base + "_dpc.tif").astype(np.float32)
        for name, kind in CONDITIONS:
            rng = np.random.default_rng(1000 * i + len(name))
            test_x.append((name, i, normalize_input(corrupt(bf.astype(np.float32), dpc, kind, rng))))

    ref = np.median([focus(x) for x in tr_x])
    f_val = np.array([abs(focus(x) - ref) for x in va_x])
    f_test = np.array([abs(focus(x) - ref) for _, _, x in test_x])
    rows = []
    for run in a.runs:
        tag = os.path.basename(run.rstrip("/"))
        seed = int(tag.rsplit("_s", 1)[1])
        model, _ = load_run(run, device)
        model.eval()
        lw = LedoitWolf().fit(np.stack([features(model, x, device) for x in tqdm(tr_x, desc=f"{tag} train")]))
        m_val = lw.mahalanobis(np.stack([features(model, x, device) for x in va_x]))
        m_test = lw.mahalanobis(np.stack([features(model, x, device) for _, _, x in tqdm(test_x, desc=f"{tag} test")]))
        thr_f, thr_m = np.quantile(f_val, 0.95), np.quantile(m_val, 0.95)
        for k, (name, i, _) in enumerate(test_x):
            rows.append({"cfg": "chipstain_nll", "seed": seed, "condition": name, "image": i,
                         "focus_score": f_test[k], "feature_score": float(m_test[k]),
                         "flag_focus": bool(f_test[k] > thr_f), "flag_features": bool(m_test[k] > thr_m)})
        rows_val = {"seed": seed, "thr_focus": float(thr_f), "thr_features": float(thr_m),
                    "val_flag_either": float(((f_val > thr_f) | (m_val > thr_m)).mean())}
        print(rows_val)
    df = pd.DataFrame(rows)
    df["flag_either"] = df.flag_focus | df.flag_features
    st = pd.read_csv(os.path.join(a.out, "shift_test.csv"))
    st = st[st.cfg == "chipstain_nll"][["seed", "condition", "image", "pearson", "mean_sigma"]]
    df = df.merge(st, on=["seed", "condition", "image"], how="left")
    v = pd.read_csv(a.val)
    thr_s = {sd: v[v.run == f"chipstain_nll_s{sd}_tta"].mean_sigma.quantile(0.95) for sd in df.seed.unique()}
    df["flag_focus_or_sigma"] = df.flag_focus | (df.mean_sigma > df.seed.map(thr_s))
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "input_qc.csv"), index=False)

    scores = [("focus (variance of Laplacian)", "focus_score"), ("encoder features (Mahalanobis)", "feature_score"), ("total σ (for comparison)", "mean_sigma")]
    det, fail = [], []
    for seed, g in df.groupby("seed"):
        clean = g[g.condition == "clean"]
        for lab, col in scores:
            for fam, conds in FAMILIES.items():
                det.append({"seed": seed, "check": lab, "family": fam, "auroc": auroc(clean[col], g[g.condition.isin(conds)][col])})
            fail.append({"seed": seed, "check": lab, "auroc": auroc(g[g.pearson > 0.7][col], g[g.pearson < 0.5][col])})
    det, fail = pd.DataFrame(det), pd.DataFrame(fail)
    order = [c for c, _ in CONDITIONS]
    flags = df.groupby(["seed", "condition"])[["flag_focus", "flag_features", "flag_either", "flag_focus_or_sigma"]].mean().reset_index()
    det.to_csv(os.path.join(a.out, "input_qc_detection.csv"), index=False)
    flags.to_csv(os.path.join(a.out, "input_qc_flags.csv"), index=False)

    pm = lambda x: f"{np.mean(x):.2f} ± {np.std(x, ddof=1):.2f}" if len(x) > 1 else f"{np.mean(x):.2f}"  # noqa: E731
    lines = ["## Shift detection (AUROC, shifted vs clean test images; mean ± s.d. over 3 ChipStain seeds)", "",
             "| check | blur | noise | modality (DPC) | failure AUROC (r < 0.5 vs r > 0.7) |", "|---|---|---|---|---|"]
    for lab, _ in scores:
        d, f = det[det.check == lab], fail[fail.check == lab]
        lines.append(f"| {lab} | " + " | ".join(pm(d[d.family == fam].auroc.values) for fam in FAMILIES) + f" | {pm(f.auroc.values)} |")
    lines += ["", "## Validation-calibrated gate (95th percentile on clean validation images): fraction of images flagged, mean over seeds (min–max)", "",
              "| check | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for col, lab in [("flag_focus", "focus"), ("flag_features", "encoder features"), ("flag_either", "either"), ("flag_focus_or_sigma", "focus or σ gate")]:
        cells = []
        for c in order:
            v = 100 * flags[flags.condition == c][col].values
            cells.append(f"{v.mean():.0f} %" + (f" ({v.min():.0f}–{v.max():.0f})" if v.max() - v.min() >= 1 else ""))
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    open(os.path.join(a.out, "input_qc.md"), "w").write("\n".join(lines) + "\n")
    json.dump({"focus_reference": float(ref), "focus_threshold": float(np.quantile(f_val, 0.95))},
              open(os.path.join(a.out, "input_qc_focus.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
