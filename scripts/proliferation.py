"""Biological read-out: population doubling time per field from label-free nuclear counts,
with and without an uncertainty gate whose threshold is fixed on the VALIDATION split.

    python scripts/proliferation.py --test report/results/multiseed_per_image.csv \
        --val outputs/multiseed_val/per_image_val.csv --out report/results

Time-points are frames 1, 10, 50, 100, 150 of a 15-min time-lapse (0-37.25 h). Per field we fit
log(count) = a + b*t by least squares; doubling time = ln2 / b. Counts from the prediction are
compared with counts of the StarDist reference annotation (made on the real H2B stain).
Gate: a frame's relative uncertainty is its mean sigma divided by the median mean sigma of the
frames with the same time-point (same seed, same split). The threshold is the 95th percentile
of that ratio on the validation split, i.e. a 5 % false-alarm budget set without test labels.
Flagged frames are dropped from the fit; a field with fewer than 3 frames left is reported as
"needs review" and excluded from the population estimate.
"""
import argparse
import os

import numpy as np
import pandas as pd

HOURS = {1: 0.0, 10: 2.25, 50: 12.25, 100: 24.75, 150: 37.25}
MODELS = {"chipstain_nll+tta": "ChipStain + TTA", "baseline_unet+tta": "U-Net + TTA"}


def split_cfg(df):
    m = df["run"].str.extract(r"^(?P<cfg>.+)_s(?P<seed>\d+)(?P<t>_tta)?$")
    df = df.assign(cfg=m["cfg"] + np.where(m["t"].notna(), "+tta", ""), seed=m["seed"].astype(int))
    return df


def rel_sigma(df):
    return df["mean_sigma"] / df.groupby(["cfg", "seed", "timepoint"])["mean_sigma"].transform("median")


def fit_dt(g, col):
    g = g.sort_values("timepoint")
    if len(g) < 3:
        return np.nan
    b = np.polyfit(g["timepoint"].map(HOURS), np.log(g[col].clip(lower=1)), 1)[0]
    return np.log(2) / b if b > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="report/results/multiseed_per_image.csv")
    ap.add_argument("--val", default="outputs/multiseed_val/per_image_val.csv")
    ap.add_argument("--out", default="report/results")
    ap.add_argument("--false_alarm", type=float, default=0.05)
    a = ap.parse_args()
    te = pd.read_csv(a.test)
    if "cfg" not in te:
        te = split_cfg(te)
    te = te[te["cfg"].isin(MODELS)].copy()
    va = split_cfg(pd.read_csv(a.val)) if os.path.exists(a.val) else None
    te["rel_sigma"] = rel_sigma(te)
    thr = {}
    for cfg in MODELS:
        if va is not None and (va["cfg"] == cfg).any():
            v = va[va["cfg"] == cfg].copy()
            thr[cfg] = float(np.quantile(rel_sigma(v), 1 - a.false_alarm))
    te["flag"] = [r.rel_sigma > thr.get(r.cfg, np.inf) for r in te.itertuples()]
    te["failed_frame"] = te["seg_f1"] < 0.5  # a frame whose nuclei detection clearly failed

    first = te[(te.cfg == list(MODELS)[0]) & (te.seed == te.seed.min())]
    ref = first.groupby("field").apply(lambda g: fit_dt(g, "n_ref"), include_groups=False)
    rows, per_field, curves = [], {"reference": ref}, []
    for f, g in first.groupby("field"):
        for _, r in g.iterrows():
            curves.append({"source": "StarDist reference (real H2B)", "seed": -1, "field": f, "hours": HOURS[r.timepoint], "count": r.n_ref})
    for (cfg, seed), g in te.groupby(["cfg", "seed"]):
        name = MODELS[cfg]
        dt_all = g.groupby("field").apply(lambda h: fit_dt(h, "n_pred"), include_groups=False)
        dt_gate = g[~g.flag].groupby("field").apply(lambda h: fit_dt(h, "n_pred"), include_groups=False).reindex(dt_all.index)
        per_field[f"{name} s{seed}"] = dt_all
        per_field[f"{name} s{seed} gated"] = dt_gate
        for _, r in g.iterrows():
            curves.append({"source": name, "seed": seed, "field": r.field, "hours": HOURS[r.timepoint], "count": r.n_pred})
        n_fail, n_flag = int(g.failed_frame.sum()), int(g.flag.sum())
        caught = int((g.failed_frame & g.flag).sum())
        for label, dt in [("all frames", dt_all), ("σ-gated", dt_gate)]:
            ok = dt.notna()
            rows.append({"model": name, "seed": seed, "frames": label,
                         "pop_dt_h": float(np.exp(np.log(dt[ok]).mean())) if ok.any() else np.nan,
                         "bias_pct": float(100 * ((dt[ok] - ref[ok]) / ref[ok]).mean()),
                         "mape_pct": float(100 * ((dt[ok] - ref[ok]).abs() / ref[ok]).mean()),
                         "fields_needing_review": int((~ok).sum()),
                         "frames_flagged": n_flag if label == "σ-gated" else 0,
                         "failed_frames": n_fail, "failed_frames_flagged": caught if label == "σ-gated" else 0})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "proliferation.csv"), index=False)
    pd.DataFrame(per_field).to_csv(os.path.join(a.out, "proliferation_per_field.csv"))
    pd.DataFrame(curves).to_csv(os.path.join(a.out, "proliferation_curves.csv"), index=False)
    te[["cfg", "seed", "field", "timepoint", "mean_sigma", "rel_sigma", "flag", "seg_f1", "failed_frame", "n_pred", "n_ref"]].to_csv(os.path.join(a.out, "proliferation_frames.csv"), index=False)
    md = [f"Reference (StarDist on real H2B): population doubling time {np.exp(np.log(ref).mean()):.1f} h (geometric mean of 25 fields).",
          f"Gate thresholds (relative mean σ, {100*(1-a.false_alarm):.0f}th percentile on validation): " + ", ".join(f"{MODELS[k]} {v:.2f}" for k, v in thr.items()), "",
          "| model | seed | frames used | doubling time (h) | mean bias | mean abs. error | fields to review | frames flagged | failed frames (F1 < 0.5) caught |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        caught = f"{r.failed_frames_flagged}/{r.failed_frames}" if r.frames == "σ-gated" else f"—/{r.failed_frames}"
        md.append(f"| {r.model} | {r.seed} | {r.frames} | {r.pop_dt_h:.1f} | {r.bias_pct:+.1f} % | {r.mape_pct:.1f} % | {r.fields_needing_review} | {r.frames_flagged} | {caught} |")
    open(os.path.join(a.out, "proliferation.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
