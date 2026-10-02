"""Biological read-out: population doubling time per field from label-free nuclear counts.

    python scripts/proliferation.py --csv report/results/multiseed_per_image.csv report/results/ensemble_per_image.csv --out report/results

The five time-points are frames 1, 10, 50, 100, 150 of a 15-min time-lapse (0-37.25 h).
For each field we fit log(count) = a + b*t by least squares and report doubling time
ln2/b, comparing counts from the predicted image with counts of the StarDist reference
annotation (made on the real H2B stain) and with the same watershed run on the real H2B.
"""
import argparse
import os

import numpy as np
import pandas as pd

HOURS = {1: 0.0, 10: 2.25, 50: 12.25, 100: 24.75, 150: 37.25}


def doubling(df, col):
    out = {}
    for f, g in df.groupby("field"):
        g = g.sort_values("timepoint")
        t = g["timepoint"].map(HOURS).values
        n = g[col].clip(lower=1).values
        b = np.polyfit(t, np.log(n), 1)[0]
        out[f] = np.log(2) / b if b > 0 else np.nan
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="report/results")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(c) for c in a.csv if os.path.exists(c)], ignore_index=True)
    ref = doubling(df[df["run"] == df["run"].iloc[0]], "n_ref")
    rows = [{"run": "StarDist reference (real H2B)", "median_dt_h": ref.median(), "bias_pct": 0.0, "mape_pct": 0.0}]
    real = doubling(df[df["run"] == df["run"].iloc[0]], "n_realfluo") if "n_realfluo" in df else None
    if real is not None:
        rows.append({"run": "watershed on real H2B", "median_dt_h": real.median(), "bias_pct": 100 * ((real - ref) / ref).mean(), "mape_pct": 100 * ((real - ref).abs() / ref).mean()})
    per_field = {"reference": ref}
    for run, g in df.groupby("run"):
        dt = doubling(g, "n_pred")
        per_field[run] = dt
        rows.append({"run": run, "median_dt_h": dt.median(), "bias_pct": 100 * ((dt - ref) / ref).mean(), "mape_pct": 100 * ((dt - ref).abs() / ref).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "proliferation.csv"), index=False)
    pd.DataFrame(per_field).to_csv(os.path.join(a.out, "proliferation_per_field.csv"))
    md = ["| counts from | median doubling time (h) | mean bias vs reference | mean abs. % error |", "|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {r.run} | {r.median_dt_h:.1f} | {r.bias_pct:+.1f} % | {r.mape_pct:.1f} % |")
    open(os.path.join(a.out, "proliferation.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
