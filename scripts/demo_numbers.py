"""Numbers quoted in the video: what the demo shows for the bundled clean and blurred examples
(released seed-0 weights, 8x TTA, CPU).   python scripts/demo_numbers.py -> report/results/demo_examples.json"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from chipstain.metrics import segment_nuclei
from scripts.inference import load_model, panel, read_grey, run

dev = torch.device("cpu")
model, _ = load_model("weights/chipstain.pt", dev)
out = {}
for key, f in [("clean", "demo/examples/example_bf_dense_t150.tif"), ("blur", "demo/examples/example_bf_dense_t150_blur1px.tif")]:
    bf = read_grey(f)
    pred, sigma = run(model, bf, dev, tta=True)
    panel(bf, pred, sigma, f"report/figures/demo_panel_{'dense_t150' if key == 'clean' else 'dense_t150_blur1px'}.png")
    out[key] = {"file": f, "count": int(segment_nuclei(pred).max()), "mean_sigma": float(sigma.mean())}
out["reference_count"] = 180  # StarDist annotation of R05-C03-F3 t150 (data/raw/hela_kyoto/test/*F3-150_nuclei.tif)
json.dump(out, open("report/results/demo_examples.json", "w"), indent=1)
print(out)
