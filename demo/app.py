"""Gradio demo: upload a bright-field image -> predicted nuclear fluorescence + uncertainty.

    python demo/app.py [--weights weights/chipstain.pt] [--share]
Runs from any working directory (paths are resolved from the repository root).
"""
import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chipstain.metrics import segment_nuclei  # noqa: E402
from scripts.inference import load_model, read_grey, run  # noqa: E402

# Mean sigma (8x TTA, released seed-0 checkpoint) above this is outside anything seen on clean images:
# maximum over the 50 validation images 0.173 (report/results/validation_sigma.csv) and over the 125 test
# images 0.133. On 50 shifted test images (report/results/shift_test.csv, seed 0) this fixed threshold flags
# 39/50 images blurred by 1 px, 50/50 blurred by 2 or 4 px, 50/50 phase-contrast inputs, 0/50 with 10 % noise
# and 15/50 with 20 % noise. It is stricter than the validation gate used in the report. Other training seeds
# do not show this sigma rise (report section 6.7), so a normal sigma does not prove the input is in domain.
OOD_SIGMA = 0.20
# reference nuclei (StarDist on the real H2B stain, from the dataset) for the bundled examples
REF = {"example_bf_sparse_t010.tif": 64, "example_bf_dense_t150.tif": 180, "example_bf_dense_t150_blur1px.tif": 180}

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.environ.get("CHIPSTAIN_WEIGHTS", str(ROOT / "weights" / "chipstain.pt")))
ap.add_argument("--share", action="store_true")
args, _ = ap.parse_known_args()
if not os.path.exists(args.weights):
    sys.exit(f"weights not found at {args.weights} - run: python scripts/download_weights.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, CFG = load_model(args.weights, DEVICE)
EX_DIR = ROOT / "demo" / "examples"
EXAMPLES = sorted(str(EX_DIR / f) for f in os.listdir(EX_DIR) if f.endswith((".tif", ".png")) and "_bf_" in f) if EX_DIR.is_dir() else []


def to_rgb(a, cmap, vmin=None, vmax=None):
    a = a.astype(np.float32)
    vmin = a.min() if vmin is None else vmin
    vmax = a.max() if vmax is None else vmax
    n = np.clip((a - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (matplotlib.colormaps[cmap](n)[..., :3] * 255).astype(np.uint8)


def predict(file, tta):
    if file is None:
        raise gr.Error("Upload a bright-field image first.")
    try:
        bf = read_grey(file)
    except Exception as e:  # noqa: BLE001 - show the reason in the UI
        raise gr.Error(f"Could not read the image: {e}")
    pred, sigma = run(MODEL, bf, DEVICE, tta=tta)
    n = int(segment_nuclei(pred).max())
    ref = REF.get(os.path.basename(str(file)))
    hi = float(np.percentile(sigma, 99))
    lines = []
    if tta and sigma.mean() > OOD_SIGMA:
        lines.append(f"⚠ **Mean σ = {sigma.mean():.3f} is above anything seen on clean validation images (≤ 0.17).** The input looks unlike the "
                     "training bright-field (defocus, another modality, fluorescence…). Treat the prediction and the count as unreliable.  \n")
    lines.append(f"**Detected nuclei:** {n}" + (f" · reference annotation (StarDist on the real H2B stain): {ref}" if ref else "") + "  \n")
    lines.append(f"**Mean σ:** {sigma.mean():.4f} · **99th pct σ:** {hi:.4f}" + ("" if tta else " · (warning check needs TTA)") + "  \n")
    lines.append("σ is the predicted uncertainty per pixel: it is higher on nuclei (brighter = noisier) and highest where the prediction is least reliable. "
                 "A low mean σ is necessary but not sufficient: it does not prove the input resembles the training data.")
    return to_rgb(bf, "gray", *np.percentile(bf, [1, 99])), to_rgb(pred, "magma", 0, 1), to_rgb(sigma, "viridis", 0, hi), "".join(lines)


with gr.Blocks(title="ChipStain") as demo:
    gr.Markdown(
        "# ChipStain — label-free nuclear staining with uncertainty\n"
        "Upload a bright-field image (TIFF/PNG). The model predicts the H2B nuclear fluorescence channel and a per-pixel "
        "uncertainty map σ. Trained on HeLa 'Kyoto' cells (R. Guiet, EPFL BIOP; Zenodo 10.5281/zenodo.6140064; CC BY 4.0). "
        "The third example is the dense example blurred by 1 px: with this released (seed-0) model σ rises and the app warns, but across three training runs σ rose under blur in only one — a normal σ does not prove the input is in domain."
    )
    with gr.Row():
        inp = gr.File(label="Bright-field image", file_types=[".tif", ".tiff", ".png", ".jpg"], type="filepath")
        tta = gr.Checkbox(value=True, label="Test-time augmentation (8×, slower, recommended)")
    btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        o1 = gr.Image(label="Input (bright-field)")
        o2 = gr.Image(label="Predicted nuclei (H2B)")
        o3 = gr.Image(label="Uncertainty σ")
    txt = gr.Markdown()
    btn.click(predict, [inp, tta], [o1, o2, o3, txt])
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=inp)

if __name__ == "__main__":
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
