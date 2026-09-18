"""Gradio demo: upload a bright-field image -> predicted nuclear fluorescence + uncertainty.

    python demo/app.py --weights weights/chipstain.pt
"""
import argparse
import os
import sys

import gradio as gr
import numpy as np
import torch
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chipstain.metrics import segment_nuclei  # noqa: E402
from scripts.inference import load_model, read_grey, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.environ.get("CHIPSTAIN_WEIGHTS", "weights/chipstain.pt"))
ap.add_argument("--share", action="store_true")
args, _ = ap.parse_known_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, CFG = load_model(args.weights, DEVICE)
EXAMPLES = sorted(
    os.path.join("demo/examples", f) for f in os.listdir("demo/examples") if f.endswith((".tif", ".png"))
) if os.path.isdir("demo/examples") else []


def to_rgb(a, cmap, vmin=None, vmax=None):
    a = a.astype(np.float32)
    vmin = a.min() if vmin is None else vmin
    vmax = a.max() if vmax is None else vmax
    n = np.clip((a - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (matplotlib.colormaps[cmap](n)[..., :3] * 255).astype(np.uint8)


def predict(file, tta):
    bf = read_grey(file)
    pred, sigma = run(MODEL, bf, DEVICE, tta=tta)
    seg = segment_nuclei(pred)
    n = int(seg.max())
    hi = float(np.percentile(sigma, 99))
    summary = (
        f"**Detected nuclei:** {n}  \n"
        f"**Mean σ:** {sigma.mean():.4f} · **99th pct σ:** {hi:.4f}  \n"
        f"Bright regions in the uncertainty map are where the model is least confident — check those by eye before trusting a count."
    )
    return to_rgb(bf, "gray", *np.percentile(bf, [1, 99])), to_rgb(pred, "magma", 0, 1), to_rgb(sigma, "viridis", 0, hi), summary


with gr.Blocks(title="ChipStain") as demo:
    gr.Markdown(
        "# ChipStain — label-free nuclear staining with uncertainty\n"
        "Upload a bright-field image (TIFF/PNG). The model predicts the H2B/nuclear fluorescence channel and a per-pixel "
        "uncertainty map (σ). Trained on HeLa 'Kyoto' cells (Zenodo 10.5281/zenodo.6140064, CC BY 4.0)."
    )
    with gr.Row():
        inp = gr.File(label="Bright-field image", file_types=[".tif", ".tiff", ".png", ".jpg"], type="filepath")
        tta = gr.Checkbox(value=True, label="Test-time augmentation (8×, slower, better)")
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
