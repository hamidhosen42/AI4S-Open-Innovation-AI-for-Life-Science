"""16:9 copies of the Kaggle media-gallery images (Kaggle crops thumbnails to 16:9).

    python scripts/make_gallery.py      # report/figures/*.png -> writeup/assets/gallery_16x9/
Each figure is centred on a white 16:9 canvas (max 1920 x 1080); the multiseed figure is redrawn as 2 x 2 and the neural examples as 2 x 3.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image

from scripts.make_figures_extra import multiseed, neural_examples

OUT = "writeup/assets/gallery_16x9"
# gallery image -> report figure it is made from
SOURCES = {
    "gallery_1_qualitative.png": "report/figures/qualitative_chipstain_nll_s0_tta.png",
    "gallery_2_multiseed.png": "report/figures/multiseed.png",
    "gallery_3_shift_test.png": "report/figures/shift_test.png",
    "gallery_4_timelapse.png": "report/figures/timelapse.png",
    "gallery_5_demo_panel.png": "report/figures/demo_panel_dense_t150.png",
    "gallery_6_proliferation.png": "report/figures/proliferation.png",
    "gallery_7_neural.png": "report/figures/neural_examples.png",
}


def to_16x9(src, dst, pad=0.04):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    W = int(max(w, h * 16 / 9) * (1 + pad))
    H = int(W * 9 / 16)
    if H < h * (1 + pad):
        H = int(h * (1 + pad))
        W = int(H * 16 / 9)
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - w) // 2, (H - h) // 2))
    if W > 1920:
        canvas = canvas.resize((1920, 1080), Image.LANCZOS)
    canvas.save(dst)
    return canvas.size


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "_multiseed_2x2.png")
    tmp_n = os.path.join(OUT, "_neural_2x3.png")
    multiseed("report/results", tmp, grid=(2, 2), figsize=(10.5, 5.6))
    neural_examples(tmp_n, grid=(2, 3), figsize=(10, 6.6))
    redrawn = {"gallery_2_multiseed.png": tmp, "gallery_7_neural.png": tmp_n}
    for name, fig in SOURCES.items():
        src = redrawn.get(name, fig) if os.path.exists(redrawn.get(name, fig)) else fig
        print(os.path.join(OUT, name), to_16x9(src, os.path.join(OUT, name)))
    for t in redrawn.values():
        if os.path.exists(t):
            os.remove(t)


if __name__ == "__main__":
    main()
