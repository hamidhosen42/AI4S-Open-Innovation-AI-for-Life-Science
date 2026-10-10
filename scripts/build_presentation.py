"""Build the presentation: one self-contained HTML deck, the video script, and per-slide frames.

    python scripts/build_presentation.py
      -> presentation/ChipStain_presentation.html  (open in any browser; ← → keys, F fullscreen, N notes)
      -> presentation/VIDEO_SCRIPT.md              (recording guide + narration, from the speaker notes)
      -> presentation/frames/slide_NN.png/.txt     (slide images + narration, not tracked)
Every number comes from report/results/key_numbers.json; every figure from report/figures/.
"""
import base64
import html
import io
import json
import os
import re
import shutil
import subprocess

from PIL import Image

TPL = "templates/presentation_template.html"
OUT = "presentation"
CHROME = os.environ.get("CHROME") or next((c for c in (shutil.which("google-chrome"), shutil.which("chromium"),
                                                       "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome") if c and os.path.exists(c)), None)


def embed(name, max_w=1600):
    im = Image.open(os.path.join("report/figures", name)).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    vals = json.load(open("report/results/key_numbers.json"))
    s = open(TPL, encoding="utf-8").read()
    s = re.sub(r"\{\{fig:([\w.\-]+)\}\}", lambda m: embed(m.group(1)), s)
    for k, v in vals.items():
        s = s.replace("{{" + k + "}}", html.escape(str(v), quote=False))
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", s)))
    if left:
        print("WARNING unfilled:", left)
        s = re.sub(r"\{\{\w+\}\}", "—", s)
    os.makedirs(OUT, exist_ok=True)
    deck = os.path.join(OUT, "ChipStain_presentation.html")
    open(deck, "w", encoding="utf-8").write(s)

    # speaker notes -> narration
    sections = re.findall(r'<section class="slide[^"]*">(.*?)</section>', s, re.S)
    rows = []
    for k, sec in enumerate(sections, 1):
        title = re.search(r"<h1[^>]*>(.*?)</h1>", sec, re.S)
        title = re.sub(r"<[^>]+>", "", title.group(1)).strip() if title else "ChipStain"
        note = re.search(r'<aside class="notes">(.*?)</aside>', sec, re.S)
        rows.append((k, html.unescape(title), html.unescape(re.sub(r"\s+", " ", note.group(1)).strip()) if note else ""))

    frames = os.path.join(OUT, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    os.makedirs(frames)
    for k, title, note in rows:
        open(os.path.join(frames, f"slide_{k:02d}.txt"), "w", encoding="utf-8").write(note)
        if CHROME:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=1920,1080",
                            "--virtual-time-budget=8000", f"--screenshot={os.path.abspath(os.path.join(frames, f'slide_{k:02d}.png'))}",
                            "file://" + os.path.abspath(deck) + f"?export=1&slide={k}"], capture_output=True)

    words = sum(len(n.split()) for _, _, n in rows)
    md = ["# ChipStain — video script", "", "Final video (4 min 55 s): https://youtu.be/bxuVhUkyIOY", "",
          f"Deck: `presentation/ChipStain_presentation.html` ({len(rows)} slides). Narration: {words} words ≈ {words / 145:.1f} min at a calm pace — "
          "the finished video must stay **under 5:00**.", "",
          "## How to record", "",
          "1. Start the demo: `python demo/app.py --weights weights/chipstain.pt`, open http://localhost:7860, and run the first example once (warm-up). Keep *Test-time augmentation* ticked.",
          "2. Open the deck in Chrome, press **F** for fullscreen. Press **N** to see these lines as speaker notes (hide them before recording).",
          "3. Record the screen with QuickTime (File → New Screen Recording) or Loom, with your microphone.",
          "4. On **slides 7–8**, switch to the browser with the demo and do it live (the judges want to see the real system running):",
          "   - slide 7: click the example `example_bf_dense_t150.tif` → **Predict**; point at the three panels and the nuclei count.",
          "   - slide 8: click `example_bf_dense_t150_blur1px.tif` → **Predict**; point at the ⚠ warning and the higher σ.",
          "5. Trim, export 1080p, upload to YouTube as **Unlisted**, check it plays in a private window, and paste the link into the Kaggle Writeup.", "",
          "## Narration", "", "| # | Slide | Say |", "|---|---|---|"]
    md += [f"| {k} | {t} | {n} |" for k, t, n in rows]
    open(os.path.join(OUT, "VIDEO_SCRIPT.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"deck: {deck} ({os.path.getsize(deck) / 1e6:.1f} MB, {len(rows)} slides) | script: {OUT}/VIDEO_SCRIPT.md ({words} words) | frames: {frames}/")


if __name__ == "__main__":
    main()
