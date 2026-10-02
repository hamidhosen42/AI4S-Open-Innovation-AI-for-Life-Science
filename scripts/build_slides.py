"""Build the slide deck and the video script from writeup/slides_template.json, filling every number
from report/results/key_numbers.json (and prose from report/report_texts.json).

    python scripts/build_slides.py   -> writeup/slides/slide_XX.png, writeup/ChipStain_slides.pdf, writeup/video_script.md
"""
import base64
import html
import json
import os
import re
import shutil
import subprocess

OUT = "writeup/slides"
CHROME = os.environ.get("CHROME") or next((c for c in (shutil.which("google-chrome"), shutil.which("chromium"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome") if c and os.path.exists(c)), None)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#14181d;--ink2:#4a525b;--muted:#8a929a;--blue:#2a78d6;--blue-soft:#eaf2fc;--line:#dfe3e8;--ground:#ffffff}
*{box-sizing:border-box;margin:0}
html,body{width:1920px;height:1080px;background:var(--ground);font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink)}
.slide{width:1920px;height:1080px;padding:84px 110px 70px;display:flex;flex-direction:column;position:relative;overflow:hidden;page-break-after:always}
.kicker{font-family:'IBM Plex Mono',monospace;font-size:26px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-bottom:18px}
h1{font-size:68px;line-height:1.08;font-weight:700;letter-spacing:-.01em;max-width:1700px}
h1.big{font-size:150px;letter-spacing:-.03em}
.sub{font-size:40px;color:var(--ink2);margin-top:20px;max-width:1500px;line-height:1.25}
.body{flex:1;display:flex;gap:64px;margin-top:48px;min-height:0}
.col{flex:1;display:flex;flex-direction:column;gap:22px;min-width:0}
.col.fig{flex:1.45;align-items:center;justify-content:center}
.col.fig img{max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--line)}
ul{list-style:none;padding:0;display:flex;flex-direction:column;gap:26px}
li{font-size:36px;line-height:1.3;padding-left:38px;position:relative;color:var(--ink)}
li::before{content:"";position:absolute;left:0;top:16px;width:16px;height:16px;border-radius:3px;background:var(--blue)}
.stats{display:grid;grid-template-columns:1fr;gap:20px}
.stat{border-left:6px solid var(--blue);background:var(--blue-soft);padding:18px 26px}
.stat .k{font-family:'IBM Plex Mono',monospace;font-size:24px;color:var(--ink2);letter-spacing:.04em}
.stat .v{font-size:52px;font-weight:700;margin-top:4px}
.footer{position:absolute;left:110px;right:110px;bottom:28px;font-size:19px;color:var(--muted);display:flex;justify-content:space-between;border-top:1px solid var(--line);padding-top:12px}
.diagram{display:flex;align-items:center;gap:22px;margin:6px 0 10px}
.box{border:3px solid var(--ink);padding:22px 26px;font-size:30px;font-weight:600;text-align:center;background:#fff}
.box.model{border-color:var(--blue);color:var(--blue)}
.box small{display:block;font-size:22px;font-weight:400;color:var(--ink2);margin-top:6px}
.arrow{font-size:46px;color:var(--muted)}
.outs{display:flex;flex-direction:column;gap:14px}
.title-body{flex:1;display:flex;gap:60px;margin-top:40px;min-height:0;align-items:flex-end;padding-bottom:40px}
.title-body .col{flex:.8}.title-body .fig{flex:1.4;display:flex;justify-content:flex-end}.title-body img{max-width:100%;max-height:600px;border:1px solid var(--line)}
.wide{flex:1;display:flex;flex-direction:column;gap:30px;margin-top:36px;min-height:0}
.wide .row{display:flex;gap:28px;flex-wrap:wrap}
.wide .row .stat{flex:1;min-width:300px}
.wide .row ul{flex-direction:row;flex-wrap:wrap;gap:14px 44px}
.wide .row li{font-size:32px}
.wide .figbox{flex:1;min-height:0;display:flex;justify-content:center;align-items:center}
.wide .figbox img{max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--line)}
"""


def b64(name):
    p = os.path.join("report/figures", name)
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else None


def aspect(name):
    from PIL import Image
    p = os.path.join("report/figures", name)
    if not os.path.exists(p):
        return 1.0
    w, h = Image.open(p).size
    return w / h


def fill(text, vals):
    for _ in range(2):
        for k, v in vals.items():
            text = text.replace("{{" + k + "}}", str(v))
    return text


def slide_html(s, i, n, footer):
    parts = [f"<div class='slide'><div class='kicker'>{html.escape(s.get('kicker', ''))}</div>"]
    if s["id"] == "title":
        parts.append(f"<h1 class='big'>{html.escape(s['title'])}</h1><div class='sub'>{html.escape(s['subtitle'])}</div>")
        src = b64(s["figure"]) if s.get("figure") else None
        parts.append("<div class='title-body'><div class='col'><ul>" + "".join(f"<li>{html.escape(b)}</li>" for b in s.get("bullets", [])) + "</ul></div>"
                     + (f"<div class='fig'><img src='{src}'></div>" if src else "") + "</div>")
    elif s.get("figure") and aspect(s["figure"]) > 2.2:
        parts.append(f"<h1>{html.escape(s['title'])}</h1><div class='wide'><div class='row'>")
        if s.get("stats"):
            parts.append("".join(f"<div class='stat'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div></div>" for k, v in s["stats"]))
        if s.get("bullets"):
            parts.append("<ul>" + "".join(f"<li>{html.escape(b)}</li>" for b in s["bullets"] if b.strip()) + "</ul>")
        parts.append(f"</div><div class='figbox'><img src='{b64(s['figure'])}'></div></div>")
    else:
        parts.append(f"<h1>{html.escape(s['title'])}</h1><div class='body'><div class='col'>")
        if s.get("diagram"):
            parts.append("<div class='diagram'><div class='box'>bright-field<small>one plane</small></div><div class='arrow'>→</div>"
                         "<div class='box model'>U-Net<small>β-NLL · 8× TTA</small></div><div class='arrow'>→</div>"
                         "<div class='outs'><div class='box'>μ — predicted H2B</div><div class='box'>σ — uncertainty</div></div></div>")
        if s.get("stats"):
            parts.append("<div class='stats'>" + "".join(f"<div class='stat'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div></div>" for k, v in s["stats"]) + "</div>")
        if s.get("bullets"):
            parts.append("<ul>" + "".join(f"<li>{html.escape(b)}</li>" for b in s["bullets"] if b.strip()) + "</ul>")
        parts.append("</div>")
        src = b64(s["figure"]) if s.get("figure") else None
        if src:
            parts.append(f"<div class='col fig'><img src='{src}'></div>")
        parts.append("</div>")
    parts.append(f"<div class='footer'><span>{html.escape(footer)}</span><span>{i}/{n}</span></div></div>")
    return "".join(parts)


def main():
    vals = json.load(open("report/results/key_numbers.json"))
    if os.path.exists("report/report_texts.json"):
        vals.update(json.load(open("report/report_texts.json")))
    tpl = json.load(open("writeup/slides_template.json"))
    slides = json.loads(fill(json.dumps(tpl["slides"], ensure_ascii=False), {k: str(v).replace('"', "'") for k, v in vals.items()}))
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", json.dumps(slides, ensure_ascii=False))))
    if left:
        print("WARNING unfilled:", left)
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    pages = []
    for i, s in enumerate(slides, 1):
        body = slide_html(s, i, len(slides), tpl["footer"])
        pages.append(body)
        page = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
        hp = os.path.abspath(os.path.join(OUT, f"slide_{i:02d}.html"))
        open(hp, "w", encoding="utf-8").write(page)
        if CHROME:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080", "--virtual-time-budget=6000",
                            f"--screenshot={os.path.abspath(os.path.join(OUT, f'slide_{i:02d}.png'))}", "file://" + hp], capture_output=True)
        open(os.path.join(OUT, f"slide_{i:02d}.txt"), "w", encoding="utf-8").write(s["narration"])
    deck = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}@page{{size:1920px 1080px;margin:0}}</style></head><body>{''.join(pages)}</body></html>"
    dp = os.path.abspath("writeup/ChipStain_slides.html")
    open(dp, "w", encoding="utf-8").write(deck)
    if CHROME:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                        f"--print-to-pdf={os.path.abspath('writeup/ChipStain_slides.pdf')}", "file://" + dp], capture_output=True)
    # video script, from the same narration
    md = ["# ChipStain — video script (generated by scripts/build_slides.py — edit writeup/slides_template.json, not this file)", "",
          "The narrated draft video `writeup/video/ChipStain_video.mp4` is built from these slides and lines by `scripts/build_video.py`. "
          "For the submission you may instead record yourself over the same slides; replace slides 4–5 with a live screen recording of `python demo/app.py` "
          "(examples `example_bf_dense_t150.tif` and `example_bf_dense_t150_blur1px.tif`, TTA on) to show the real system running.", "",
          "| # | Slide | Narration |", "|---|---|---|"]
    for i, s in enumerate(slides, 1):
        md.append(f"| {i} | {s['title']} | {s['narration']} |")
    md.append("")
    md.append(f"Narration: {sum(len(s['narration'].split()) for s in slides)} words.")
    open("writeup/video_script.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"slides: {len(slides)} -> {OUT}/, writeup/ChipStain_slides.pdf, writeup/video_script.md")


if __name__ == "__main__":
    main()
