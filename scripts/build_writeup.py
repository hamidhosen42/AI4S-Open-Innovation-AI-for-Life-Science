"""Render README.md and writeup/kaggle_writeup.md from their templates, filling every number
from report/results/key_numbers.json (written by scripts/build_report.py) and the prose
bullets from templates/report_texts.json, so the README, the Writeup and the report agree.

    python scripts/build_report.py && python scripts/build_writeup.py
"""
import json
import os
import re
import subprocess

RES = "report/results"


def main_table_md():
    """The main 3-seed table from multiseed_summary.md, trimmed for the README / Writeup."""
    lines = open(f"{RES}/multiseed_summary.md", encoding="utf-8").read().split("## ")[1].splitlines()
    tab = [l for l in lines if l.startswith("|")]
    return "\n".join(tab)


def pdf_pages(path="report/ChipStain_Technical_Report.pdf"):
    try:
        from pypdf import PdfReader
        return str(len(PdfReader(path).pages))
    except Exception:  # noqa: BLE001
        out = subprocess.run(["mdls", "-name", "kMDItemNumberOfPages", path], capture_output=True, text=True).stdout
        m = re.search(r"(\d+)", out)
        return m.group(1) if m else "?"


def render(tpl_path, out_path, values):
    s = open(tpl_path, encoding="utf-8").read()
    for _ in range(2):
        for k, v in values.items():
            s = s.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", s)))
    s = re.sub(r"\{\{\w+\}\}", "(pending)", s)  # results still running: never leave raw placeholders
    open(out_path, "w", encoding="utf-8").write(s)
    print("wrote", out_path, "| unfilled:", left or "none")
    return s


def main():
    vals = json.load(open(f"{RES}/key_numbers.json"))
    texts = json.load(open("templates/report_texts.json")) if os.path.exists("templates/report_texts.json") else {}
    vals.update(texts)
    vals["results_table"] = main_table_md()
    vals["report_pages"] = pdf_pages()
    vals.setdefault("report_link", "https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science/blob/main/report/ChipStain_Technical_Report.pdf")
    vals.setdefault("video_link", "[VIDEO LINK — paste the YouTube link here]")
    vals.setdefault("demo_link", "Local demo only (no hosted Space) — see below")
    render("templates/README_template.md", "README.md", vals)
    w = render("templates/kaggle_writeup_template.md", "writeup/kaggle_writeup.md", vals)
    summ = w.split("## Project summary")[1].split("## Technical report")[0]
    print("Project summary words:", len(summ.split()))


if __name__ == "__main__":
    main()
