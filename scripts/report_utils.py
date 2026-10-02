"""Helpers for scripts/build_report.py: load result files, convert markdown tables, embed figures."""
import base64
import html
import os
import re

import numpy as np
import pandas as pd

RES = "report/results"
FIG = "report/figures"


def b64(name):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return None
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def figure(name, caption):
    src = b64(name)
    if src is None:
        return f"<p class='small'><i>[figure {html.escape(name)} not available]</i></p>"
    return f"<figure><img src='{src}'><figcaption>{caption}</figcaption></figure>"


def md_tables(path):
    """Return the pipe tables of a markdown file as a list of (preceding_heading, rows)."""
    if not os.path.exists(path):
        return []
    out, cur, head = [], [], ""
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            head = line.lstrip("#").strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                cur.append(cells)
        elif cur:
            out.append((head, cur)); cur = []
    if cur:
        out.append((head, cur))
    return out


def html_table(rows, highlight=None, num_cols_from=1):
    """rows[0] = header. highlight: substring of first cell to highlight."""
    h = "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in rows[0]) + "</tr>"
    for r in rows[1:]:
        cls = " class='hl'" if highlight and highlight in r[0] else ""
        h += f"<tr{cls}>" + "".join(f"<td{' class=n' if i >= num_cols_from else ''}>{html.escape(c).replace('`', '')}</td>" for i, c in enumerate(r)) + "</tr>"
    return h + "</table>"


def table_from(path, heading_contains, **kw):
    for head, rows in md_tables(path):
        if heading_contains.lower() in head.lower():
            return html_table(rows, **kw)
    tabs = md_tables(path)
    return html_table(tabs[0][1], **kw) if (tabs and not heading_contains) else "<p class='small'><i>[table not available]</i></p>"


def read(name, **kw):
    p = os.path.join(RES, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def lines(name):
    p = os.path.join(RES, name)
    return open(p, encoding="utf-8").read().splitlines() if os.path.exists(p) else []


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
