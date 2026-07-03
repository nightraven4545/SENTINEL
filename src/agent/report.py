"""Render the risk memo to reports/ as Markdown and PDF.

The PDF renderer is intentionally simple: it understands the memo's own
markdown subset (#/## headings, tables, paragraphs) — not general markdown.
"""
# ponytail: minimal markdown->PDF; upgrade path is weasyprint/pandoc if
# richer formatting is ever needed.
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
MD_FILE = REPORTS_DIR / "risk_memo.md"
PDF_FILE = REPORTS_DIR / "risk_memo.pdf"


def save_markdown(memo: str, path: Path = MD_FILE) -> Path:
    path.parent.mkdir(exist_ok=True)
    path.write_text(memo, encoding="utf-8")
    return path


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_pdf(memo: str, path: Path = PDF_FILE) -> Path:
    path.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    flow = []
    table_rows: list[list[str]] = []

    def flush_table():
        if not table_rows:
            return
        t = Table(table_rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12161F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00E5A0")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        flow.extend([t, Spacer(1, 0.4 * cm)])
        table_rows.clear()

    for line in memo.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= {"-", ":", " "} for c in cells):  # skip |---|
                table_rows.append(cells)
            continue
        flush_table()
        if not line:
            flow.append(Spacer(1, 0.25 * cm))
        elif line.startswith("## "):
            flow.append(Paragraph(_escape(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            flow.append(Paragraph(_escape(line[2:]), styles["Title"]))
        else:
            flow.append(Paragraph(_escape(line), styles["BodyText"]))
    flush_table()

    SimpleDocTemplate(str(path), pagesize=A4).build(flow)
    return path
