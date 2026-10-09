#!/usr/bin/env python3
"""Render the Markdown method description to the required submission PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DOC_DIR = Path(__file__).resolve().parents[2] / "doc"
SOURCE = DOC_DIR / "method_description.md"
OUTPUT = DOC_DIR / "method_description.pdf"
CHINESE_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text).replace("`", "")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _table_rows(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    cursor = start
    while cursor < len(lines) and lines[cursor].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[cursor].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        cursor += 1
    return rows, cursor


def main() -> None:
    if not CHINESE_FONT.is_file():
        raise FileNotFoundError(
            "缺少中文字体。请安装 fonts-wqy-zenhei 后重新生成 PDF。"
        )
    pdfmetrics.registerFont(TTFont("WenQuanYi", str(CHINESE_FONT), subfontIndex=0))
    styles = getSampleStyleSheet()
    for style_name in ("Title", "Heading1", "Heading2", "Heading3", "BodyText", "Normal"):
        styles[style_name].fontName = "WenQuanYi"
    styles.add(ParagraphStyle(name="CenteredTitle", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(
        ParagraphStyle(
            name="ChineseTable",
            parent=styles["BodyText"],
            fontName="WenQuanYi",
            fontSize=7.5,
            leading=10,
        )
    )
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="反应条件优化技术方案",
    )
    story = []
    in_code = False
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    cursor = 0
    while cursor < len(lines):
        raw_line = lines[cursor]
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            cursor += 1
            continue
        if not line:
            story.append(Spacer(1, 3 * mm))
            cursor += 1
            continue
        if not in_code and line.startswith("|"):
            rows, cursor = _table_rows(lines, cursor)
            if rows:
                table_data = [
                    [Paragraph(_inline_markdown(cell), styles["ChineseTable"]) for cell in row]
                    for row in rows
                ]
                if len(rows[0]) == 2:
                    column_widths = [38 * mm, 132 * mm]
                else:
                    column_widths = [34 * mm, 34 * mm, 34 * mm, 28 * mm, 40 * mm]
                table = Table(table_data, colWidths=column_widths, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECEF")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                            ("FONTNAME", (0, 0), (-1, -1), "WenQuanYi"),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.extend([table, Spacer(1, 3 * mm)])
            continue
        if in_code:
            story.append(
                Paragraph(
                    f"<font name='WenQuanYi'>{html.escape(line)}</font>",
                    styles["Code"],
                )
            )
        elif line.startswith("# "):
            story.append(Paragraph(_inline_markdown(line[2:]), styles["CenteredTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline_markdown(line[3:]), styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(_inline_markdown(line[4:]), styles["Heading3"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"&bull; {_inline_markdown(line[2:])}", styles["BodyText"]))
        else:
            story.append(Paragraph(_inline_markdown(line), styles["BodyText"]))
        cursor += 1
    document.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
