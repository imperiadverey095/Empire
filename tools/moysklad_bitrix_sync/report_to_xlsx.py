"""Turn a daily reconciliation report (.md) into an .xlsx for Google Sheets.

Usage: python3 report_to_xlsx.py <report.md> [<report.md> ...] -o <out.xlsx>

One report gives three tabs: «Заказы» (the order table), «Итоги» (every other
section as text) and «Проверить вручную». Several reports go into one workbook
with the same three tabs and a «Дата сверки» column.
"""
import argparse
import re
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

MONEY_COLS = {"Сумма", "Оплачено", "Отгружено"}
HEADER_FILL = PatternFill("solid", fgColor="DDE7F3")


def parse_money(text):
    """'81 600 ₽' -> 81600.0, '248 070,60 ₽' -> 248070.6; other text stays."""
    t = text.replace("₽", "").replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return text


def parse_report(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    m = re.search(r"(\d{2}\.\d{2}\.\d{4})", lines[0] if lines else "")
    date = m.group(1) if m else Path(path).stem
    sections, current = [], None
    for line in lines[1:]:
        if line.startswith("## "):
            current = [line[3:].strip(), []]
            sections.append(current)
        elif current is not None:
            current[1].append(line)
        elif line.strip():
            sections.append(["Прогон", [line]])
    header, rows = None, []
    for name, body in sections:
        table = [l for l in body if l.startswith("|")]
        if name == "Что проверено" and table:
            header = [c.strip() for c in table[0].strip("|").split("|")]
            for l in table[2:]:
                rows.append([c.strip() for c in l.strip("|").split("|")])
    return date, header, rows, sections


def clean(text):
    return text.replace("**", "").strip()


def build(paths, out):
    wb = Workbook()
    ws_o = wb.active
    ws_o.title = "Заказы"
    ws_s = wb.create_sheet("Итоги")
    ws_c = wb.create_sheet("Проверить вручную")
    multi = len(paths) > 1
    o_head = None
    s_rows, c_rows, o_rows = [], [], []
    for p in paths:
        date, header, rows, sections = parse_report(p)
        if header:
            head = ["Заказ", "Дата заказа"] + header[1:]
            if o_head is None or len(head) > len(o_head):
                o_head = head
            for r in rows:
                m = re.match(r"(\S+)\s*\((.+)\)", r[0])
                num, odate = (m.group(1), m.group(2)) if m else (r[0], "")
                vals = [num, odate]
                for h, v in zip(header[1:], r[1:]):
                    vals.append(parse_money(v) if h in MONEY_COLS else clean(v))
                o_rows.append(([date] if multi else []) + [vals, head])
        for name, body in sections:
            items = [clean(l.lstrip("-").strip()) for l in body
                     if l.strip() and not l.startswith("|")]
            if name == "Проверить вручную":
                c_rows += [([date] if multi else []) + [i] for i in items]
            else:
                s_rows += [([date] if multi else []) + [name, i] for i in items]

    pre = ["Дата сверки"] if multi else []
    o_head = o_head or ["Заказ"]
    ws_o.append(pre + o_head)
    for row in o_rows:
        *d, vals, head = row
        by = dict(zip(head, vals))
        ws_o.append(d + [by.get(h, "") for h in o_head])
    ws_s.append(pre + ["Раздел", "Текст"])
    for r in s_rows:
        ws_s.append(r)
    ws_c.append(pre + ["Что проверить", "Ответственный", "Сделано"])
    for r in c_rows:
        ws_c.append(r + ["", ""])

    for ws in (ws_o, ws_s, ws_c):
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = HEADER_FILL
        ws.freeze_panes = "A2"
        for i, col in enumerate(ws.columns, 1):
            width = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 90)
            for c in col[1:]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    for col in ws_o.iter_cols(min_row=1, max_row=1):
        if col[0].value in MONEY_COLS:
            for c in ws_o[col[0].column_letter][1:]:
                if isinstance(c.value, float):
                    c.number_format = '#,##0.00 "₽"' if c.value % 1 else '#,##0 "₽"'
    ws_o.auto_filter.ref = ws_o.dimensions
    wb.save(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    build(a.reports, a.out)
    print(a.out)


if __name__ == "__main__":
    sys.exit(main())
