#!/usr/bin/env python3
"""WWW 2027 layout audit: 8 content pages and <=12 total pages."""
import re
import sys
from pathlib import Path
import fitz

pdf = Path(sys.argv[1] if len(sys.argv)>1 else "main.pdf")
with fitz.open(pdf) as doc:
    total = len(doc)
    txt = [page.get_text(sort=True) for page in doc]
    ref_page = next((i+1 for i,t in enumerate(txt)
                     if re.search(r"\b(?:REFERENCES|References)\b",t)),None)
    method_page = next((i+1 for i,t in enumerate(txt)
                        if re.search(r"Overview of\s+(?:ServeSID|S\s*erveSID)",t,re.I)
                        or ("single-depth attribution" in t and "HServe" in t and "DA-CDRS" in t)),None)
    aux=pdf.with_suffix(".aux")
    body_end=None
    if aux.is_file():
        for line in aux.read_text(errors="replace").splitlines():
            if "newlabel{www:body-end}" in line:
                mat = re.search(r"}{(\d+)}", line)
                if mat:
                    body_end = int(mat.group(1))
                break
    pass_body = body_end is not None and body_end <= 8
    table_pages = {}
    for k in (1, 2):
        table_pages[k] = next((i+1 for i, t in enumerate(txt)
                               if f"Table {k}:" in t), None)
    figure_pages = {}
    for k in range(1, 6):
        figure_pages[k] = next((i+1 for i, t in enumerate(txt)
                                if f"Figure {k}:" in t), None)
    all_floats_in_first_8 = all(p is not None and p <= 8 for p in list(table_pages.values()) + list(figure_pages.values()))
    references_follow_tables = ref_page is not None and all(p is not None and p < ref_page for p in table_pages.values())
    ok = pass_body and total <= 12 and all_floats_in_first_8 and references_follow_tables
    out = [
        "WWW 2027 Research Track layout audit",
        "Source: official www2027.thewebconf.org/research-track-papers/",
        f"Total PDF pages: {total} (maximum 12)",
        f"References start on page: {ref_page or 'NOT DETECTED'}",
        f"Body last page via LaTeX source label: {body_end or 'NOT DETECTED'}",
        f"All non-reference body content <= page 8: {'PASS' if pass_body else 'FAIL'}",
        f"Method overview caption detected on page: {method_page or 'NOT DETECTED'}",
        f"Table caption pages: {table_pages}",
        f"Figure caption pages: {figure_pages}",
        f"References appear after all main tables: {'PASS' if references_follow_tables else 'FAIL'}",
        f"All five figures and both tables fit in first 8 pages: {'PASS' if all_floats_in_first_8 else 'FAIL'}",
        f"Total page limit: {'PASS' if total<=12 else 'FAIL'}",
        f"OVERALL: {'PASS' if ok else 'FAIL'}",
        "Check first eight pages visually; this does not validate experimental numbers.",
    ]
    report="\n".join(out)+"\n"
    print(report)
    (pdf.parent/"layout_audit.txt").write_text(report)
    render=pdf.parent/"layout_previews"
    render.mkdir(exist_ok=True)
    for idx in [0,3,6,7]:
        if idx<total:
            pix=doc[idx].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
            pix.save(render/f"page-{idx+1}.png")
if not ok:
    sys.exit(2)
