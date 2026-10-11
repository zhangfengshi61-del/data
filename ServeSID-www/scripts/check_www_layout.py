#!/usr/bin/env python3
"""WWW 2027 author-side layout audit: 8 content pages, <=12 PDF pages."""
import re
import subprocess
import sys
from pathlib import Path

pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "main.pdf")
if not pdf.is_file():
    sys.exit(f"ERROR: missing PDF: {pdf}")

def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

info = command("pdfinfo", str(pdf))
mt = re.search(r"^Pages:\s*(\d+)", info, re.M)
total = int(mt.group(1)) if mt else 0
pages = command("pdftotext", "-layout", str(pdf), "-").split("\f")
page_text = [p for p in pages if p.strip()]
ref_page = next((i + 1 for i, p in enumerate(page_text)
                 if re.search(r"^\s*REFERENCES\s*$", p, re.I | re.M)), None)
method_page = next((i + 1 for i, p in enumerate(page_text)
                    if "Overview of" in p and ("HServe" in p or "ServeSID" in p)), None)
content_pages = min(total, ref_page - 1) if ref_page else None
ok = ref_page is not None and ref_page <= 9 and total <= 12
lines = [
    "WWW 2027 Research Track layout audit",
    "Official format: acmart sigconf, anonymous, review",
    f"Total PDF pages: {total} (maximum 12)",
    f"References begin on page: {ref_page or 'NOT DETECTED'}",
    f"Content-only full pages before references: {content_pages if content_pages is not None else 'unknown'}",
    f"Main method figure caption appears on page: {method_page or 'NOT DETECTED'}",
    f"8-content-page budget: {'PASS' if ok else 'FAIL'}",
    "NOTE: If references share a page with body text, inspect that page manually.",
    "NOTE: This checks layout, not experimental provenance or scientific correctness.",
]
report = "\n".join(lines) + "\n"
print(report)
pdf.with_name("layout_audit.txt").write_text(report)
if not ok:
    sys.exit(2)
