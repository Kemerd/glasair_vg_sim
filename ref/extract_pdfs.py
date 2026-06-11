# -*- coding: utf-8 -*-
"""
PDF -> plain-text extractor for the reference documents.

Extracts every page of the given PDFs into sibling .txt files with
per-page markers so the source page of any figure/table reference can
be located later. Output is UTF-8; console prints are ASCII-safe so
this runs cleanly in a default Windows console.
"""
import sys
from pathlib import Path

from pypdf import PdfReader

# Reference documents to convert (relative to this script's directory)
DOCS = [
    "Impulse74_Translated.pdf",
    "virginia_tech_GlassairIII.pdf",
]


def extract(pdf_path: Path) -> Path:
    """Extract all text from pdf_path into a UTF-8 .txt with page markers."""
    reader = PdfReader(str(pdf_path))
    out_path = pdf_path.with_suffix(".txt")

    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        # Page marker keeps figure/table references traceable to the PDF
        chunks.append(f"\n========== PAGE {i} of {len(reader.pages)} ==========\n")
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - report and continue per page
            text = f"[EXTRACTION ERROR on page {i}: {exc}]"
        chunks.append(text)

    out_path.write_text("".join(chunks), encoding="utf-8")
    print(f"{pdf_path.name}: {len(reader.pages)} pages -> {out_path.name} "
          f"({out_path.stat().st_size} bytes)")
    return out_path


def main() -> int:
    base = Path(__file__).parent
    ok = True
    for name in DOCS:
        pdf = base / name
        if not pdf.exists():
            print(f"MISSING: {pdf}")
            ok = False
            continue
        extract(pdf)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
