# -*- coding: utf-8 -*-
"""Rasterize NASA report pages for manual curve digitization (M1 support tool).

The Phase-1 Clmax/stall gate (validation/compare_gate.py, gate b) compares the
RANS polar against hand-digitized NASA wind-tunnel curves. The source scans in
validation/nasa/ are 1970s microfiche-quality PDFs with no vector content, so
digitization is a VISUAL process: render the figure page at high dpi, calibrate
the axes from the printed tick labels, and read off (alpha, cl) per plotted
symbol against the fine grid. This script is the rendering half of that loop -
poppler is not installed on the workstation, so PyMuPDF does the rasterizing.

Typical session (TM X-72843, figure 5, the LS(1)-0413 section characteristics):

    # whole page, get bearings
    python scripts/digitize_helper.py validation/nasa/NASA_TMX72843_thickness_effects_GAW_family.pdf \
        --page 27 --dpi 200 -o results/digitize_scratch/fig5_full.png

    # zoom the lift-curve quadrant for symbol reading (crop is normalized 0-1,
    # x0,y0,x1,y1 with the origin at the TOP-LEFT of the page)
    python scripts/digitize_helper.py validation/nasa/NASA_TMX72843_thickness_effects_GAW_family.pdf \
        --page 27 --dpi 400 --crop 0.05,0.25,0.60,0.60 -o results/digitize_scratch/fig5_zoom.png

Outputs land wherever -o points; results/ is gitignored apart from summaries,
which makes results/digitize_scratch/ the natural scratch pad. The digitized
CSVs themselves are committed under validation/nasa/digitized/ with provenance
headers (see validation/nasa/SOURCES.md for the page/figure map and the
uncertainty bookkeeping).

Dependency note: requires PyMuPDF ("pip install pymupdf"). It is deliberately
NOT in requirements.txt - digitization is a one-off manual task, not part of
the mesh/run/post pipeline, and the pipeline must stay installable without it.

CLI: python scripts/digitize_helper.py PDF --page N [--dpi 300]
         [--crop x0,y0,x1,y1] [--rotate {0,90,180,270}] -o OUT.png
ASCII-only console output; exit 0 on success, 2 on bad arguments/missing deps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Not strictly needed today (no repo-package imports below), but every script
#  in scripts/ carries the same prologue so adding a geometry/units import
#  later cannot silently break depending on the caller's working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _parse_crop(text: str) -> Tuple[float, float, float, float]:
    """Parse and validate an 'x0,y0,x1,y1' normalized crop rectangle.

    Coordinates are fractions of the page width/height with the origin at the
    TOP-LEFT corner (the PyMuPDF page convention), so y grows downward. The
    rectangle must be non-degenerate and inside [0, 1] on both axes.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "crop must be four comma-separated numbers: x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"non-numeric crop value: {exc}")
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise argparse.ArgumentTypeError(
            "crop must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def render_page(pdf_path: Path, page_1based: int, dpi: int,
                crop: Optional[Tuple[float, float, float, float]],
                rotate: int, out_path: Path) -> Tuple[int, int]:
    """Render one PDF page (optionally cropped/rotated) to a PNG file.

    Returns the (width, height) of the written pixmap so the caller can echo
    it - the digitizer needs the pixel size to estimate reading uncertainty
    (one pixel at the rendered scale maps to a known fraction of an axis
    division, recorded in the CSV provenance headers).
    """
    # PyMuPDF import is deferred so --help works without the dependency and
    # the failure message says exactly what to install.
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF is required (pip install pymupdf).")
        raise SystemExit(2)

    doc = fitz.open(str(pdf_path))
    if not (1 <= page_1based <= doc.page_count):
        print(f"ERROR: page {page_1based} out of range "
              f"(document has {doc.page_count} pages).")
        raise SystemExit(2)
    page = doc[page_1based - 1]

    # Optional rotation: applied at the page level so landscape-mounted
    # figures (e.g. TM X-72843 figure 12, printed sideways) come out upright.
    if rotate:
        page.set_rotation((page.rotation + rotate) % 360)

    # Optional crop: normalized rectangle scaled onto the (possibly rotated)
    # page box, passed as the pixmap clip so only the region of interest is
    # rasterized - keeps 400+ dpi zooms small enough to inspect comfortably.
    clip = None
    if crop is not None:
        rect = page.rect  # rotation-aware page box in PDF points
        x0, y0, x1, y1 = crop
        clip = fitz.Rect(rect.x0 + x0 * rect.width,
                         rect.y0 + y0 * rect.height,
                         rect.x0 + x1 * rect.width,
                         rect.y0 + y1 * rect.height)

    pix = page.get_pixmap(dpi=dpi, clip=clip)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return pix.width, pix.height


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse arguments, render, echo what was written."""
    parser = argparse.ArgumentParser(
        description="Render a PDF page region to PNG for visual curve "
                    "digitization (PyMuPDF backend; poppler not required).")
    parser.add_argument("pdf", type=Path, help="source PDF (validation/nasa/...)")
    parser.add_argument("--page", type=int, required=True,
                        help="1-based page number to render")
    parser.add_argument("--dpi", type=int, default=300,
                        help="raster resolution (default 300)")
    parser.add_argument("--crop", type=_parse_crop, default=None,
                        metavar="X0,Y0,X1,Y1",
                        help="normalized crop rectangle, origin top-left")
    parser.add_argument("--rotate", type=int, default=0,
                        choices=(0, 90, 180, 270),
                        help="clockwise rotation applied before cropping")
    parser.add_argument("-o", "--out", type=Path, required=True,
                        help="output PNG path (parent dirs created)")
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        print(f"ERROR: PDF not found: {args.pdf}")
        return 2

    w, h = render_page(args.pdf, args.page, args.dpi, args.crop,
                       args.rotate, args.out)
    # Echo the geometry the digitizer needs for the uncertainty estimate.
    print(f"wrote {args.out} ({w} x {h} px at {args.dpi} dpi, "
          f"page {args.page}"
          + (f", crop {args.crop}" if args.crop else "")
          + (f", rotate {args.rotate}" if args.rotate else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
