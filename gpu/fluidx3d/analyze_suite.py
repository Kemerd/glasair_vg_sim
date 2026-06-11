# -*- coding: utf-8 -*-
"""
Standalone analyzer for the VG comparison suite: builds REPORT.md directly
from whatever per-case CSVs exist in results/suite/, independent of each
case's exit code (the tunnel exe historically crashed in TEARDOWN after all
data was flushed, so exit codes understate success; the CSV content is the
ground truth).

Verdict logic per speed block, against the clean baseline at the same speed:
  * dCl > +noise  -> the VG row is holding extra lift at the near-stall angle
  * stall-speed projection (indicative only -- single-AoA proxy, NOT a
    measured CLmax ratio):  Vs_new ~ Vs * sqrt(Cl_clean / Cl_vg)
  * dCd            -> the parasite price of the vane row
  * buffet pk-pk   -> oscillation amplitude (separation activity indicator)

Run any time (mid-suite or after):  python gpu/fluidx3d/analyze_suite.py
"""
from __future__ import annotations

import csv as csv_mod
import math
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent / "results" / "suite"

SPEEDS = [("80mph", 80.0), ("60mph", 60.0)]
DESIGNS = ["clean", "vg08mm", "vg12mm", "vg16mm"]
SETTLE_FRAC = 0.4          # discard the first 40% of samples (startup transient)
MIN_ROWS = 20              # below this the case is treated as incomplete


def settled_stats(case_csv: Path):
    """(mean Cl, mean Cd, Cl pk-pk, n) over the post-transient window."""
    rows = []
    with open(case_csv, encoding="utf-8") as fh:
        for r in csv_mod.reader(ln for ln in fh if not ln.startswith("#")):
            if len(r) >= 6:
                rows.append((float(r[0]), float(r[4]), float(r[5])))
    if len(rows) < MIN_ROWS:
        return None
    cut = rows[int(SETTLE_FRAC * len(rows)):]
    n = len(cut)
    cl = sum(r[2] for r in cut) / n
    cd = sum(r[1] for r in cut) / n
    pp = max(r[2] for r in cut) - min(r[2] for r in cut)
    return cl, cd, pp, n


def main() -> int:
    lines = ["# VG comparison suite -- analysis from per-case force CSVs", "",
             "Slice mode, 14 deg AoA, settled-window averages "
             f"(first {int(SETTLE_FRAC * 100)}% of samples discarded).",
             "Trust DELTAS against the clean baseline; absolute values carry",
             "shared coarse-lattice bias. Stall-speed projections are an",
             "indicative single-AoA proxy, not a measured CLmax ratio.", ""]
    any_block = False
    for sp_label, mph in SPEEDS:
        stats = {}
        for d in DESIGNS:
            p = SUITE / f"{sp_label}_{d}.csv"
            if p.exists():
                s = settled_stats(p)
                if s:
                    stats[d] = s
        if "clean" not in stats:
            lines += [f"## {sp_label}: no complete clean baseline yet", ""]
            continue
        any_block = True
        cl0, cd0, pp0, n0 = stats["clean"]
        lines += [f"## {sp_label} (clean baseline: Cl={cl0:.4f} Cd={cd0:.4f} "
                  f"buffet={pp0:.4f}, {n0} samples)", "",
                  "| design | mean Cl | dCl | dCl % | proj. stall speed | mean Cd | dCd | buffet |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in DESIGNS:
            if d not in stats:
                lines.append(f"| {d} | (incomplete) | | | | | | |")
                continue
            cl, cd, pp, _ = stats[d]
            if d == "clean":
                lines.append(f"| clean | {cl:.4f} | -- | -- | {mph:.0f} mph (ref) "
                             f"| {cd:.4f} | -- | {pp:.4f} |")
            else:
                vs = mph * math.sqrt(max(1e-9, cl0) / max(1e-9, cl))
                lines.append(f"| {d} | {cl:.4f} | {cl - cl0:+.4f} | "
                             f"{(cl - cl0) / abs(cl0) * 100.0:+.1f}% | "
                             f"~{vs:.1f} mph | {cd:.4f} | {cd - cd0:+.4f} | {pp:.4f} |")
        lines.append("")
    out = SUITE / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")
    return 0 if any_block else 1


if __name__ == "__main__":
    sys.exit(main())
