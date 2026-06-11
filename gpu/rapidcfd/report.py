# -*- coding: utf-8 -*-
"""
Collect RapidCFD forceCoeffs histories and write the wing/VG study report.

Reads gpu/rapidcfd/results/<case>/postProcessing/forceCoeffs1/0/forceCoeffs.dat
(2.3-era format: '# comment' header, then columns Time Cm Cd Cl Cl(f) Cl(r)),
tail-averages the converged window, and prints a delta table of every VG
article against the clean wing at the same alpha.

Run:  python gpu/rapidcfd/report.py [--tail 500]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_coeffs(case: str) -> np.ndarray | None:
    """(N,4) array of [iter, Cm, Cd, Cl] or None when the case has no data."""
    # 2.3 writes one dat file per startTime directory; take them all in
    # time order so restarted runs concatenate cleanly.
    dats = sorted((RESULTS / case / "postProcessing").rglob("forceCoeffs.dat"))
    rows: list[list[float]] = []
    for dat in dats:
        for line in dat.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # Whitespace OR parenthesis separated; keep the leading 4 cols.
            vals = [float(v) for v in re.split(r"[\s()]+", line.strip()) if v]
            if len(vals) >= 4:
                rows.append(vals[:4])
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    return arr[np.argsort(arr[:, 0], kind="stable")]


def tail_stats(arr: np.ndarray, tail: int) -> dict[str, float]:
    """Mean +/- peak-to-peak of the last `tail` iterations (converged window)."""
    win = arr[-min(tail, len(arr)):]
    out: dict[str, float] = {"iters": float(arr[-1, 0]), "n": float(len(win))}
    for j, key in ((1, "Cm"), (2, "Cd"), (3, "Cl")):
        out[key] = float(win[:, j].mean())
        out[key + "_pp"] = float(win[:, j].max() - win[:, j].min())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="RapidCFD VG study report")
    ap.add_argument("--tail", type=int, default=500,
                    help="averaging window, iterations (default 500)")
    a = ap.parse_args()

    cases = sorted(p.name for p in RESULTS.iterdir() if p.is_dir()) \
        if RESULTS.exists() else []
    stats: dict[str, dict[str, float]] = {}
    for c in cases:
        arr = load_coeffs(c)
        if arr is None:
            print(f"[report] {c}: no forceCoeffs data")
            continue
        stats[c] = tail_stats(arr, a.tail)

    if not stats:
        print("[report] nothing to report yet")
        return

    print(f"\n{'case':<16}{'iters':>7}{'Cl':>9}{'pk-pk':>8}{'Cd':>9}"
          f"{'pk-pk':>8}{'Cm':>9}")
    for c, s in stats.items():
        print(f"{c:<16}{s['iters']:>7.0f}{s['Cl']:>9.4f}{s['Cl_pp']:>8.4f}"
              f"{s['Cd']:>9.5f}{s['Cd_pp']:>8.5f}{s['Cm']:>9.4f}")

    # Delta table: each VG article against the clean article at its alpha.
    print("\nVG deltas vs clean (same alpha):")
    for c, s in stats.items():
        m = re.match(r"vg.+_a(\d+)", c)
        if not m:
            continue
        ref = f"clean_a{m.group(1)}"
        if ref not in stats:
            continue
        r = stats[ref]
        dcl, dcd = s["Cl"] - r["Cl"], s["Cd"] - r["Cd"]
        print(f"  {c} vs {ref}:  dCl={dcl:+.4f} ({100 * dcl / max(abs(r['Cl']), 1e-9):+.1f}%)  "
              f"dCd={dcd:+.5f} ({100 * dcd / max(abs(r['Cd']), 1e-9):+.1f}%)")


if __name__ == "__main__":
    main()
