# -*- coding: utf-8 -*-
"""
Print the two tunnel-config placement constants for a VG row at ANY
chordwise station -- the only part of a VG sweep that needs airfoil data.

    python gpu/fluidx3d/vg_station.py 0.05      (row at 5% chord)
    python gpu/fluidx3d/vg_station.py 0.10      (row at 10% chord)

Paste the printed vg_skin_off_frac / vg_le_off_frac lines into
tunnel_config.txt (they replace the 7%-chord defaults baked into setup.cpp)
and relaunch -- no STL, no rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from geometry.airfoil import load_airfoil, resample_airfoil


def main() -> None:
    x_frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.07
    coords = resample_airfoil(
        load_airfoil(REPO / "geometry" / "ls413.dat"), 241, "blunt")
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]                 # LE -> TE, ascending x
    y_surf = float(np.interp(x_frac, upper[:, 0], upper[:, 1]))
    # Constants are relative to the mesh BOUNDING-BOX center because
    # read_stl centers the wing by bbox (see the stamper in setup.cpp).
    y_c = 0.5 * (coords[:, 1].min() + coords[:, 1].max())
    x_c = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
    print(f"# VG row at {x_frac*100:.1f}% chord (upper skin y = {y_surf:.5f} c):")
    print(f"vg_skin_off_frac = {y_surf - y_c:.5f}")
    print(f"vg_le_off_frac   = {x_frac - x_c:.5f}")


if __name__ == "__main__":
    main()
