# -*- coding: utf-8 -*-
"""
Generate binary STLs for the FluidX3D virtual wind tunnel: the clean wing
section, and the same wing wearing a row of REAL vortex-generator vanes.

Unlike the RANS pipeline (which models VGs as a jBAY momentum source), the
LBM tunnel voxelizes actual vane geometry, so the vanes here are physical
plates: counter-rotating pairs, placed per the Strausak sweep centers that
live in aircraft.yaml (vg_defaults): row at 7% chord, 50 mm pair pitch,
+/-15 deg incidence to the local flow, vane length l = 3h.

Vane pair convention (STOLspeed-style, viewed from above, flow left->right):
each pair is two plates whose leading edges diverge (toe-out) so the pair
pumps a counter-rotating vortex pair downward between them. Pair pitch is
the aircraft.yaml outboard spacing; the two vanes within a pair sit half a
pitch apart.

Outputs (binary STL, FluidX3D requirement):
    assets/wing_a0_binary.stl        clean wing  (refreshed)
    assets/wing_vg_h{H}mm.stl        wing + VG row, H = vane height in mm

Run:  python gpu/fluidx3d/make_vg_wing.py [--height-mm 10] [--span-m 1.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import trimesh

from geometry.airfoil import load_airfoil, resample_airfoil
from geometry.stl_gen import extrude_section
from geometry.units import load_aircraft

ASSETS = REPO / "gpu" / "fluidx3d" / "assets"


def upper_surface_point(coords: np.ndarray, x_frac: float) -> tuple[float, float]:
    """(y, local slope angle) of the UPPER surface at chordwise station x_frac.

    The resampled loop runs TE-upper -> LE -> TE-lower (Selig order); the
    upper surface is the first half. Interpolated in x, slope from the
    neighboring points - good enough to sit a vane flush on the skin.
    """
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]                     # LE -> TE, ascending x
    y = float(np.interp(x_frac, upper[:, 0], upper[:, 1]))
    i = int(np.searchsorted(upper[:, 0], x_frac))
    i = max(1, min(i, len(upper) - 1))
    dx, dy = (upper[i] - upper[i - 1])
    return y, float(np.arctan2(dy, dx))


def make_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One rectangular vane plate, LE at origin, extending +x, standing +y.

    Plate footprint l x t in the (x,z) plane, height h in +y; the caller
    rotates it to incidence and onto the wing skin. Sharp simple plates
    voxelize cleanly at the lattice scale - aerodynamic refinement of the
    vane profile is far below LBM resolution.
    """
    box = trimesh.creation.box(extents=(length, h, thick))
    box.apply_translation((length / 2.0, h / 2.0, 0.0))
    return box


def build_vg_wing(span_m: float, h_m: float, out_path: Path,
                  pitch_m: float | None = None) -> None:
    """Wing section + counter-rotating VG row, exported as one binary STL.

    pitch_m overrides the aircraft.yaml pair spacing (sweep variable); None
    keeps the Strausak outboard default from vg_defaults.
    """
    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station        # 0.9022 m [DXF]
    x_frac = ac.vg_defaults.wing.chord_position_frac    # 0.07 [IMP74]
    pitch = pitch_m if pitch_m else ac.vg_defaults.wing.spacing_outboard  # 0.050 m [IMP74]
    beta = ac.vg_defaults.vane_incidence                # radians (15 deg)
    l_per_h = ac.vg_defaults.vane_length_per_height     # 3.0 [SPEC]

    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")
    wing = extrude_section(coords, chord, span_m)

    # Vane row geometry on the upper skin at the VG station.
    y_surf, slope = upper_surface_point(coords, x_frac)
    x_le = x_frac * chord
    y_le = y_surf * chord
    vane_l = l_per_h * h_m
    vane_t = max(0.0015, h_m / 8.0)                    # plate thickness

    vanes = []
    n_pairs = int(span_m / pitch)
    z0 = -0.5 * span_m + 0.5 * (span_m - (n_pairs - 1) * pitch)
    for p in range(n_pairs):
        z_pair = z0 + p * pitch
        for sgn in (+1.0, -1.0):                       # toe-out pair: +/- beta
            v = make_vane(h_m, vane_l, vane_t)
            v.apply_transform(trimesh.transformations.rotation_matrix(
                sgn * beta, (0.0, 1.0, 0.0)))          # incidence about vertical
            v.apply_transform(trimesh.transformations.rotation_matrix(
                slope, (0.0, 0.0, 1.0)))               # lay flush on local skin slope
            v.apply_translation((x_le, y_le - 0.10 * h_m,   # sink slightly into skin:
                                 z_pair + sgn * pitch / 4.0))  # guarantees overlap for the union
            vanes.append(v)

    # Boolean union (manifold3d backend) keeps the voxelizer's inside/outside
    # parity sane; plain concatenation of overlapping solids can flip it.
    model = trimesh.boolean.union([wing] + vanes)
    model.export(out_path)
    print(f"wrote {out_path.name}: {len(vanes)} vanes ({n_pairs} pairs), "
          f"h={h_m * 1000:.1f} mm, l={vane_l * 1000:.1f} mm, row at "
          f"{x_frac * 100:.0f}%c, pitch {pitch * 1000:.0f} mm | watertight="
          f"{model.is_watertight} faces={len(model.faces)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FluidX3D wing/VG STL generator")
    ap.add_argument("--height-mm", type=float, default=10.0,
                    help="VG vane height in mm (default 10, STOLspeed-class)")
    ap.add_argument("--span-m", type=float, default=1.5,
                    help="extruded wing span in m (default 1.5)")
    ap.add_argument("--pitch-mm", type=float, default=0.0,
                    help="VG pair spacing in mm (0 = aircraft.yaml default, 50)")
    a = ap.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)

    # Clean wing (refresh the binary export alongside the VG variant).
    ac_coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                                 n_points=241, te="blunt")
    ac = load_aircraft(REPO / "aircraft.yaml")
    clean = extrude_section(ac_coords, ac.wing.aileron.chord_at_mid_station,
                            a.span_m)
    # Span suffix keeps slice-mode (0.25 m) assets from clobbering the 1.5 m
    # wing-mode ones; the legacy names stay for the default 1.5 m span.
    tag = "" if abs(a.span_m - 1.5) < 1e-9 else f"_s{a.span_m:g}m"
    clean_path = ASSETS / f"wing_a0_binary{tag}.stl"
    clean.export(clean_path)
    print(f"wrote {clean_path.name}: clean wing | watertight="
          f"{clean.is_watertight} faces={len(clean.faces)}")

    # Pitch token in the filename only when overridden, so the legacy 50 mm
    # asset names from earlier suites keep working.
    ptag = f"_p{a.pitch_mm:g}mm" if a.pitch_mm > 0.0 else ""
    build_vg_wing(a.span_m, a.height_mm / 1000.0,
                  ASSETS / f"wing_vg_h{a.height_mm:g}mm{ptag}{tag}.stl",
                  pitch_m=a.pitch_mm / 1000.0 if a.pitch_mm > 0.0 else None)


if __name__ == "__main__":
    main()
