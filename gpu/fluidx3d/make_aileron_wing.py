# -*- coding: utf-8 -*-
"""
Aileron-effectiveness test articles for the FluidX3D tunnel: the real
deflected-aileron sections from the project's geometry toolkit, packaged as
single binary STLs (main element + aileron as two disjoint closed shells in
one file -- ray-parity voxelization handles disjoint shells cleanly).

Why these exist: the owner's actual question is CONTROL AUTHORITY, not lift.
Effectiveness = force difference between aileron-down and aileron-up at the
same angle of attack; a wing whose effectiveness survives high alpha is the
Strausak claim, measured the way the pilot feels it.

Articles per design (clean and optionally VG'd main element):
    *_ail_n.stl    aileron neutral, hinge gap open
    *_ail_d15.stl  aileron 15 deg DOWN (TE down = positive per stl_gen)
    *_ail_u15.stl  aileron 15 deg UP

Resolution honesty (also in the analysis): at ~1.6 mm lattice cells the
1.59 mm hinge gap is ~1 cell -- its leak jet is NOT resolved; what is
measured is the rigid-body deflection response of the section.

Run:  python gpu/fluidx3d/make_aileron_wing.py [--span-m 0.25]
      [--vg-height-mm 16] [--vg-pitch-mm 50]
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import trimesh

from geometry.stl_gen import gen_wing_section_stl
from geometry.units import load_aircraft

sys.path.insert(0, str(REPO / "gpu" / "fluidx3d"))
from make_vg_wing import build_vg_wing  # reused for the VG'd main element check

ASSETS = REPO / "gpu" / "fluidx3d" / "assets"
YAML = REPO / "aircraft.yaml"


def deflected_article(deflection_deg: float, span_m: float, tag: str,
                      vg_height_mm: float = 0.0,
                      vg_pitch_mm: float = 50.0,
                      vane_thick_mm: float = 0.0) -> Path:
    """One test article: toolkit-generated section (+gap), optional VG row
    unioned onto the MAIN element only, exported as a single binary STL."""
    with tempfile.TemporaryDirectory() as td:
        res = gen_wing_section_stl(
            str(YAML), td,
            aileron_deflection_rad=math.radians(deflection_deg),
            span_m=span_m, include_gap=True)
        solids = [trimesh.load(p) for p in res.paths]   # [main, control]
        # The main element is solids[0] by the stl_gen naming contract; a VG
        # row only ever attaches to the fixed element, never the aileron.
        if vg_height_mm > 0.0:
            vg_tmp = Path(td) / "vg_main.stl"
            # build_vg_wing unions vanes onto a fresh plain extrusion -- for
            # the split section we union the same vane solids onto the main
            # element instead: rebuild vanes via build_vg_wing's own pass on
            # a throwaway wing, then steal the vane geometry by difference.
            # Simpler and robust: union vanes directly using its helpers.
            from make_vg_wing import make_vane, upper_surface_point
            from geometry.airfoil import load_airfoil, resample_airfoil
            ac = load_aircraft(YAML)
            chord = ac.wing.aileron.chord_at_mid_station
            coords = resample_airfoil(
                load_airfoil(REPO / "geometry" / "ls413.dat"), 241, "blunt")
            y_surf, slope = upper_surface_point(
                coords, ac.vg_defaults.wing.chord_position_frac)
            x_le = ac.vg_defaults.wing.chord_position_frac * chord
            y_le = y_surf * chord
            h = vg_height_mm / 1000.0
            pitch = vg_pitch_mm / 1000.0
            beta = ac.vg_defaults.vane_incidence
            vanes = []
            n_pairs = int(span_m / pitch)
            z0 = -0.5 * span_m + 0.5 * (span_m - (n_pairs - 1) * pitch)
            vane_t = (vane_thick_mm / 1000.0 if vane_thick_mm > 0.0
                      else max(0.0015, h / 8.0))
            for p_i in range(n_pairs):
                for sgn in (+1.0, -1.0):
                    v = make_vane(h, 3.0 * h, vane_t)
                    v.apply_transform(trimesh.transformations.rotation_matrix(
                        sgn * beta, (0, 1, 0)))
                    v.apply_transform(trimesh.transformations.rotation_matrix(
                        slope, (0, 0, 1)))
                    v.apply_translation((x_le, y_le - 0.10 * h,
                                         z0 + p_i * pitch + sgn * pitch / 4.0))
                    vanes.append(v)
            solids[0] = trimesh.boolean.union([solids[0]] + vanes)
        # Disjoint shells -> one file. NOT a boolean union: the hinge gap must
        # stay open, and the shells do not touch by construction.
        combined = trimesh.util.concatenate(solids)
        out = ASSETS / f"{tag}.stl"
        combined.export(out)
        gap_ok = all(s.is_watertight for s in solids)
        print(f"wrote {out.name}: {len(solids)} shells, deflection "
              f"{deflection_deg:+.0f} deg, vg={vg_height_mm:g}mm | "
              f"shells watertight={gap_ok} faces={len(combined.faces)}")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="aileron-effectiveness articles")
    ap.add_argument("--span-m", type=float, default=0.25)
    ap.add_argument("--vg-height-mm", type=float, default=16.0,
                    help="VG height for the VG'd variants (16 = the night's whisper candidate)")
    ap.add_argument("--vg-pitch-mm", type=float, default=50.0)
    ap.add_argument("--vane-thickness-mm", type=float, default=0.0,
                    help="override vane plate thickness (0 = physical; use "
                         ">= 1.5x cell size for coarse visual lattices)")
    a = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    tagb = f"_s{a.span_m:g}m"
    for defl, dtag in ((0.0, "n"), (+15.0, "d15"), (-15.0, "u15")):
        deflected_article(defl, a.span_m, f"wing_clean_ail_{dtag}{tagb}")
    ttag = f"_t{a.vane_thickness_mm:g}mm" if a.vane_thickness_mm > 0.0 else ""
    for defl, dtag in ((0.0, "n"), (+15.0, "d15"), (-15.0, "u15")):
        deflected_article(defl, a.span_m,
                          f"wing_vg{a.vg_height_mm:g}p{a.vg_pitch_mm:g}{ttag}_ail_{dtag}{tagb}",
                          vg_height_mm=a.vg_height_mm,
                          vg_pitch_mm=a.vg_pitch_mm,
                          vane_thick_mm=a.vane_thickness_mm)


if __name__ == "__main__":
    main()
