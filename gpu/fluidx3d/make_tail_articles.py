# -*- coding: utf-8 -*-
"""
Tail control-effectiveness test articles for the FluidX3D tunnel: the
horizontal-stab + elevator and the vertical-fin + rudder sections from the
project's geometry toolkit, packaged exactly like the aileron articles --
one binary STL per case, main element + control surface as two disjoint
closed shells in a single file (ray-parity voxelization handles disjoint
shells cleanly, and the hinge gap must stay OPEN).

Why these exist: same question as the aileron suite, moved to the tail.
Effectiveness = force difference between opposite control deflections at the
same onset angle; what matters at low airspeed is whether the elevator keeps
its nose-up authority in the flare (suction side = UNDERSIDE of the stab)
and whether the rudder keeps authority at low-q / high-sideslip. VG variants
carry real vane geometry per the Strausak elevator convention in
aircraft.yaml (vg_defaults.elevator): counter-rotating pairs on the suction
side, vane LE 100 mm ahead of the hinge line, 30 mm pair pitch.

Articles per surface (clean and optionally VG'd main element):
    stab_clean_elev_n.stl    elevator neutral, hinge gap open
    stab_clean_elev_d15.stl  elevator 15 deg TE-DOWN (positive per stl_gen)
    stab_clean_elev_u15.stl  elevator 15 deg TE-UP   (= NOSE-UP command)
    fin_clean_rud_{n,d15,u15}.stl   same family for the rudder; for the fin
                             the section lies in a horizontal plane and +y is
                             simply "toward +y yaw" -- the symmetric section
                             makes the two signs mirror images, both kept so
                             effectiveness differencing works unchanged.

VG placement notes (be honest about provenance):
  * elevator: vanes on the UNDERSIDE only (the flare suction side), station
    and pitch straight from aircraft.yaml vg_defaults.elevator [IMP74].
  * rudder: Strausak published no rudder row; the rudder swings both ways,
    so VG variants carry the SAME row convention mirrored onto BOTH sides
    of the fin. This is an analog, not flight-test data -- the printout
    flags it on every run.

Section caveats (inherited from the toolkit, flagged there too): both tail
airfoils are NACA 0010 placeholders (actual as-built sections unmeasured),
and the 22.1 deg rudder hinge rake is a 3D planform feature a constant
section cannot carry -- the 2.5D study treats the hinge as section-normal.

Resolution honesty: at coarse lattice cells the 1.59 mm hinge gap is ~1
cell, so its leak jet is NOT resolved; what is measured is the rigid-body
deflection response of the section, same caveat as the aileron articles.

Run:  python gpu/fluidx3d/make_tail_articles.py [--span-m 0.25]
      [--deflection-deg 15] [--vg-height-mm 10] [--vg-pitch-mm 0]
      [--vane-thickness-mm 0] [--surfaces both|stab|fin] [--no-vg]
      [--no-render]
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

import numpy as np
import trimesh

from geometry.airfoil import naca4_coords
from geometry.stl_gen import gen_fin_section_stl, gen_stab_section_stl
from geometry.units import load_aircraft

ASSETS = REPO / "gpu" / "fluidx3d" / "assets"
PREVIEWS = REPO / "gpu" / "fluidx3d" / "results" / "previews"
YAML = REPO / "aircraft.yaml"


# =============================================================================
#  Surface probing + vane construction (side-aware versions of the helpers
#  in make_vg_wing.py, which only ever needed the wing UPPER surface)
# =============================================================================

def surface_point(coords: np.ndarray, x_frac: float,
                  sign_y: float) -> tuple[float, float]:
    """(y, local slope angle) of one surface at chordwise station x_frac.

    sign_y = +1 probes the upper surface, -1 the lower. The Selig loop runs
    TE-upper -> LE -> TE-lower, so the split at the minimum-x point yields
    both surfaces LE -> TE in ascending x; interpolation and a neighboring-
    point slope are plenty to sit a vane flush on the skin.
    """
    le = int(np.argmin(coords[:, 0]))
    surf = coords[:le + 1][::-1] if sign_y > 0 else coords[le:]
    y = float(np.interp(x_frac, surf[:, 0], surf[:, 1]))
    i = int(np.searchsorted(surf[:, 0], x_frac))
    i = max(1, min(i, len(surf) - 1))
    dx, dy = (surf[i] - surf[i - 1])
    return y, float(np.arctan2(dy, dx))


def make_side_vane(h: float, length: float, thick: float,
                   sign_y: float) -> trimesh.Trimesh:
    """One rectangular vane plate, LE at origin, extending +x, standing
    toward sign_y (+1 = up off an upper surface, -1 = down off a lower one).

    Same sharp simple plate as the wing generator: aerodynamic refinement of
    the vane profile is far below LBM resolution, and plain boxes voxelize
    cleanly at the lattice scale.
    """
    box = trimesh.creation.box(extents=(length, h, thick))
    box.apply_translation((length / 2.0, sign_y * h / 2.0, 0.0))
    return box


def vane_row(coords: np.ndarray, chord_m: float, x_row_m: float,
             span_m: float, h_m: float, pitch_m: float, beta_rad: float,
             thick_m: float, sign_y: float) -> list[trimesh.Trimesh]:
    """Counter-rotating VG pair row on one surface of a section.

    Pair convention matches make_vg_wing.py (STOLspeed-style toe-out pairs):
    pair centers at pitch_m spacing, the two vanes of a pair half a pitch
    apart with opposite incidence signs. Each plate is rotated to incidence
    about the surface normal (~y), laid flush on the local skin slope, and
    sunk 10% of its height into the skin so the boolean union always sees
    real overlap.
    """
    y_surf, slope = surface_point(coords, x_row_m / chord_m, sign_y)
    y_row = y_surf * chord_m
    vanes: list[trimesh.Trimesh] = []
    n_pairs = int(span_m / pitch_m)
    # Center the row on midspan (z = 0 convention shared by every article).
    z0 = -0.5 * span_m + 0.5 * (span_m - (n_pairs - 1) * pitch_m)
    for p in range(n_pairs):
        z_pair = z0 + p * pitch_m
        for sgn in (+1.0, -1.0):                       # toe-out pair: +/- beta
            v = make_side_vane(h_m, 3.0 * h_m, thick_m, sign_y)
            v.apply_transform(trimesh.transformations.rotation_matrix(
                sgn * beta_rad, (0.0, 1.0, 0.0)))      # incidence about normal
            v.apply_transform(trimesh.transformations.rotation_matrix(
                slope, (0.0, 0.0, 1.0)))               # flush on local skin slope
            v.apply_translation((x_row_m, y_row - sign_y * 0.10 * h_m,
                                 z_pair + sgn * pitch_m / 4.0))
            vanes.append(v)
    return vanes


# =============================================================================
#  Article assembly — toolkit section + optional VG row(s), one binary STL
# =============================================================================

def tail_article(surface: str, deflection_deg: float, span_m: float,
                 tag: str, vg_height_mm: float = 0.0,
                 vg_pitch_mm: float = 0.0,
                 vane_thick_mm: float = 0.0) -> Path:
    """One tail test article exported to assets/<tag>.stl (binary).

    surface = 'stab' (elevator case) or 'fin' (rudder case). The toolkit
    generator builds the gapped main+control pair; a VG row is unioned onto
    the MAIN element only (vanes never ride the moving surface), then the
    shells are concatenated -- NOT boolean-unioned, because the hinge gap
    must stay open and the shells do not touch by construction.
    """
    ac = load_aircraft(YAML)
    with tempfile.TemporaryDirectory() as td:
        # ---- toolkit section: gapped pair on the shared z convention -------
        if surface == "stab":
            res = gen_stab_section_stl(
                str(YAML), td,
                elevator_deflection_rad=math.radians(deflection_deg),
                span_m=span_m)
            airfoil = ac.horizontal_tail.airfoil
            # Elevator row: suction side in the flare is the UNDERSIDE.
            vg_sides = (-1.0,)
            dist_ahead = ac.vg_defaults.elevator.distance_ahead_of_hinge
        elif surface == "fin":
            res = gen_fin_section_stl(
                str(YAML), td,
                rudder_deflection_rad=math.radians(deflection_deg),
                span_m=span_m)
            airfoil = ac.vertical_tail.airfoil
            # Rudder swings both ways -> mirrored rows on BOTH sides. This
            # reuses the elevator convention (no published rudder row).
            vg_sides = (+1.0, -1.0)
            dist_ahead = ac.vg_defaults.elevator.distance_ahead_of_hinge
        else:
            raise ValueError(f"unknown surface {surface!r}")

        solids = [trimesh.load(p) for p in res.paths]   # [main, control]

        # ---- optional VG row(s) on the fixed element ------------------------
        if vg_height_mm > 0.0:
            # Same NACA loop the generator extruded, for skin probing.
            coords = naca4_coords("0010", n_points=241, te="sharp")
            h = vg_height_mm / 1000.0
            pitch = (vg_pitch_mm / 1000.0 if vg_pitch_mm > 0.0
                     else ac.vg_defaults.elevator.spacing)   # 30 mm [IMP74]
            beta = ac.vg_defaults.vane_incidence              # 15 deg [IMP74]
            thick = (vane_thick_mm / 1000.0 if vane_thick_mm > 0.0
                     else max(0.0015, h / 8.0))
            # Row station: vane LE sits dist_ahead forward of the hinge line.
            x_row = res.hinge_frac * res.chord_m - dist_ahead
            if x_row <= 0.0:
                raise ValueError(
                    f"VG row station {x_row:.4f} m is ahead of the LE; "
                    "check vg_defaults.elevator.distance_ahead_of_hinge")
            vanes: list[trimesh.Trimesh] = []
            for sign_y in vg_sides:
                vanes += vane_row(coords, res.chord_m, x_row, span_m,
                                  h, pitch, beta, thick, sign_y)
            # Boolean union (manifold3d backend) keeps the voxelizer's
            # inside/outside parity sane, same as the wing generators.
            solids[0] = trimesh.boolean.union([solids[0]] + vanes)
            print(f"  vg row: {len(vanes)} vanes on {len(vg_sides)} side(s), "
                  f"h={h * 1000:.1f} mm, l={3 * h * 1000:.1f} mm, t="
                  f"{thick * 1000:.2f} mm, LE at x={x_row * 1000:.0f} mm "
                  f"({x_row / res.chord_m * 100:.1f}%c), pitch "
                  f"{pitch * 1000:.0f} mm")

        # ---- export: two disjoint closed shells in one binary file ----------
        combined = trimesh.util.concatenate(solids)
        out = ASSETS / f"{tag}.stl"
        combined.export(out)                            # .stl -> binary
        shells_ok = all(s.is_watertight for s in solids)
        print(f"wrote {out.name}: {len(solids)} shells, {airfoil} "
              f"chord={res.chord_m:.4f} m, deflection {deflection_deg:+.0f} "
              f"deg, vg={vg_height_mm:g}mm | shells watertight={shells_ok} "
              f"faces={len(combined.faces)}")
        return out


# =============================================================================
#  Preview rendering — offscreen PyVista, three views per article
# =============================================================================

def render_preview(stl_path: Path) -> Path:
    """Render assets/<stem>.stl to results/previews/<stem>.png.

    Three panes: isometric overview, true section profile (shows the open
    hinge gap and the deflection), and a suction-side oblique that makes the
    VG row legible. Offscreen so suite runs never pop windows.
    """
    import pyvista as pv

    mesh = pv.read(str(stl_path))
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2400, 800),
                   border=False)
    for col, view in enumerate(("iso", "profile", "suction")):
        p.subplot(0, col)
        p.add_mesh(mesh, color="#d8dde3", smooth_shading=True,
                   specular=0.4, specular_power=12)
        p.set_background("#1c1e22")
        if view == "iso":
            p.view_isometric()
        elif view == "profile":
            p.view_xy()                      # straight down +z: the section
        else:
            # Oblique from below-front: vane row + gap channel in one look.
            p.view_vector((-0.5, -1.0, 0.6))
        p.camera.zoom(1.25)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    png = PREVIEWS / f"{stl_path.stem}.png"
    p.screenshot(str(png))
    p.close()
    print(f"  preview -> {png.relative_to(REPO)}")
    return png


# =============================================================================
#  CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="tail control-effectiveness articles")
    ap.add_argument("--span-m", type=float, default=0.25,
                    help="spanwise extrusion (0.25 = slice-mode convention)")
    ap.add_argument("--deflection-deg", type=float, default=15.0,
                    help="control deflection magnitude for the u/d articles")
    ap.add_argument("--vg-height-mm", type=float, default=10.0,
                    help="VG vane height for the VG'd variants (0 disables)")
    ap.add_argument("--vg-pitch-mm", type=float, default=0.0,
                    help="VG pair pitch (0 = aircraft.yaml elevator default, 30)")
    ap.add_argument("--vane-thickness-mm", type=float, default=0.0,
                    help="override plate thickness (0 = physical; use >= "
                         "1.5x cell size for coarse visual lattices)")
    ap.add_argument("--surfaces", choices=("both", "stab", "fin"),
                    default="both")
    ap.add_argument("--no-vg", action="store_true",
                    help="clean articles only (inspection pass)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the PyVista preview PNGs")
    a = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    surfaces = ("stab", "fin") if a.surfaces == "both" else (a.surfaces,)
    ctrl_tag = {"stab": "elev", "fin": "rud"}
    d = a.deflection_deg
    span_tag = f"_s{a.span_m:g}m"
    written: list[Path] = []

    for surf in surfaces:
        # Clean family: neutral + both deflections, gap open throughout.
        # (TE-up = negative deflection = NOSE-UP elevator command.)
        for defl, dtag in ((0.0, "n"), (+d, f"d{d:g}"), (-d, f"u{d:g}")):
            written.append(tail_article(
                surf, defl, a.span_m,
                f"{surf}_clean_{ctrl_tag[surf]}_{dtag}{span_tag}"))
        # VG'd family: same deflections, vane row(s) on the main element.
        if not a.no_vg and a.vg_height_mm > 0.0:
            pitch_mm = a.vg_pitch_mm if a.vg_pitch_mm > 0.0 else 30.0
            ttag = (f"_t{a.vane_thickness_mm:g}mm"
                    if a.vane_thickness_mm > 0.0 else "")
            for defl, dtag in ((0.0, "n"), (+d, f"d{d:g}"), (-d, f"u{d:g}")):
                written.append(tail_article(
                    surf, defl, a.span_m,
                    f"{surf}_vg{a.vg_height_mm:g}p{pitch_mm:g}{ttag}_"
                    f"{ctrl_tag[surf]}_{dtag}{span_tag}",
                    vg_height_mm=a.vg_height_mm,
                    vg_pitch_mm=a.vg_pitch_mm,
                    vane_thick_mm=a.vane_thickness_mm))

    if not a.no_render:
        for stl in written:
            render_preview(stl)


if __name__ == "__main__":
    main()
