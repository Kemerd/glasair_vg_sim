# -*- coding: utf-8 -*-
"""
Tail-surface VG study (RudderFIRST, then elevator) on the user's printed v3 vane.

Closes the loop on the OWNER's tail question: VGs mounted ~100 mm ahead of the
control-surface hinge (per IMP74 / "Impulse 74": "under the elevator, 100 mm in
front of the hinge, distance 30 mm") to keep the rudder/elevator AUTHORITATIVE
when slow. Unlike the wing (a lifting surface -> Clmax), the tail is SYMMETRIC,
so the figure of merit is ATTACHMENT over the DEFLECTED control surface: does the
VG keep flow attached at large deflection where it would otherwise separate and
the control "goes soft"? We deflect the surface and read the control normal force
(Cl of the slice = side force for the rudder) + its steadiness.

The question to answer = SPACING ("how many mm across"). IMP74's field value is
30 mm; we bracket it (30 / 50 / 70 mm) at the IMP74 chordwise station (100 mm
ahead of the hinge) using the SAME printed v3 vane, toe-out, beta10.

SECTION CAVEAT: the as-built Glasair III tail airfoil is NOT published (DXF gives
planform/hinge only; construction-manual ribs are login-gated). NACA 0010 is the
educated placeholder (web research: NACA 0009-0012 are the standard tail
sections; symmetric ~10% is textbook for this composite homebuilt). The SPACING
TREND is robust to the exact section; absolute numbers are provisional.

Geometry [DXF / aircraft.yaml]:
  RUDDER : fin-root chord 44.778 in = 1.1374 m; hinge @ 62.876% c; deflect 25deg.
  ELEV   : stab root chord 27.578 in = 0.7005 m; hinge @ 66% c (elev 34% c).
Both NACA 0010.

Run:  python gpu/rapidcfd/build_tail_vane.py --surface rudder
      python gpu/rapidcfd/build_tail_vane.py --surface elevator
"""

import math

import numpy as np
import trimesh

from build_cases import (
    write_case,
    SPAN_OVERHANG,
    ASSETS,
    NU,
    REPO,
    HERE,
    RUNNER,
)
# Reuse the validated v3-vane loader (remap + meters) from the wing re-validation.
from build_user_vane import load_user_vane_local, VANE_STL
import build_user_vane
from geometry.stl_gen import extrude_section
from geometry.airfoil import naca4_coords, resample_airfoil

# ---------------------------------------------------------------------------
# Tail-surface geometry (real DXF numbers from aircraft.yaml) -----------------
# ---------------------------------------------------------------------------
SURFACES = {
    # name : (chord_m, hinge_frac, deflect_deg, section)
    #   chord    : streamwise chord of the (stab+control) section, meters
    #   hinge    : hinge line as fraction of chord (control surface starts here)
    #   deflect  : control deflection tested (the case where flow wants to separate)
    "rudder":   (1.1374, 0.62876, 25.0, "0010"),   # fin-root chord, rudder 37% c
    "elevator": (0.7005, 0.66000, 20.0, "0010"),   # stab root chord, elev 34% c
}

# VG placement (IMP74 + the user's printed champion params) -------------------
AHEAD_OF_HINGE_MM = 100.0   # IMP74: "100 mm in front of the hinge"
BETA_DEG = 10.0             # toe-out incidence (validated champion)
VANE_VER = "v3"             # the printed filleted vane (one part everywhere)

# Tail inflow: the tail sees lower dynamic pressure + its own local AoA. Use the
# wing study's stall-regime Re scaled to the tail chord, at a representative slow
# inflow angle (the surface is symmetric so "alpha" here is the local tail AoA;
# the CONTROL deflection is what loads it). Keep the same NU + a slow-flight Re.
RE_TAIL = 3.0e6             # ~slow-flight tail chord Reynolds (provisional)
ALPHA_INFLOW = 4.0         # deg, modest tail AoA in slow flight (symmetric sec)

SPACINGS_MM = [30.0, 50.0, 70.0]   # bracket IMP74's 30 mm = the "what mm" answer


def deflected_section(chord, hinge_frac, deflect_deg, naca):
    """
    NACA section with the aft (control) part rotated about the hinge by
    deflect_deg (TE toward +y = positive). Returns a Selig loop scaled to the
    chord in METERS, ready for extrude_section.
    """
    # Unit-chord symmetric section (Selig loop), blunt TE for the mesher.
    coords = resample_airfoil(naca4_coords(naca, n_points=241, te="sharp"),
                              n_points=241, te="blunt")
    c = coords.copy()
    # Rotate every point AFT of the hinge about the hinge point on the chord line.
    th = math.radians(deflect_deg)
    ct, st = math.cos(th), math.sin(th)
    hx = hinge_frac
    for i in range(len(c)):
        if c[i, 0] > hx:
            dx, dy = c[i, 0] - hx, c[i, 1] - 0.0
            c[i, 0] = hx + (dx * ct - dy * st)   # rotate about (hx,0)
            c[i, 1] = (dx * st + dy * ct)
    c *= chord                                   # scale to meters
    return c


def upper_point_at(coords, x_frac):
    """(y, slope) of the UPPER surface at chord fraction x_frac (meters in)."""
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]                # ascending x
    xc = coords[:, 0].max()                      # == chord
    xq = x_frac * xc
    y = float(np.interp(xq, upper[:, 0], upper[:, 1]))
    i = int(np.searchsorted(upper[:, 0], xq))
    i = max(1, min(i, len(upper) - 1))
    dx, dy = (upper[i] - upper[i - 1])
    return y, float(np.arctan2(dy, dx))


def build_tail_article(name, surface, pitch_mm, ac_alpha):
    """Stamp a v3 VG pair on the deflected tail section, bake the inflow alpha."""
    chord, hinge_frac, deflect, naca = SURFACES[surface]
    pitch = pitch_mm / 1000.0
    slab = pitch
    beta = math.radians(BETA_DEG)
    stl_span = SPAN_OVERHANG * slab

    # Deflected tail section -> wall solid (same extrusion toolkit as the wing).
    coords = deflected_section(chord, hinge_frac, deflect, naca)
    wing = extrude_section(coords, 1.0, stl_span)   # coords already in meters

    # VG row: AHEAD_OF_HINGE_MM forward of the hinge, on the UPPER surface (the
    # suction side at deflection). Chord fraction of that station:
    x_vg = hinge_frac - (AHEAD_OF_HINGE_MM / 1000.0) / chord
    y_surf, slope = upper_point_at(coords, x_vg)
    x_le = x_vg * chord

    base_vane = load_user_vane_local()              # printed v3, meters, our frame
    vanes = []
    for sgn, z_off in ((+1.0, +pitch / 4.0), (-1.0, -pitch / 4.0)):
        v = base_vane.copy()
        v.apply_transform(trimesh.transformations.rotation_matrix(
            sgn * beta, (0.0, 1.0, 0.0)))           # toe-out
        v.apply_transform(trimesh.transformations.rotation_matrix(
            slope, (0.0, 0.0, 1.0)))                # sit on local skin slope
        v.apply_translation((x_le, y_surf - 0.0006, z_off))
        vanes.append(v)

    wall = trimesh.boolean.union([wing] + vanes)

    # Bake the tail inflow alpha (rotate about quarter chord, like the wing).
    rot = trimesh.transformations.rotation_matrix(
        -math.radians(ac_alpha), (0.0, 0.0, 1.0), (0.25 * chord, 0.0, 0.0))
    wall.apply_transform(rot)

    ASSETS.mkdir(parents=True, exist_ok=True)
    wall_path = ASSETS / f"{name}_wall.stl"
    wall.export(wall_path)
    print(f"[tail] {wall_path.name}: watertight={wall.is_watertight} "
          f"faces={len(wall.faces)} {surface} defl={deflect:g} pitch={pitch_mm:g}mm")

    vblob = trimesh.util.concatenate(vanes)
    vblob.apply_transform(rot)
    vanes_path = ASSETS / f"{name}_vanes.stl"
    vblob.export(vanes_path)
    return wall_path, vanes_path, chord


def build_clean(name, surface, ac_alpha):
    """The no-VG deflected baseline (so we measure the VG's attachment benefit)."""
    chord, hinge_frac, deflect, naca = SURFACES[surface]
    slab = 0.05
    coords = deflected_section(chord, hinge_frac, deflect, naca)
    wall = extrude_section(coords, 1.0, SPAN_OVERHANG * slab)
    rot = trimesh.transformations.rotation_matrix(
        -math.radians(ac_alpha), (0.0, 0.0, 1.0), (0.25 * chord, 0.0, 0.0))
    wall.apply_transform(rot)
    ASSETS.mkdir(parents=True, exist_ok=True)
    wall_path = ASSETS / f"{name}_wall.stl"
    wall.export(wall_path)
    print(f"[tail] {wall_path.name}: CLEAN baseline {surface} defl={deflect:g}")
    return wall_path, None, chord, slab


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", choices=["rudder", "elevator"], default="rudder")
    opts = ap.parse_args()
    surf = opts.surface
    build_user_vane.USER_STL = VANE_STL[VANE_VER]    # point loader at v3

    chord = SURFACES[surf][0]
    names = []

    # Clean deflected baseline (no VG).
    cname = f"tail_{surf}_clean"
    names.append(cname)
    wall, vanes, chord, slab = build_clean(cname, surf, ALPHA_INFLOW)
    u_inf = RE_TAIL * NU / chord
    print(f"[build] {cname}: U={u_inf:.2f} m/s  chord={chord:.4f}m  Re={RE_TAIL:.2g}")
    write_case(cname, wall, vanes, u_inf, chord, slab, n_iter=4000)

    # VG spacing sweep at the IMP74 station.
    for s in SPACINGS_MM:
        tag = f"tail_{surf}_p{int(s):03d}"
        names.append(tag)
        wall, vanes, chord = build_tail_article(tag, surf, s, ALPHA_INFLOW)
        u_inf = RE_TAIL * NU / chord
        print(f"[build] {tag}: U={u_inf:.2f} m/s pitch={s:g}mm "
              f"(100mm ahead of {surf} hinge, v3 vane)")
        write_case(tag, wall, vanes, u_inf, chord, s / 1000.0, n_iter=4000)

    (HERE / "run_all.sh").write_text(RUNNER, newline="\n")
    print("\n[build] WSL:  bash gpu/rapidcfd/run_all.sh " + " ".join(names))


if __name__ == "__main__":
    main()
