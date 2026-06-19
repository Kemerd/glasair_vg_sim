# -*- coding: utf-8 -*-
"""
Build CFD cases for the USER'S custom 6 mm arrowhead delta vane (v3).

This vane (assets/user_vanes/6mm_deltavg_v3.stl) is the user's 3D-printed
refinement of our 6 mm delta champion:
  * a small bonding FLANGE/skirt around the base,
  * the front (nose) faces filleted + curved into an arrowhead / paper-airplane
    so the leading edges are smooth -> the goal is to KEEP the stall recovery
    (the BACK / trailing edge is left tall + sharp for clean separation) while
    REDUCING cruise drag (smooth, rounded front faces shed less).

We stamp it onto the SAME wing slice, at the SAME 7% chord station, as a
counter-rotating pair at the chosen pitch, then bake alpha in -- identical
pipeline to build_cases.py so the result is a fair A/B against vg06d70b10.

The user's STL local frame (measured): +x = aft(tall sharp TE at x=0) -> fwd
(swept nose apex at x~+17); +z = height (0..6mm); +y = span; flange in z~=0.
We must remap to OUR vane frame, which make_delta_vane / build_article expect:
  LE(apex) at x=0, +x downstream along the chord, +y UP (height), thin in z.
So: user-z (height) -> our-y; user-x (chord, nose@+x) -> our-x but FLIPPED so
the nose leads (sits at small x, into the wind); user-y (span) -> our-z.

Run:  python gpu/rapidcfd/build_user_vane.py
      bash gpu/rapidcfd/run_all.sh uvg06v3_a02 uvg06v3_a15 uvg06v3_a18 uvg06v3_a20
"""

import math
from pathlib import Path

import numpy as np
import trimesh

# Reuse the validated project pipeline wholesale.
from build_cases import (
    upper_surface_point,
    write_case,
    SPAN_OVERHANG,
    ASSETS,
    RE,
    NU,
    REPO,
    HERE,
    RUNNER,
)
from geometry.stl_gen import extrude_section
from geometry.airfoil import load_airfoil, resample_airfoil
from geometry.units import load_aircraft

USER_STL = ASSETS / "user_vanes" / "6mm_deltavg_v3.stl"

# Match the champion config exactly so this is a clean A/B vs vg06d70b10.
X_FRAC = 0.07          # tips at 7% chord
BETA_DEG = 10.0        # 10 deg toe-out
PITCH_MM = 70.0        # champion pitch
ALPHAS = [2.0, 15.0, 18.0, 20.0]   # cruise + the stall-bracket the champion used


def load_user_vane_local():
    """
    Load the user's arrowhead vane and remap it into OUR vane frame:
      our-x = chord (nose/apex near x=0, leading into the wind),
      our-y = height (0 at the skin, up to ~6 mm),
      our-z = thickness/span direction (centered on 0).
    Returns a trimesh in METERS, base sitting on y=0, apex toward -? handled by
    caller's placement (same as the parametric builders).
    """
    m = trimesh.load(USER_STL)
    v = m.vertices.copy()
    v -= v.min(axis=0)                 # corner to origin: x 0..17.2, y 0..12, z 0..6

    # --- Remap axes: (user x,y,z) -> (our x,y,z) -------------------------------
    # user-z (height 0..6)      -> our-y (height)
    # user-x (chord)            -> our-x. The user's SHARP TALL edge sits at the
    #   user-x MAX end; we want that sharp edge AFT (large our-x, the trailing
    #   edge for clean separation) and the swept/curved NOSE leading at small
    #   our-x (into the wind). So map user-x straight through (NO flip): the
    #   tall sharp end stays at large x = aft, the low curved nose at small x.
    #   (An earlier flip put the sharp edge forward -- visually confirmed wrong;
    #   this is the user's "rotate 180 about vertical" correction.)
    # user-y (span 0..12)       -> our-z (thickness/span), recentred about 0
    # The tall SHARP 6mm edge is at user-x=0; the swept NOSE at user-x=max.
    # We want sharp edge AFT (large our-x) + nose leading (small our-x), so FLIP.
    ux, uy, uz = v[:, 0], v[:, 1], v[:, 2]
    new = np.empty_like(v)
    new[:, 0] = ux.max() - ux          # FLIP: sharp tall edge (user-x0) -> aft
    new[:, 1] = uz                     # height
    new[:, 2] = uy - uy.mean()         # span/thickness, centered on 0
    m.vertices = new

    # millimeters -> meters (the wing pipeline is in meters).
    m.apply_scale(1.0 / 1000.0)
    m.fix_normals()
    return m


def build_user_article(name, alpha_deg, ac, coords, slab):
    """Stamp the user vane (counter-rotating pair) on the wing at 7%c, bake alpha."""
    chord = ac.wing.aileron.chord_at_mid_station
    pitch = PITCH_MM / 1000.0
    beta = math.radians(BETA_DEG)
    stl_span = SPAN_OVERHANG * slab

    # Clean wing solid (same M0 toolkit as every other case).
    wing = extrude_section(coords, chord, stl_span)

    # Local skin height + slope at the 7% station -> where/how the vane sits.
    y_surf, slope = upper_surface_point(coords, X_FRAC)
    x_le, y_le = X_FRAC * chord, y_surf * chord

    base_vane = load_user_vane_local()

    # Counter-rotating pair, +/- pitch/4 in z, yawed +/-beta (toe-out), then the
    # base rotated to the local skin slope and dropped onto the surface -- the
    # IDENTICAL placement transform build_article uses for the parametric vanes.
    vanes = []
    for sgn, z_off in ((+1.0, +pitch / 4.0), (-1.0, -pitch / 4.0)):
        v = base_vane.copy()
        v.apply_transform(trimesh.transformations.rotation_matrix(
            sgn * beta, (0.0, 1.0, 0.0)))           # toe-out yaw about vertical
        v.apply_transform(trimesh.transformations.rotation_matrix(
            slope, (0.0, 0.0, 1.0)))                # match local skin slope
        # seat slightly INTO the skin so the boolean union is clean (flange + a
        # hair of bite), same -0.10*h trick the parametric builder uses.
        v.apply_translation((x_le, y_le - 0.0006, z_off))
        vanes.append(v)

    wall = trimesh.boolean.union([wing] + vanes)

    # Bake alpha (nose-up = -alpha about z through the quarter chord).
    rot = trimesh.transformations.rotation_matrix(
        -math.radians(alpha_deg), (0.0, 0.0, 1.0), (0.25 * chord, 0.0, 0.0))
    wall.apply_transform(rot)

    ASSETS.mkdir(parents=True, exist_ok=True)
    wall_path = ASSETS / f"{name}_wall.stl"
    wall.export(wall_path)
    print(f"[user] {wall_path.name}: watertight={wall.is_watertight} "
          f"faces={len(wall.faces)} alpha={alpha_deg:g}deg")

    vblob = trimesh.util.concatenate(vanes)
    vblob.apply_transform(rot)
    vanes_path = ASSETS / f"{name}_vanes.stl"
    vblob.export(vanes_path)
    return wall_path, vanes_path


def main():
    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station
    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")
    pitch = PITCH_MM / 1000.0
    slab = pitch                                    # paired layout = one pitch

    names = []
    for a in ALPHAS:
        tag = f"uvg06v3_a{int(a):02d}"
        names.append(tag)
        re_eff = RE
        u_inf = re_eff * NU / chord
        print(f"[build] {tag}: alpha={a:g}deg Re={re_eff:.2g} U={u_inf:.2f} m/s "
              f"pitch={PITCH_MM:.0f}mm  (USER arrowhead vane)")
        wall, vanes = build_user_article(tag, a, ac, coords, slab)
        write_case(tag, wall, vanes, u_inf, chord, slab, n_iter=4000)

    # Refresh the runner so WSL can launch these names.
    (HERE / "run_all.sh").write_text(RUNNER, newline="\n")
    print("\n[build] WSL:  bash gpu/rapidcfd/run_all.sh " + " ".join(names))


if __name__ == "__main__":
    main()
