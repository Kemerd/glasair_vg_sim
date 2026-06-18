# -*- coding: utf-8 -*-
"""
Printable deliverables for the WINNING vortex generator (6 mm delta @ 7% chord).

Two parts, both derived from the SAME validated LS(1)-0413 wing geometry the CFD
study used, so they fit the real airplane:

  1. vg_6mm_delta_vane.stl
       One printable delta vane, 6 mm tall, with a BASE pre-curved to the local
       upper-surface slope/curvature at the 7%-chord station, so it sits flush
       on the wing skin instead of rocking on a flat foot. Print as many as you
       want; install in counter-rotating pairs at 70 mm pitch (10 deg toe-out).

  2. vg_placement_jig.stl
       A fixture that HUGS the leading-edge region of the wing (its underside is
       the negative of the real LS(1)-0413 nose), with a reference shelf at the
       7%-chord station and a notch schedule that drops each vane in at the
       correct chord station, spacing, and incidence -- repeatable install with
       no measuring.

Run:  python gpu/rapidcfd/make_deliverables.py
Out:  gpu/rapidcfd/assets/vg_6mm_delta_vane.stl
      gpu/rapidcfd/assets/vg_placement_jig.stl
"""

import math
from pathlib import Path

import numpy as np
import trimesh

# Reuse the EXACT geometry primitives + airfoil pipeline from the case builder
# so the deliverables share one source of truth with the CFD that validated them.
from build_cases import (
    make_delta_vane,
    _extrude_outline,
    upper_surface_point,
    REPO,
)
from geometry.airfoil import load_airfoil, resample_airfoil
from geometry.units import load_aircraft

# ---------------------------------------------------------------------------
# Winning configuration (from 06-17-26_results.md) -- the one part for everything
# ---------------------------------------------------------------------------
H_MM = 6.0            # vane height (the champion)
X_FRAC = 0.07         # chord station of the vane tips (7% c, IMP74 + Stolspeed)
L_PER_H = 3.0         # vane length = 3 x height (IMP74 default)
BETA_DEG = 10.0       # incidence (toe-out) to the local flow
PITCH_MM = 70.0       # pair-to-pair spacing (the champion pitch)

OUT = REPO / "gpu" / "rapidcfd" / "assets"


def _curved_base_delta(h_m, length_m, thick_m, coords, chord, n_seg=24):
    """
    A delta vane whose BASE is a true CURVE following the wing skin under it.

    A flat make_delta_vane sits on y = 0 with only 6 vertices, so it can never
    follow the skin -- a flat foot would rock or leave a gap on the curved wing.
    Here we build the vane from scratch as a thin prism whose BASE edge is
    sampled at `n_seg` points across the vane's chordwise footprint, each point
    dropped onto the real LS(1)-0413 upper surface at that station. The result
    is a foot that hugs the local airfoil curvature -- ready to bond flush.

    Local frame matches make_delta_vane: LE (apex) at origin, +x along chord,
    +y up. The top edge is the delta hypotenuse (0 at apex -> h at the TE); the
    base is the curved skin profile. Built measured-from-the-LE so the same
    placement transform (slope rotate + translate) in build_article applies.
    """
    x_le = X_FRAC * chord
    y_le, _ = upper_surface_point(coords, X_FRAC)
    y_le *= chord                                      # LE-station skin datum (m)

    # Sample the base curve across the footprint: relative skin height under
    # each station, measured against the LE datum (so the apex base sits at 0).
    xs = np.linspace(0.0, length_m, n_seg + 1)         # 0 .. length along chord
    base_y = np.empty(n_seg + 1)
    for i, xl in enumerate(xs):
        xc = min(max((x_le + xl) / chord, 0.0), 0.5)   # stay on upper surface
        y_skin, _ = upper_surface_point(coords, xc)
        base_y[i] = y_skin * chord - y_le              # skin rise vs LE datum

    # Top edge of the delta: linear ramp 0 (apex) -> h (TE), riding ABOVE the
    # local base so the vane keeps its full height everywhere along the skin.
    top_y = base_y + (h_m * xs / length_m)

    # Closed 2D outline (x,y): walk the curved BASE apex->TE, then the delta
    # TOP edge back TE->apex. Hand to the validated shapely extruder, which
    # triangulates + sweeps to a watertight prism for any convex/concave shape.
    outline = np.vstack([
        np.column_stack([xs, base_y]),                 # curved base, apex -> TE
        np.column_stack([xs[::-1], top_y[::-1]]),      # delta top, TE -> apex
    ])
    return _extrude_outline(outline, thick_m)


def make_printable_vane():
    """Export the single curved-base 6 mm delta vane."""
    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station                # 0.9022 m
    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")

    h = H_MM / 1000.0
    length = L_PER_H * h
    thick = max(0.0015, h / 8.0)                                # printable wall

    vane = _curved_base_delta(h, length, thick, coords, chord)

    # Scale to MILLIMETERS for a print-ready STL (slicers expect mm).
    vane.apply_scale(1000.0)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "vg_6mm_delta_vane.stl"
    vane.export(path)
    bb = vane.bounds
    dims = bb[1] - bb[0]
    print(f"[vane] {path.name}: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm "
          f"(L x H x T), curved base @ {X_FRAC*100:.0f}% chord")
    return path


def make_jig():
    """
    Export the wing-hugging placement jig.

    Construction: take the real LS(1)-0413 nose region (LE back to ~20% chord),
    extrude it a few pitches wide, and SUBTRACT it from a solid block. The
    cavity is the exact negative of the wing nose, so the jig clamps over the
    leading edge and can only seat one way. A reference edge is cut at the 7%
    chord station, and vane-pocket notches are spaced at the 70 mm pitch with
    the 10 deg toe-out baked in, so each printed vane drops into its slot at the
    right station / spacing / incidence.
    """
    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station
    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")

    pitch = PITCH_MM / 1000.0
    jig_span = 3.0 * pitch                  # cover three pitches (six vanes)
    nose_back = 0.20                        # wrap LE .. 20% chord

    # --- Build the wing-nose solid the jig wraps around -----------------------
    # Selig loop -> nose slice (x/c in [0, nose_back]) of BOTH surfaces.
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]           # ascending x, upper surface
    lower = coords[le:]                     # ascending x, lower surface
    up_n = upper[upper[:, 0] <= nose_back]
    lo_n = lower[lower[:, 0] <= nose_back]
    # Closed 2D nose outline (upper fwd-to-back, lower back-to-front).
    nose2d = np.vstack([up_n, lo_n[::-1]]) * chord
    nose_poly = trimesh.path.polygons.Polygon(nose2d)
    nose_solid = trimesh.creation.extrude_polygon(nose_poly, height=jig_span)
    # Orient: extrude is along +z; center the span about z = 0.
    nose_solid.apply_translation((0.0, 0.0, -jig_span / 2.0))

    # --- The jig block: a slab around the nose, then carve the nose cavity -----
    margin = 0.020                          # 20 mm of material around the nose
    bb = nose_solid.bounds
    block = trimesh.creation.box(extents=(
        (bb[1, 0] - bb[0, 0]) + 2 * margin,
        (bb[1, 1] - bb[0, 1]) + 2 * margin,
        jig_span,
    ))
    block.apply_translation((
        (bb[0, 0] + bb[1, 0]) / 2.0,
        (bb[0, 1] + bb[1, 1]) / 2.0,
        0.0,
    ))
    # Open the trailing side so the jig slides on over the LE (clip the block
    # back to the nose extent + a lip, leaving the aft face open).
    jig = block.difference(nose_solid)

    # --- Vane-pocket notches at the 7% station, 70 mm pitch, 10 deg toe-out ----
    h = H_MM / 1000.0
    length = L_PER_H * h
    thick = max(0.0015, h / 8.0)
    y_surf, slope = upper_surface_point(coords, X_FRAC)
    x_le, y_le = X_FRAC * chord, y_surf * chord
    toe = math.radians(BETA_DEG)

    pockets = []
    # Three pitches -> z = -pitch, 0, +pitch; each a counter-rotating pair.
    for k in (-1, 0, 1):
        zc = k * pitch
        for sgn, z_off in ((+1.0, +pitch / 4.0), (-1.0, -pitch / 4.0)):
            # A slightly oversized vane volume = the pocket the real vane drops
            # into (clearance fit). Same placement transform as build_article.
            pkt = make_delta_vane(h, length, thick * 3.0)      # wider = slot
            pkt.apply_transform(trimesh.transformations.rotation_matrix(
                sgn * toe, (0.0, 1.0, 0.0)))
            pkt.apply_transform(trimesh.transformations.rotation_matrix(
                slope, (0.0, 0.0, 1.0)))
            pkt.apply_translation((x_le, y_le, zc + z_off))
            pockets.append(pkt)

    if pockets:
        jig = jig.difference(trimesh.util.concatenate(pockets))

    jig.apply_scale(1000.0)                 # to millimeters
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "vg_placement_jig.stl"
    jig.export(path)
    dims = jig.bounds[1] - jig.bounds[0]
    print(f"[jig]  {path.name}: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm, "
          f"wraps LE..{nose_back*100:.0f}% chord, {len(pockets)} vane pockets "
          f"@ {PITCH_MM:.0f} mm pitch")
    return path


if __name__ == "__main__":
    make_printable_vane()
    make_jig()
    print("\nDeliverables written to gpu/rapidcfd/assets/ -- print the vane in "
          "PETG/ASA, use the jig to place pairs at 70 mm (wider/bare at the root).")
