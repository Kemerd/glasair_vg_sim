# -*- coding: utf-8 -*-
"""
Full 3D empennage assembly for the FluidX3D tunnel: aft-fuselage stinger +
vertical fin + raked-hinge rudder (with its tip horn balance) + tapered
horizontal stab + elevators (with their tip horn balances), all lofted
directly from the factory 3-view linework, with the real 1/16 in open hinge
gaps and rigid control deflections. One binary STL per case, four disjoint
closed shells per file: [fixed assembly, right elevator, left elevator,
rudder] -- ray-parity voxelization handles disjoint shells cleanly.

Why this exists: the 2.5D tail articles (make_tail_articles.py) answer the
periodic-section question; this article answers the OWNER's question -- does
the whole tail keep control authority at low airspeed -- including the tip
horns, the taper, the fin/stab/fuselage interference, and the rudder bottom
passing between the elevator roots.

GEOMETRY PROVENANCE -- all coordinates below are direct entity reads from
the owner-converted factory 3-view DXFs (geometry/dxf/glasair_topview.dxf
and glasair_sideview.dxf, verified 1:1 scale in inches), the same drawings
scripts/measure_dxf.py distilled into geometry/dxf/measured.yaml and
aircraft.yaml [DXF] entries. Cross-checks reproduced here at import time
would be circular; instead the assembly is validated against the scalar
[DXF] values already in aircraft.yaml (root chords, hinge rake, gap).

Drawing-decoded facts this model carries (and the entities they came from):
  * stab hinge line at constant FS 223.469, half-extent 48.225 in from CL
    (LINE 223.469,141.578 -> 223.469,188.145; CL at y=139.92)
  * elevator tip HORN: the split runs forward from the hinge tip and wraps
    the stab tip (LINEs to 216.446,189.757 and 216.414,192.091)
  * elevator roots stop 1.658 in from CL with a slanted closeout (LINE
    229.298,141.637 -> 232.335,144.267) -- the channel the rudder bottom
    passes through
  * fin LE straight (206.703,66.277 -> 238.118,92.588), 12 in tip arc,
    flat tip cap at WL 95.39, dorsal fillet arc r=18 tangent to the LE
  * rudder hinge raked 22.125 deg aft (LINE 227.619,48.469 -> 242.998,
    86.293); the horn split (-> 234.315,89.404) lands exactly ON the fin
    LE and is PERPENDICULAR to the hinge axis, so the horn cannot bind
  * the fuselage bottom upsweep line (157.911,36.168 -> 248.165,52.094)
    passes through the hinge bottom -- the rudder bottom edge follows it
  * stab waterline 58.25 from the sideview stab-section silhouette arcs

SECTION CAVEAT (same TODO as everywhere in this repo): both tail airfoils
are NACA 0010 placeholders; the as-built sections are unmeasured. The
planforms and hinge geometry above are drawing-true.

COORDINATES (FluidX3D tunnel convention): x = chordwise/flow (fuselage
station), y = up (waterline), z = spanwise from centerline. Built in
drawing inches, scaled to meters at export, origin moved to the stab
hinge / stab waterline / CL.

SIGN CONVENTIONS: elevator positive = TE down (stl_gen convention), so a
NOSE-UP command is a NEGATIVE deflection (the u* articles). Rudder
positive = TE toward +z; the symmetric section makes the sign a mirror.

Run:  python gpu/fluidx3d/make_tail_assembly.py [--elev-deg 15]
      [--rud-deg 15] [--vg-height-mm 10] [--vg-pitch-mm 0]
      [--vane-thickness-mm 0] [--no-vg] [--no-render] [--only TAG]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon

from geometry.units import UNIT_TO_SI, load_aircraft

# Sibling import for the shared offscreen preview renderer.
sys.path.insert(0, str(REPO / "gpu" / "fluidx3d"))
from make_tail_articles import render_preview

ASSETS = REPO / "gpu" / "fluidx3d" / "assets"
YAML = REPO / "aircraft.yaml"

IN = UNIT_TO_SI["in"]            # the single sanctioned inch->meter factor

# =============================================================================
#  Drawing constants (inches; [DXF] entity reads, see module docstring)
# =============================================================================

# ---- topview, z measured from the aircraft centerline (drawing y 139.92) ----
STAB_LE_CL = 205.017             # LE line extrapolated to CL (chord 27.578 [DXF])
STAB_TE_CL = 232.595             # TE line extrapolated to CL
STAB_LE_TIP = (212.232, 50.627)  # LE line outboard end
STAB_TIP_LE_ARC = ((214.212, 50.345), 2.0, 171.9, 85.3)   # (center, r, a0, a1)
STAB_TIP_EDGE = (214.376, 52.338, 228.670, 51.168)        # tip edge segment
STAB_TIP_TE_ARC = ((228.589, 50.171), 1.0, 85.3, 3.4)
STAB_TE_TIP = (229.587, 50.231)  # TE line outboard end
STAB_HINGE_X = 223.469           # constant-FS elevator hinge [DXF]
STAB_HINGE_Z = (1.658, 48.225)   # hinge line half-span extents from CL
STAB_HORN = ((223.469, 48.225), (216.446, 49.837), (216.414, 52.171))
STAB_ROOT_RIB = ((229.298, 1.717), (232.335, 4.347))      # slanted inboard rib
STAB_WL = 58.25                  # stab waterline (sideview silhouette arcs)

# ---- sideview, (x = fuselage station, y = waterline) ------------------------
FIN_LE = ((206.703, 66.277), (238.118, 92.588))           # straight LE
FIN_DORSAL_ARC = ((195.145, 80.077), 18.0, 262.1, 309.9)  # fillet, tangent to LE
FIN_TIP_ARC = ((245.823, 83.389), 12.0, 129.9, 90.1)      # LE -> flat tip cap
FIN_TIP_TOP = (245.810, 95.389, 258.295, 95.182)          # flat cap, fwd -> aft
FIN_TE_TIP_ARC = ((258.238, 93.986), 1.198, 87.3, -92.7)  # TE tip rounding
FIN_TE = ((257.582, 92.312), (248.158, 52.094))           # rudder TE line
RUD_HINGE_P0 = (227.619, 48.469)                          # hinge bottom point
RUD_HINGE_P1 = (242.998, 86.293)                          # hinge top (horn start)
RUD_HORN_LE = (234.315, 89.404)  # horn split endpoint, ON the fin LE
UPSWEEP_P = (157.911, 36.168)    # fuselage bottom line through hinge bottom
UPSWEEP_SLOPE = (52.094 - 36.168) / (248.165 - 157.911)   # 0.176458
FUSE_SIDE = (129.334, 20.191, 219.271, 1.909)             # topview side line:
                                                          # (x0, hw0, x1, hw1)
FUSE_HW_MIN = 1.55               # stern-post half-width floor: covers the
                                 # rudder root thickness, clears the 1.658 in
                                 # elevator root rib by ~2.7 mm
FUSE_NOSE_X = 180.0              # stinger nose shoulder station
FUSE_NOSE_LEN = 18.0             # ellipsoidal nose cap length. A smooth
                                 # closed nose is the standard forebody for
                                 # an empennage-only article -- a flat front
                                 # cut would be a bluff face (separation
                                 # bubble + shedding washing over the tail).
FIN_ROOT_WL = 66.277             # fin root rib waterline (fuselage top aft)
DORSAL_X0 = 192.657              # dorsal arc forward tangency station

# Rudder hinge unit direction (raked 22.125 deg aft of vertical [DXF]).
_RAKE = math.atan2(RUD_HINGE_P1[0] - RUD_HINGE_P0[0],
                   RUD_HINGE_P1[1] - RUD_HINGE_P0[1])
RUD_DIR = np.array([math.sin(_RAKE), math.cos(_RAKE), 0.0])

# Rudder TE line slope (dx/dy) and the fin-root chord/rudder-chord anchors
# used by the lower-rudder section construction below the root rib.
TE_SLOPE = (FIN_TE[0][0] - FIN_TE[1][0]) / (FIN_TE[0][1] - FIN_TE[1][1])
FIN_ROOT_CHORD = 44.778          # chord_root_incl_rudder [DXF/aircraft.yaml]
RUD_FRAC_ROOT = 0.62876          # hinge x/c at the fin root (1 - 0.3712 [DXF])


# =============================================================================
#  Small geometry helpers
# =============================================================================

def naca0010_half(x_frac: np.ndarray) -> np.ndarray:
    """NACA 0010 half-thickness y_t/c at x/c (sharp-TE closure, a4=-0.1036).

    Same closed form as geometry.airfoil.naca4_coords; restated here so the
    surface probes (VG seating, cove radii) need no loop bookkeeping.
    """
    x = np.clip(np.asarray(x_frac, dtype=float), 0.0, 1.0)
    return 5.0 * 0.10 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                         + 0.2843 * x ** 3 - 0.1036 * x ** 4)


def arc_pts(center: tuple[float, float], r: float, a0_deg: float,
            a1_deg: float, n: int = 24) -> list[tuple[float, float]]:
    """Sample a DXF arc from start angle to end angle (degrees, CCW+)."""
    th = np.radians(np.linspace(a0_deg, a1_deg, n))
    return [(center[0] + r * math.cos(t), center[1] + r * math.sin(t))
            for t in th]


def _ring_unit(n_pts: int = 121) -> np.ndarray:
    """Closed unit-chord NACA 0010 ring (no duplicate TE point), cosine x."""
    n_side = n_pts // 2 + 1
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_side)))
    yt = naca0010_half(x)
    upper = np.column_stack([x[::-1], yt[::-1]])      # TE -> LE
    lower = np.column_stack([x[1:-1], -yt[1:-1]])     # LE+1 -> TE-1
    return np.vstack([upper, lower])                  # single TE/LE points

_RING = _ring_unit()


def loft(rings: list[np.ndarray]) -> trimesh.Trimesh:
    """Watertight solid from a stack of same-count 3D section rings.

    Side walls are quads split into triangles; the two end caps are centroid
    fans (every section here is star-shaped about its centroid, so the fan
    is valid). Winding is normalized afterwards by the volume sign.
    """
    m = len(rings)
    n = len(rings[0])
    verts = np.vstack(rings)
    faces: list[tuple[int, int, int]] = []
    for i in range(m - 1):
        a0 = i * n
        b0 = (i + 1) * n
        for j in range(n):
            j1 = (j + 1) % n
            faces.append((a0 + j, a0 + j1, b0 + j1))
            faces.append((a0 + j, b0 + j1, b0 + j))
    # End caps: fan from the appended centroid vertex of each end ring.
    c0 = len(verts)
    verts = np.vstack([verts, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    for j in range(n):
        j1 = (j + 1) % n
        faces.append((c0, j1, j))                                  # first cap
        faces.append((c0 + 1, (m - 1) * n + j, (m - 1) * n + j1))  # last cap
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    if mesh.volume < 0.0:
        mesh.invert()
    if not mesh.is_watertight:                        # unexpected for clean input
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    return mesh


def chord_interval(poly: Polygon, station: float, horizontal: bool) -> tuple[float, float]:
    """[min,max] chordwise extent of a planform polygon at one station.

    horizontal=True probes y=station (fin outline); False probes z=station
    (stab planform, second coordinate is z). Every station used here cuts a
    single interval; the overall bounds of the intersection are that interval.
    """
    minx, miny, maxx, maxy = poly.bounds
    if horizontal:
        probe = LineString([(minx - 10, station), (maxx + 10, station)])
    else:
        probe = LineString([(minx - 10, station), (maxx + 10, station)])
    xsec = poly.intersection(probe)
    if xsec.is_empty:
        raise ValueError(f"station {station} outside planform")
    return xsec.bounds[0], xsec.bounds[2]


def prism(poly_pts: list[tuple[float, float]], gap: float,
          lo: float, hi: float, plane: str) -> trimesh.Trimesh:
    """Extruded cutting prism from a 2D polygon, optionally grown by `gap`.

    plane='xz': polygon lives in (x, z), extruded along y from lo..hi (the
    elevator split regions). plane='xy': polygon in (x, y), extruded along z
    (the rudder split region). gap > 0 buffers the polygon outward -- that is
    the open-gap construction: control = solid INTERSECT prism(0), fixed =
    solid MINUS prism(gap), so the strip between is removed everywhere.
    """
    p = Polygon(poly_pts)
    if gap > 0.0:
        p = p.buffer(gap, quad_segs=16)
    mesh = trimesh.creation.extrude_polygon(p, height=hi - lo)
    if plane == "xy":
        mesh.apply_translation((0.0, 0.0, lo))
    else:
        # extrude_polygon builds (x, y)->+z; remap (x, z, h) -> (x, h, z):
        # swap y/z axes (negative-determinant transform; trimesh re-winds).
        T = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0],
                      [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
        mesh.apply_transform(T)
        mesh.apply_translation((0.0, lo, 0.0))
    return mesh


def frustum_along(p0: np.ndarray, direction: np.ndarray,
                  stations: np.ndarray, radii: np.ndarray) -> trimesh.Trimesh:
    """Solid of revolution with varying radius along an arbitrary axis.

    stations are distances along `direction` from p0; radii the local cove
    radius at each. Built with trimesh.creation.revolve (+z axis, profile =
    (radius, height)) then aligned to the hinge axis. This is the clearance
    cove: the control nose sweeps a hinge-centered circle when deflected, so
    the fixed side is relieved to (local half-thickness + gap) everywhere.
    """
    prof = [(0.0, float(stations[0]))]
    prof += [(float(r), float(s)) for s, r in zip(stations, radii)]
    prof += [(0.0, float(stations[-1]))]
    m = trimesh.creation.revolve(np.array(prof), sections=48)
    T = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    m.apply_transform(T)
    m.apply_translation(p0)
    return m


# =============================================================================
#  Planform outlines (closed shapely polygons in drawing inches)
# =============================================================================

def stab_planform() -> Polygon:
    """Full stab planform in (x, z), both halves, tip arcs included."""
    half = [(STAB_LE_CL, 0.0), STAB_LE_TIP]
    half += arc_pts(*STAB_TIP_LE_ARC)[1:]            # LE tip corner
    half += [(STAB_TIP_EDGE[2], STAB_TIP_EDGE[3])]   # tip edge aft end
    half += arc_pts(*STAB_TIP_TE_ARC)[1:]            # TE tip corner
    half += [(STAB_TE_CL, 0.0)]
    mirror = [(x, -z) for x, z in reversed(half)][1:-1]
    return Polygon(half + mirror)


def fin_outline() -> Polygon:
    """Fin + rudder side outline in (x, y), virtual LE carried down to the
    fuselage upsweep so sections stay continuous through the root region
    (everything forward of the hinge below the root rib ends up buried in
    the fuselage and is absorbed by the union)."""
    y_a = UPSWEEP_P[1] + UPSWEEP_SLOPE * (DORSAL_X0 - UPSWEEP_P[0])
    pts = [(DORSAL_X0, y_a), (DORSAL_X0, 62.250)]
    pts += arc_pts(*FIN_DORSAL_ARC)[1:]              # dorsal fillet -> root LE
    pts += [FIN_LE[1]]                               # straight LE -> tip
    pts += arc_pts(*FIN_TIP_ARC)[1:]                 # LE tip arc -> flat cap
    pts += [(FIN_TIP_TOP[2], FIN_TIP_TOP[3])]        # flat cap aft end
    pts += arc_pts(*FIN_TE_TIP_ARC)[1:]              # TE tip rounding
    pts += [FIN_TE[0], FIN_TE[1]]                    # TE line down
    return Polygon(pts)                              # closes along the upsweep


def rud_hinge_x(y: float) -> float:
    """Rudder hinge-line station at waterline y (raked 22.125 deg aft)."""
    return RUD_HINGE_P0[0] + math.tan(_RAKE) * (y - RUD_HINGE_P0[1])


def rud_te_x(y: float) -> float:
    """Rudder aft boundary at waterline y: the TE line above its bottom
    corner, the fuselage upsweep line below it (the drawing runs the
    bottom line exactly through both the hinge bottom and the TE corner)."""
    if y >= FIN_TE[1][1]:
        return FIN_TE[1][0] + TE_SLOPE * (y - FIN_TE[1][1])
    return UPSWEEP_P[0] + (y - UPSWEEP_P[1]) / UPSWEEP_SLOPE


def lower_fin_section(y: float) -> tuple[float, float, float]:
    """(x_le_virtual, x_te, scale) for stations BELOW the fin root rib.

    Below the root the only real structure is the rudder; its section is
    modeled as the fin-root section scaled congruently (constant hinge x/c)
    to the local rudder chord and anchored at the local TE/upsweep. The
    scale is capped slightly below 1 so the lower rudder stays thinner
    than the 1.658 in elevator root-rib channel it passes through -- the
    real rudder is a narrow slab here; its thickness is not on the drawing,
    so this cap is the honest stand-in (clearance >= ~3 mm against the
    elevator ribs instead of an interference).
    """
    hi = rud_te_x(y)
    s_raw = (hi - rud_hinge_x(y)) / (FIN_ROOT_CHORD * (1.0 - RUD_FRAC_ROOT))
    s_cap = 1.0 - 0.07 * min(max((FIN_ROOT_WL - y) / 6.0, 0.0), 1.0)
    s = min(s_raw, s_cap)
    chord = max(FIN_ROOT_CHORD * s, 0.05)
    return hi - chord, hi, s


# =============================================================================
#  Control-region polygons (the split machinery)
# =============================================================================

def elev_ctrl_pts(fixed_cut: bool = False) -> list[tuple[float, float]]:
    """Right-elevator region in (x, z): aft of the hinge plus the tip horn,
    bounded inboard by the slanted root rib from the drawing.

    fixed_cut=True returns the variant used to relieve the FIXED side: the
    slanted horn-root boundary (h1 -> h0) is set back an extra 0.12 in
    toward the fixed wedge. That slot's gap direction is mostly spanwise,
    and rotation about the spanwise hinge LEANS the horn's front wall
    forward by ~yt*sin(defl) -- a uniform 1/16 in buffer would pinch shut
    near 25 deg. Real horn balances carry a wider rib gap here for the
    same reason. The control itself always keeps its drawn size.
    """
    (rib0, rib1) = STAB_ROOT_RIB
    (h0, h1, h2) = STAB_HORN
    if fixed_cut:
        # Unit normal of the h0->h1 boundary, pointing into the horn; the
        # setback moves the cut the other way (more relief in the wedge).
        seg = np.subtract(h1, h0)
        n = np.array([seg[1], -seg[0]]) / np.linalg.norm(seg)
        if n[1] < 0:                       # orient into the horn (+z) first;
            n = -n                         # the setback then subtracts it
        h0 = tuple(np.subtract(h0, 0.12 * n))
        h1 = tuple(np.subtract(h1, 0.12 * n))
    return [
        (STAB_HINGE_X, STAB_HINGE_Z[0]),   # hinge @ inboard rib
        rib0, rib1,                        # slanted inboard closeout [DXF]
        (236.0, 4.6), (236.0, 55.0),       # envelope aft/outboard of the tip
        (h2[0], 55.0), h2, h1, h0,         # horn front edge, then to hinge tip
    ]


def rud_ctrl_pts() -> list[tuple[float, float]]:
    """Rudder region in (x, y): aft of the raked hinge plus the tip horn
    (the horn split continues past the fin LE so the whole tip cap moves
    with the rudder, as drawn)."""
    p_lo = (rud_hinge_x(40.0), 40.0)       # hinge extended below everything
    ext = (RUD_HORN_LE[0] - 0.8 * (RUD_HINGE_P1[0] - RUD_HORN_LE[0]),
           RUD_HORN_LE[1] + 0.8 * (RUD_HORN_LE[1] - RUD_HINGE_P1[1]))
    return [p_lo, RUD_HINGE_P1, RUD_HORN_LE, ext,
            (ext[0], 105.0), (270.0, 105.0), (270.0, 40.0)]


# =============================================================================
#  Lofted solids
# =============================================================================

def loft_stab(n_st: int = 161, n_ring: int = 121) -> trimesh.Trimesh:
    """Full (undeflected, unsplit) stab solid, built at LOCAL waterline y=0."""
    poly = stab_planform()
    z_lim = poly.bounds[3] - 0.02                    # stay inside the tip arcs
    z = -z_lim * np.cos(np.linspace(0.0, math.pi, n_st))
    rings = []
    for zi in z:
        lo, hi = chord_interval(poly, zi, horizontal=True)
        c = max(hi - lo, 0.05)
        r2 = _RING * c                               # scale unit ring
        rings.append(np.column_stack([lo + r2[:, 0], r2[:, 1],
                                      np.full(len(r2), zi)]))
    return loft(rings)


def loft_fin(n_st: int = 121) -> trimesh.Trimesh:
    """Fin + rudder solid in absolute (x, y), thickness along z.

    Above the root rib the section spans the real outline (LE line, tip
    arcs, TE). Below it the section is the scaled lower-rudder construction
    from lower_fin_section() -- see its docstring for why the full virtual
    chord is NOT carried down (it would be twice as thick as the fuselage
    stern and would foul the elevator roots).
    """
    poly = fin_outline()
    y0 = RUD_HINGE_P0[1] + 0.4                       # just above the bottom corner
    y1 = poly.bounds[3] - 0.02                       # just under the flat cap
    y = 0.5 * (y0 + y1) - 0.5 * (y1 - y0) * np.cos(np.linspace(0, math.pi, n_st))
    # Pin stations either side of the root rib so the slope break is sharp.
    y = np.unique(np.concatenate([y, [FIN_ROOT_WL - 0.02, FIN_ROOT_WL + 0.02]]))
    rings = []
    for yi in y:
        if yi >= FIN_ROOT_WL:
            lo, hi = chord_interval(poly, yi, horizontal=True)
        else:
            lo, hi, _ = lower_fin_section(yi)
        c = max(hi - lo, 0.05)
        r2 = _RING * c
        rings.append(np.column_stack([lo + r2[:, 0], np.full(len(r2), yi),
                                      r2[:, 1]]))
    return loft(rings)


# Sideview turtledeck ridge line: runs from the canopy fairing AFT and ends
# EXACTLY on the fin LE line at (209.41, 68.544) [DXF entity check] -- the
# drawing itself merges the deck into the fin, which is why the fuselage top
# is modeled as a narrowing ridge rather than a round ellipse crown.
SPINE_P0 = (172.480, 65.066)
SPINE_SLOPE = (68.544 - 65.066) / (209.410 - 172.480)    # 0.09418


def fin_root_half_thickness(x: float) -> float:
    """Half-thickness of the NACA 0010 fin-root section at station x (in).

    Zero outside the root chord footprint; this is what the fuselage ridge
    width tracks under the fin so deck and fin read as ONE surface.
    """
    frac = (x - FIN_LE[0][0]) / FIN_ROOT_CHORD
    if frac <= 0.0 or frac >= 1.0:
        return 0.0
    return float(naca0010_half(frac)) * FIN_ROOT_CHORD


def loft_fuselage(n_st: int = 90, n_ring: int = 64) -> trimesh.Trimesh:
    """Aft-fuselage stinger lofted from TEARDROP sections: a full-beam lower
    lobe (topview side lines, stern-post floor) blending into a narrow top
    ridge that follows the sideview spine line straight into the fin LE,
    with ridge width tracking the local fin thickness. This makes deck and
    fin one continuous body instead of a fin plate stuck on a round hull.
    Ellipsoidal nose cap forward so the tunnel sees a closed body."""
    def y_top(x: float) -> float:
        # Spine line to its fin-LE intersection, then ride the fin LE up a
        # little so the union with the fin solid overlaps generously; the
        # cap keeps the fuselage from ballooning into a second fin.
        if x <= 209.410:
            return SPINE_P0[1] + SPINE_SLOPE * (x - SPINE_P0[0])
        return min(FIN_LE[0][1] + (x - FIN_LE[0][0]) / 1.19399, 70.0)

    def y_bot(x: float) -> float:
        return UPSWEEP_P[1] + UPSWEEP_SLOPE * (x - UPSWEEP_P[0])

    def hw(x: float) -> float:
        x0, w0, x1, w1 = FUSE_SIDE
        return max(FUSE_HW_MIN, w0 + (w1 - w0) * (x - x0) / (x1 - x0))

    def w_ridge(x: float, beam: float) -> float:
        # Under the fin: 0.97x the fin's local half-thickness (fin stays a
        # hair proud, so the seam reads as the fin emerging from the deck).
        # Forward: the deck crown narrows smoothly from near-round at the
        # nose cut toward the fin-LE thickness, canopy-fairing style.
        fwd = beam * (0.72 - 0.50 * min(max((x - 162.0) / 44.7, 0.0), 1.0))
        return max(0.97 * fin_root_half_thickness(x), fwd, 0.30)

    xs = np.linspace(FUSE_NOSE_X - FUSE_NOSE_LEN + 0.15, 248.0, n_st)
    phi = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
    v = -np.cos(phi)                                 # -1 bottom pole, +1 apex
    v_m = -0.10                                      # widest beam waterline
    rings = []
    for x in xs:
        xc = max(x, FUSE_NOSE_X)                     # profiles frozen on the cap
        yb, yt = y_bot(xc), y_top(xc)
        beam = hw(xc)
        ridge = min(w_ridge(xc, beam), beam)
        s = 1.0
        if x < FUSE_NOSE_X:                          # ellipsoidal nose shrink
            s = math.sqrt(max(1.0 - ((FUSE_NOSE_X - x) / FUSE_NOSE_LEN) ** 2,
                              1.0e-4))
        # Width profile: elliptical from the bottom pole up to the widest
        # waterline, then a cosine blend that lands on the ridge width; the
        # sin(phi) ring closure rounds the ridge over the apex.
        w = np.where(
            v <= v_m,
            beam * np.sqrt(np.clip(1.0 - ((v - v_m) / (1.0 + v_m)) ** 2,
                                   0.0, 1.0)),
            ridge + (beam - ridge)
            * np.cos(0.5 * math.pi * (v - v_m) / (1.0 - v_m)) ** 1.35,
        )
        y = yb + (v + 1.0) / 2.0 * (yt - yb)
        cy = 0.5 * (yt + yb)
        rings.append(np.column_stack([np.full(n_ring, x),
                                      cy + (y - cy) * s,
                                      w * np.sin(phi) * s]))
    return loft(rings)


# =============================================================================
#  Vortex generators (Strausak elevator convention from aircraft.yaml)
# =============================================================================

def vane_boxes(h: float, length: float, thick: float, beta: float,
               positions: list[tuple[float, float, float, float, str]],
               ) -> list[trimesh.Trimesh]:
    """Counter-rotating vane plates at precomputed seats.

    positions: (x, y, z, sgn, axis) per vane -- sgn alternates the incidence
    (toe-out pairs), axis is the surface-normal: 'y' for the stab underside
    row (plates stand -y) or 'z' for the fin rows (plates stand +/-z, the
    z sign rides in the seat coordinate's sign).
    """
    out = []
    for (x, y, z, sgn, axis) in positions:
        box = trimesh.creation.box(extents=(length, thick, h) if axis == "z"
                                   else (length, h, thick))
        if axis == "y":      # stab underside: stand -y, sink 15% into skin
            box.apply_translation((length / 2.0, -h / 2.0, 0.0))
            rot = trimesh.transformations.rotation_matrix(sgn * beta, (0, 1, 0))
        else:                # fin side: stand toward z sign of the seat
            sz = 1.0 if z >= 0 else -1.0
            box.apply_translation((length / 2.0, 0.0, sz * h / 2.0))
            rot = trimesh.transformations.rotation_matrix(sgn * beta, (0, 0, 1))
        box.apply_transform(rot)
        box.apply_translation((x, y, z))
        out.append(box)
    return out


def elevator_vg_seats(stab_poly: Polygon, x_row: float, pitch: float,
                      h: float) -> list[tuple[float, float, float, float, str]]:
    """Seats for the stab UNDERSIDE row (the flare suction side [IMP74]):
    pairs along both elevator spans, vane base sunk 15% of h into the skin."""
    seats = []
    z_in, z_out = STAB_HINGE_Z[0] + 0.6, STAB_HINGE_Z[1] - 0.6
    n_pairs = int((z_out - z_in) / pitch)
    z0 = z_in + 0.5 * ((z_out - z_in) - (n_pairs - 1) * pitch)
    for side in (+1.0, -1.0):
        for p in range(n_pairs):
            zp = z0 + p * pitch
            for sgn in (+1.0, -1.0):
                z = side * (zp + sgn * pitch / 4.0)
                lo, hi = chord_interval(stab_poly, z, horizontal=True)
                yt = naca0010_half((x_row - lo) / (hi - lo)) * (hi - lo)
                seats.append((x_row, -yt + 0.15 * h, z, sgn, "y"))
    return seats


def rudder_vg_seats(fin_poly: Polygon, dist_ahead: float, pitch: float,
                    h: float) -> list[tuple[float, float, float, float, str]]:
    """Seats for the fin rows, BOTH sides (no published rudder row -- this
    mirrors the Strausak elevator convention; flagged on every run)."""
    seats = []
    y_lo, y_hi = FIN_ROOT_WL + 1.2, RUD_HINGE_P1[1] - 0.8
    # Stations every `pitch` of hinge length, vane LE dist_ahead (chordwise
    # equivalent) forward of the raked hinge at constant waterline.
    dx = dist_ahead / math.cos(_RAKE)
    t0 = (y_lo - RUD_HINGE_P0[1]) / math.cos(_RAKE)
    t1 = (y_hi - RUD_HINGE_P0[1]) / math.cos(_RAKE)
    n_pairs = int((t1 - t0) / pitch)
    tb = t0 + 0.5 * ((t1 - t0) - (n_pairs - 1) * pitch)
    for side in (+1.0, -1.0):
        for p in range(n_pairs):
            for sgn in (+1.0, -1.0):
                t = tb + p * pitch + sgn * pitch / 4.0
                y = RUD_HINGE_P0[1] + t * math.cos(_RAKE)
                x = rud_hinge_x(y) - dx
                lo, hi = chord_interval(fin_poly, y, horizontal=True)
                zt = naca0010_half((x - lo) / (hi - lo)) * (hi - lo)
                seats.append((x, y, side * (zt - 0.15 * h), sgn, "z"))
    return seats


# =============================================================================
#  Assembly
# =============================================================================

def build_assembly(elev_deg: float, rud_deg: float, tag: str,
                   vg_height_mm: float = 0.0, vg_pitch_mm: float = 0.0,
                   vane_thick_mm: float = 0.0) -> Path:
    """One full-empennage article -> assets/<tag>.stl (binary, 4 shells)."""
    ac = load_aircraft(YAML)
    gap = ac.horizontal_tail.elevator.hinge_gap / IN     # 1/16 in, in inches

    stab_poly = stab_planform()
    fin_poly = fin_outline()

    # ---- raw solids ----------------------------------------------------------
    stab = loft_stab()
    fin = loft_fin()
    fuse = loft_fuselage()

    # ---- elevator split (local stab coords, waterline y=0) -------------------
    ctrl_r = elev_ctrl_pts()
    ctrl_l = [(x, -z) for x, z in ctrl_r]
    cut_r = elev_ctrl_pts(fixed_cut=True)            # extra horn-rib relief
    cut_l = [(x, -z) for x, z in cut_r]
    elev_r = trimesh.boolean.intersection([stab, prism(ctrl_r, 0.0, -8, 8, "xz")])
    elev_l = trimesh.boolean.intersection([stab, prism(ctrl_l, 0.0, -8, 8, "xz")])
    stab_fixed = trimesh.boolean.difference(
        [stab, prism(cut_r, gap, -8, 8, "xz"), prism(cut_l, gap, -8, 8, "xz")])

    # Elevator clearance coves: hinge-centered frusta over the hinge spans
    # only (the center carry-through and the horn wedge never face a swung
    # nose, see the 2.5D analysis in geometry/stl_gen.py).
    # The cove runs TWO gaps past each hinge end: the swung nose face is
    # axially coplanar with a cove that stops exactly at the hinge span, so
    # the end corners (center-strip rib, tip-wedge rib) must be relieved
    # too -- as they are on the real airplane. The radius is sized for the
    # nose's swing at 30 deg (control skin points slightly aft of the cut
    # reach radius ~ yt*sec(defl) when rotated -- the same r_fwd logic as
    # split_at_hinge in geometry/stl_gen.py), so one cove serves the whole
    # deflection family without pinching below the nominal gap.
    sec30 = 1.0 / math.cos(math.radians(30.0))
    zs = np.linspace(STAB_HINGE_Z[0] - 2 * gap, STAB_HINGE_Z[1] + 2 * gap, 17)
    chords = np.interp(np.abs(zs), [0.0, STAB_HINGE_Z[1]],
                       [STAB_TE_CL - STAB_LE_CL, 17.817])
    les = np.interp(np.abs(zs), [0.0, 50.627], [STAB_LE_CL, STAB_LE_TIP[0]])
    radii = (naca0010_half((STAB_HINGE_X - les) / chords) * chords * sec30
             + gap + 0.04)
    for sd in (+1.0, -1.0):
        cove = frustum_along(np.array([STAB_HINGE_X, 0.0, 0.0]),
                             np.array([0.0, 0.0, sd]), zs, radii)
        stab_fixed = trimesh.boolean.difference([stab_fixed, cove])

    # ---- rudder split (absolute coords) ---------------------------------------
    ctrl_rud = rud_ctrl_pts()
    rudder = trimesh.boolean.intersection([fin, prism(ctrl_rud, 0.0, -7, 7, "xy")])
    fin_fixed = trimesh.boolean.difference([fin, prism(ctrl_rud, gap, -7, 7, "xy")])

    # The fixed fin exists only above the root rib: everything below is
    # fuselage (carrying the virtual section down was tried and pokes a
    # NACA-thick tongue out of the slender stern, into the elevator roots).
    below = trimesh.creation.box(extents=(300.0, 100.0, 60.0))
    below.apply_translation((220.0, FIN_ROOT_WL - 0.05 - 50.0, 0.0))
    fin_fixed = trimesh.boolean.difference([fin_fixed, below])

    # Rudder cove along the raked hinge axis, bottom corner to horn start.
    # 33 stations: the radius rises steeply along the upsweep (the local
    # rudder chord grows fast off the bottom corner), so a coarse polyline
    # would sag below the swing radius of the rudder nose and pinch the gap.
    p0 = np.array([RUD_HINGE_P0[0], RUD_HINGE_P0[1], 0.0])
    ts = np.linspace(-2 * gap, np.linalg.norm(
        np.subtract(RUD_HINGE_P1, RUD_HINGE_P0)) + 2 * gap, 33)
    r_cove = []
    for t in ts:
        y = RUD_HINGE_P0[1] + t * math.cos(_RAKE)
        if y >= FIN_ROOT_WL:
            lo, hi = chord_interval(fin_poly, y, horizontal=True)
        else:
            lo, hi, _ = lower_fin_section(y)
        frac = min(max((rud_hinge_x(y) - lo) / max(hi - lo, 0.05), 0.0), 1.0)
        # Same 30-deg swing sizing as the elevator coves above.
        r_cove.append(naca0010_half(frac) * (hi - lo)
                      / math.cos(math.radians(30.0)) + gap + 0.04)
    cove_rud = frustum_along(p0, RUD_DIR, ts, np.array(r_cove))

    # ---- stern-post cut: fuselage (and the stab carry-through poking out of
    # it) end one gap forward of the raked hinge plane -- the same open gap
    # the rudder sees, continued down the post, exactly as built ------------
    n_hat = np.array([math.cos(_RAKE), -math.sin(_RAKE), 0.0])
    box = trimesh.creation.box(extents=(200.0, 300.0, 60.0))
    rot = trimesh.transformations.rotation_matrix(-_RAKE, (0, 0, 1))
    box.apply_transform(rot)
    box.apply_translation(p0 - gap * n_hat + 100.0 * n_hat)

    # ---- deflections (before any translation games) ---------------------------
    # Elevator: positive = TE down -> negative rotation about +z (stl_gen
    # convention, single encoding point). Both halves move together.
    if abs(elev_deg) > 0.0:
        T = trimesh.transformations.rotation_matrix(
            -math.radians(elev_deg), (0, 0, 1), (STAB_HINGE_X, 0.0, 0.0))
        elev_r.apply_transform(T)
        elev_l.apply_transform(T)
    # Rudder: positive = TE toward +z -> negative rotation about the raked
    # hinge direction (right-hand rule check: +rot about RUD_DIR sends the
    # TE toward -z).
    if abs(rud_deg) > 0.0:
        T = trimesh.transformations.rotation_matrix(
            -math.radians(rud_deg), RUD_DIR, p0)
        rudder.apply_transform(T)

    # ---- lift the stab family to its waterline --------------------------------
    for m in (stab_fixed, elev_r, elev_l):
        m.apply_translation((0.0, STAB_WL, 0.0))

    # ---- optional VG rows on the fixed elements -------------------------------
    vanes: list[trimesh.Trimesh] = []
    if vg_height_mm > 0.0:
        h = vg_height_mm / 25.4
        pitch = (vg_pitch_mm if vg_pitch_mm > 0.0
                 else ac.vg_defaults.elevator.spacing / UNIT_TO_SI["mm"]) / 25.4
        beta = ac.vg_defaults.vane_incidence
        thick = (vane_thick_mm / 25.4 if vane_thick_mm > 0.0
                 else max(0.0015 / IN, h / 8.0))
        dist = ac.vg_defaults.elevator.distance_ahead_of_hinge / IN
        seats_e = elevator_vg_seats(stab_poly, STAB_HINGE_X - dist, pitch, h)
        seats_r = rudder_vg_seats(fin_poly, dist, pitch, h)
        vanes = vane_boxes(h, 3.0 * h, thick, beta,
                           [(x, y + STAB_WL, z, s, ax)
                            for (x, y, z, s, ax) in seats_e])
        vanes += vane_boxes(h, 3.0 * h, thick, beta, seats_r)
        print(f"  vg: {len(seats_e)} elevator vanes (underside) + "
              f"{len(seats_r)} rudder vanes (both sides, elevator-convention "
              f"ANALOG, no flight-test source), h={vg_height_mm:g} mm, pitch "
              f"{pitch * 25.4:.0f} mm, t={thick * 25.4:.2f} mm")

    # ---- fixed union, then the stern/cove reliefs ------------------------------
    fixed = trimesh.boolean.union([fuse, fin_fixed, stab_fixed] + vanes)
    fixed = trimesh.boolean.difference([fixed, box, cove_rud])

    # Scrub boolean debris: a grazing cut occasionally leaves a degenerate
    # zero-volume sliver floating in a gap channel; every shell here is one
    # connected body by construction, so keep only the dominant component.
    def scrub(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        parts = mesh.split(only_watertight=False)
        return max(parts, key=lambda p: len(p.faces)) if len(parts) else mesh

    # ---- export: 4 disjoint closed shells, meters, origin at stab hinge/WL ----
    shells = [scrub(s) for s in (fixed, elev_r, elev_l, rudder)]
    out = ASSETS / f"{tag}.stl"
    combined = trimesh.util.concatenate(shells)
    combined.apply_scale(IN)
    combined.apply_translation((-STAB_HINGE_X * IN, -STAB_WL * IN, 0.0))
    combined.export(out)
    wt = [s.is_watertight for s in shells]
    ext = combined.extents
    print(f"wrote {out.name}: elev {elev_deg:+.0f} deg, rud {rud_deg:+.0f} "
          f"deg | shells watertight={wt} faces={len(combined.faces)} | "
          f"bbox {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m")
    return out


# =============================================================================
#  CLI
# =============================================================================

def single_tag(elev_deg: float, rud_deg: float) -> str:
    """Canonical asset tag for one clean article at signed deflections.

    Shared naming contract with the GUI launcher (tunnel_gui.py predicts the
    file name to decide whether it must generate before launching):
    en/eu15/ed15 for the elevator (u = TE up = nose-up command), rn/rd15 for
    the rudder; magnitudes formatted %g so 15.0 -> '15'.
    """
    e = "en" if elev_deg == 0 else (f"eu{-elev_deg:g}" if elev_deg < 0
                                    else f"ed{elev_deg:g}")
    r = "rn" if rud_deg == 0 else f"rd{abs(rud_deg):g}"
    return f"tail_asm_clean_{e}_{r}"


def main() -> None:
    ap = argparse.ArgumentParser(description="full empennage assembly articles")
    ap.add_argument("--elev-deg", type=float, default=15.0)
    ap.add_argument("--rud-deg", type=float, default=15.0)
    ap.add_argument("--vg-height-mm", type=float, default=10.0)
    ap.add_argument("--vg-pitch-mm", type=float, default=0.0,
                    help="pair pitch in mm (0 = aircraft.yaml elevator 30)")
    ap.add_argument("--vane-thickness-mm", type=float, default=0.0,
                    help="0 = physical; use >= 1.5x lattice cell for coarse runs")
    ap.add_argument("--no-vg", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--only", type=str, default="",
                    help="build just the article whose tag contains this text")
    ap.add_argument("--single", nargs=2, type=float, metavar=("ELEV", "RUD"),
                    default=None,
                    help="build exactly ONE clean article at these signed "
                         "deflections (deg; elevator + = TE down, rudder "
                         "+ = TE toward +z) -- the GUI launcher's on-demand "
                         "path; VG rows come from analytic tunnel stamping, "
                         "not the STL, so no VG variant is baked here")
    a = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("[tail_assembly] NOTE: tail sections are the NACA 0010 placeholder "
          "(aircraft.yaml TODO); planforms/hinges are drawing-true [DXF].")

    if a.single is not None:
        de, dr = a.single
        build_assembly(de, dr, single_tag(de, dr))
        return

    e, r = a.elev_deg, a.rud_deg
    cases = [
        (0.0, 0.0, "en_rn"),                 # baseline
        (-e, 0.0, f"eu{e:g}_rn"),            # NOSE-UP command (TE up)
        (+e, 0.0, f"ed{e:g}_rn"),            # nose-down
        (0.0, +r, f"en_rd{r:g}"),            # rudder one side (symmetric)
    ]
    families = [("clean", 0.0)]
    if not a.no_vg and a.vg_height_mm > 0.0:
        pitch_mm = a.vg_pitch_mm if a.vg_pitch_mm > 0.0 else 30.0
        families.append((f"vg{a.vg_height_mm:g}p{pitch_mm:g}", a.vg_height_mm))

    written = []
    for fam, vg_h in families:
        for de, dr, ctag in cases:
            tag = f"tail_asm_{fam}_{ctag}"
            if a.only and a.only not in tag:
                continue
            written.append(build_assembly(
                de, dr, tag, vg_height_mm=vg_h, vg_pitch_mm=a.vg_pitch_mm,
                vane_thick_mm=a.vane_thickness_mm))

    if not a.no_render:
        for stl in written:
            render_preview(stl)


if __name__ == "__main__":
    main()
