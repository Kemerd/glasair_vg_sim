# -*- coding: utf-8 -*-
"""
RapidCFD (GPU OpenFOAM 2.3 fork) case factory for the Glasair wing VG study.

WHAT THIS BUILDS
  Span-periodic slice cases of the aileron-station wing section with NO
  aileron gap (the clean single-solid article from geometry/stl_gen.py,
  include_gap=False equivalent) plus variants wearing one counter-rotating
  vortex-generator pair, meshed with stock OpenFOAM v2506 snappyHexMesh on
  the CPU and solved by RapidCFD's simpleFoam on the RTX 5090.

DOMAIN STRATEGY (one VG pitch, mirror walls)
  A counter-rotating VG array with pair pitch P is mirror-symmetric about
  the planes midway between pairs AND about each pair's own centerline, so
  a slice exactly one pitch wide with symmetry (slip) side walls reproduces
  the infinite array EXACTLY for the steady mean flow - no cyclic/AMI
  machinery, which keeps every boundary type on RapidCFD's well-trodden
  GPU path. The vane pair sits centered: vanes at z = +/- P/4, toe-out.

  The wing solid is extruded 20% wider than the domain so it pierces the
  side planes cleanly (snappy clips it; no degenerate slivers at the
  corners). Angle of attack is baked into the STL (geometry rotated about
  the quarter chord) so the far-field box stays axis-aligned and the
  force decomposition is simply lift = +y, drag = +x.

MESH FAMILY PARITY
  Clean and VG cases share the identical background mesh, boxes, surface
  levels and layer recipe; the only difference is the vane-distance
  refinement band (meaningless on a vaneless article). Case-to-case deltas
  therefore ride on near-identical mesh families.

PRECISION NOTE
  RapidCFD is built WM_PRECISION_OPTION=SP (fp32 fields - full bandwidth
  on the GeForce, whose native fp64 runs at 1/64 rate) with the in-house
  patch that emulates fp64 where it actually matters: every solver-side
  global reduction (CG dot products, residual norms, norm factors)
  accumulates in double on the GPU. See docs/BACKLOG.md and the patched
  RapidCFD tree in WSL (~/RapidCFD-dev).

OUTPUTS
  gpu/rapidcfd/assets/<article>.stl        rotated wall + vane solids
  gpu/rapidcfd/assets/<article>_vanes.stl  vanes-only (refinement source)
  gpu/rapidcfd/cases/<case>/               complete OpenFOAM case dirs
  gpu/rapidcfd/run_all.sh                  WSL-side mesh+solve driver

Run:  python gpu/rapidcfd/build_cases.py
"""
from __future__ import annotations

import math
import shutil
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

HERE = REPO / "gpu" / "rapidcfd"
ASSETS = HERE / "assets"
CASES = HERE / "cases"

# ---------------------------------------------------------------------------
#  Flow & study constants
# ---------------------------------------------------------------------------
NU = 1.48e-05            # m^2/s, standard air (matches the M1 pipeline)
RE = 2.2e6               # stall-regime chord Reynolds (80 mph @ 0.9022 m)
SPAN_OVERHANG = 1.2      # STL span / domain span: pierce the side planes

# Each study case is a dict so fields can be added without re-numbering
# positional tuples. `case()` fills the defaults; only the knobs that differ
# from the study baseline need to be named per row.
#   name   : case + results dir name
#   h_mm   : vane height in mm, or None for the clean wing
#   alpha  : geometric angle of attack, degrees (baked into the STL)
#   x_frac : chord station of the vane row (None -> IMP74 0.07)
#   re     : chord Reynolds (None -> study default 2.2e6 stall regime)
#   shape  : "rect" flat plate (default) or "delta" triangular ramp
#   toe    : "out" splayed LEs (IMP74 default) or "in" converged LEs
#   pitch_mm: VG-to-VG spacing in mm (None -> IMP74 50 mm); also sets the
#             periodic slice width so the array stays consistent
#   count   : "pair" counter-rotating pair per pitch (IMP74 default) or
#             "single" one fin per pitch alternating yaw (Stolspeed pattern);
#             single doubles the periodic slice to 2*pitch so the alternation
#             is captured by the cyclic BCs
#   beta_deg: vane incidence to the flow (None -> IMP74/Stolspeed 15 deg);
#             the angle-sweep cases set 5/10/20 to find the best incidence
def case(name, *, h_mm=None, alpha=18.0, x_frac=None, re=None,
         shape="rect", toe="out", pitch_mm=None, count="pair", beta_deg=None):
    return dict(name=name, h_mm=h_mm, alpha=alpha, x_frac=x_frac, re=re,
                shape=shape, toe=toe, pitch_mm=pitch_mm, count=count,
                beta_deg=beta_deg)


CASE_MATRIX = [
    # --- original 7: sanity + alpha-18 placement/height sweep ----------------
    case("clean_a08", alpha=8.0),
    case("clean_a18", alpha=18.0),
    case("vg12p50_a18", h_mm=12.0),
    case("vg16p50_a18", h_mm=16.0),
    case("vg12x15_a18", h_mm=12.0, x_frac=0.15),
    case("vg12x30_a18", h_mm=12.0, x_frac=0.30),
    case("vg12x45_a18", h_mm=12.0, x_frac=0.45),
    # --- batch 2: cruise drag tax (200 mph) + stall-onset (16 deg) + delta ---
    case("clean_a02", alpha=2.0, re=5.52e6),
    case("vg12p50_a02", h_mm=12.0, alpha=2.0, re=5.52e6),
    case("clean_a16", alpha=16.0),
    case("vg12p50_a16", h_mm=12.0, alpha=16.0),
    case("vg12d50_a18", h_mm=12.0, shape="delta"),
    # --- batch 3: overnight optima search around the alpha-18 winner ---------
    # toe-in vs the toe-out winner (direct A/B the user asked for)
    case("vg12p50i_a18", h_mm=12.0, toe="in"),
    # delta + toe-in (does the better planform prefer the other toe sense?)
    case("vg12d50i_a18", h_mm=12.0, shape="delta", toe="in"),
    # spacing sweep at the winning height/station: tighter and wider pitch
    case("vg12p35_a18", h_mm=12.0, pitch_mm=35.0),
    case("vg12p70_a18", h_mm=12.0, pitch_mm=70.0),
    # height fine-step between the 12 mm winner and the 16 mm spoiler
    case("vg10p50_a18", h_mm=10.0),
    # --- batch 4: Stolspeed philosophy vs IMP74 (ref/Smokey__Clear.jpg) ------
    # JG's two claims to settle on the GPU:
    #  (1) single alternating fins make ~zero cruise drag where counter-
    #      rotating pairs make drag -> test single rect at 18deg AND cruise;
    #  (2) the swept rounded-LE Stolspeed fin makes a tighter, lower-drag
    #      vortex than a flat plate -> test the stol planform at 18deg.
    # Stall-recovery head-to-head at the alpha-18 decision point:
    case("vg12s50_a18", h_mm=12.0, shape="stol"),               # swept fin, pair
    case("vg12single_a18", h_mm=12.0, count="single"),          # single rect alt
    case("vg12ssingle_a18", h_mm=12.0, shape="stol", count="single"),  # both
    # Cruise drag-tax A/B: does single alternating really beat the pair?
    case("vg12single_a02", h_mm=12.0, alpha=2.0, re=5.52e6, count="single"),
    case("vg12s50_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="stol"),
    # --- batch 5: incidence-angle sweep on the DELTA winner @ alpha 18 -------
    # 15deg is vg12d50_a18 (done). Bracket the effective 10-20 band plus a
    # shallow 5deg low-end probe to confirm 15 is near-optimal, not just the
    # inherited default. Best (spacing, angle) falls out of batch3 + this.
    case("vg12d50b05_a18", h_mm=12.0, shape="delta", beta_deg=5.0),
    case("vg12d50b10_a18", h_mm=12.0, shape="delta", beta_deg=10.0),
    case("vg12d50b20_a18", h_mm=12.0, shape="delta", beta_deg=20.0),
    # --- batch 6: converge on the delta@beta10 optimum + the CRUISE TAX ------
    # Leaders so far: delta beta10 (best Cd) and stol fin (best Cl, steady);
    # wider pitch (70mm) and shallower beta both helped. This wave nails the
    # 2D optimum (pitch x beta) on the delta AND measures the cruise tax of
    # every stall contender so we can rank by stall-win-per-cruise-cost.
    # Pitch on the delta@beta10 (70mm was great for rect - is delta same?):
    case("vg12d70b10_a18", h_mm=12.0, shape="delta", beta_deg=10.0, pitch_mm=70.0),
    case("vg12d60b10_a18", h_mm=12.0, shape="delta", beta_deg=10.0, pitch_mm=60.0),
    # Even shallower / mid incidence on the delta to find the beta minimum:
    case("vg12d50b08_a18", h_mm=12.0, shape="delta", beta_deg=8.0),
    case("vg12d50b12_a18", h_mm=12.0, shape="delta", beta_deg=12.0),
    # CRUISE TAX of the stall leaders (alpha=2, Re5.52e6) - the tradeoff axis:
    case("vg12d50b10_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="delta", beta_deg=10.0),
    case("vg12d70b10_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="delta", beta_deg=10.0, pitch_mm=70.0),
    case("vg12ssingle_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="stol", count="single"),
    # --- batch 7: SHORT micro-VG (the cruise-friendly bet from research) -----
    # Lin's insight: a VG ~0.2-0.5x BL height recovers separation for a
    # FRACTION of the cruise drag. Our BL ~3-6mm at 9%c, so 6-8mm delta sits
    # mostly inside the stall BL but barely pokes the thin cruise BL = stall
    # win for near-zero cruise cost. Test 6/8mm delta@beta10 at BOTH alphas.
    case("vg06d50b10_a18", h_mm=6.0, shape="delta", beta_deg=10.0),
    case("vg08d50b10_a18", h_mm=8.0, shape="delta", beta_deg=10.0),
    case("vg08d50b10_a02", h_mm=8.0, alpha=2.0, re=5.52e6, shape="delta", beta_deg=10.0),
    # --- batch 8 (Wave E): "crazy" emerging-research planforms ---------------
    # From the 2026-06-14 design research (see vg-research-emerging-designs):
    #  - TRAPEZOID (cropped delta, l=4h): the ICAS-2020 / V-22 convergence
    #    shape, claims best vortex persistence; our top stall hope.
    #  - GOTHIC (concave swept LE): highest Clmax in lit BUT only <14deg AoA -
    #    we are at 18, so this is a "verify, don't assume" probe.
    # Run each at the current-best settings (beta10, 50mm) at stall, plus the
    # trapezoid cruise tax since it is the leading stall candidate.
    case("vg12t50b10_a18", h_mm=12.0, shape="trap", beta_deg=10.0),
    case("vg12g50b10_a18", h_mm=12.0, shape="gothic", beta_deg=10.0),
    case("vg12t50b10_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="trap", beta_deg=10.0),
    # Trapezoid at the winning wide pitch too (70mm helped every other shape):
    case("vg12t70b10_a18", h_mm=12.0, shape="trap", beta_deg=10.0, pitch_mm=70.0),
    # --- batch 9 (Wave E cont.): AIRFOIL-SECTION cambered vane ---------------
    # Best cruise-vs-stall planform on paper: cambered section = stronger
    # vortex per height at LOWER device drag (no sharp-edge self-separation).
    # Test at stall and cruise; if it holds the delta's stall win at a smaller
    # cruise tax, this is the cruise-friendly champion.
    case("vg12a50b10_a18", h_mm=12.0, shape="airfoil", beta_deg=10.0),
    case("vg12a50b10_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="airfoil", beta_deg=10.0),
    # --- batch 10 (Wave G): BEST-OF-BOTH combos ------------------------------
    # The data points to single-alternating (cuts cruise tax to +34%) + beta10
    # (best stall) + wide pitch (70mm) as the untested optimum. Test it on the
    # two best shapes (delta, swept) at BOTH alphas to find the config that
    # recovers stall for the smallest cruise cost. Single-alt -> 2-pitch slab.
    case("vg12dsingle70b10_a18", h_mm=12.0, shape="delta", count="single", beta_deg=10.0, pitch_mm=70.0),
    case("vg12dsingle70b10_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="delta", count="single", beta_deg=10.0, pitch_mm=70.0),
    case("vg12ssingle70b10_a18", h_mm=12.0, shape="stol", count="single", beta_deg=10.0, pitch_mm=70.0),
    case("vg12ssingle70b10_a02", h_mm=12.0, alpha=2.0, re=5.52e6, shape="stol", count="single", beta_deg=10.0, pitch_mm=70.0),
]


# ---------------------------------------------------------------------------
#  Geometry: clean / VG'd slice articles, alpha baked in
# ---------------------------------------------------------------------------

def upper_surface_point(coords: np.ndarray, x_frac: float) -> tuple[float, float]:
    """(y/c, local slope) of the upper surface at chordwise station x_frac.

    Same construction as gpu/fluidx3d/make_vg_wing.py: the resampled loop is
    Selig-ordered (TE-upper -> LE -> TE-lower) so the upper surface is the
    first half, reversed to ascending x for interpolation.
    """
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]
    y = float(np.interp(x_frac, upper[:, 0], upper[:, 1]))
    i = int(np.searchsorted(upper[:, 0], x_frac))
    i = max(1, min(i, len(upper) - 1))
    dx, dy = (upper[i] - upper[i - 1])
    return y, float(np.arctan2(dy, dx))


def make_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One rectangular vane plate: LE at origin, +x along the plate, +y up."""
    box = trimesh.creation.box(extents=(length, h, thick))
    box.apply_translation((length / 2.0, h / 2.0, 0.0))
    return box


def make_delta_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One triangular ("delta") vane plate in the same frame as make_vane:
    LE (apex) at the origin, +x along the plate, +y up.

    The planform ramps from zero height at the apex to the full height h at
    the vane trailing edge - the shape most retrofit VG kits ship. Built as
    an explicit 6-vertex prism (two triangular caps + three quad sides)
    rather than via extrude_polygon so there is no shapely dependency.
    """
    t = thick / 2.0
    verts = np.array([
        [0.0, 0.0, -t], [length, 0.0, -t], [length, h, -t],   # cap z = -t
        [0.0, 0.0, +t], [length, 0.0, +t], [length, h, +t],   # cap z = +t
    ])
    faces = np.array([
        [0, 2, 1], [3, 4, 5],          # the two triangular caps
        [0, 1, 4], [0, 4, 3],          # bottom edge (sits on / in the skin)
        [1, 2, 5], [1, 5, 4],          # vertical trailing edge
        [2, 0, 3], [2, 3, 5],          # ramped hypotenuse (the "delta" edge)
    ])
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    tri.fix_normals()                  # guarantee outward, consistent winding
    return tri


def make_stolspeed_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One Stolspeed-style fin in the same frame as make_vane: base on the
    z=0..(thin) skin plane, +x downstream, +y up, LE (apex) at the origin.

    Captures the design JG describes in ref/Smokey__Clear.jpg + the Stolspeed
    "Design" writeup, the features he says matter for a tight low-drag vortex:
      * a SWEPT, ROUNDED leading edge - the fin rises from near-zero height at
        the apex to full height h toward the rear along a smooth curve, so the
        airflow "progressively spills over the leading edge" instead of
        dumping off a square corner;
      * a TAPERED top (rounded, not a sharp blade) - the upper edge eases back
        down past the peak so there is no long flat blade trailing;
      * a SLIM fin (the same physical thickness as the other vanes for a fair
        A/B), no close-pair companion (the matrix decides single vs pair).

    Built as a thin extrusion of a 2D fin OUTLINE in the (x,y) plane swept
    +/- thick/2 in z. The outline is a polyline so the LE/TE curvature is
    explicit and reproducible; no shapely needed - we triangulate the closed
    loop as a fan from the base-front apex (the loop is convex enough that a
    fan is valid, and process=True repairs any sliver).
    """
    # Fin outline, fractions of (length, h). x runs 0..1 of length, y 0..1 of
    # h. Start at the base apex, sweep UP-and-BACK along the rounded LE to the
    # peak, then DOWN-and-BACK along the tapered top to the base trailing
    # edge, and close along the base. Peak sits ~70% aft (raked-back fin).
    n = 14
    le = np.array([[                              # rounded, swept leading edge
        0.62 * (1 - math.cos(0.5 * math.pi * s)),   # x: eased back
        1.00 * math.sin(0.5 * math.pi * s),         # y: rises to the peak
    ] for s in np.linspace(0.0, 1.0, n)])
    te = np.array([[                              # tapered top down to base TE
        0.62 + 0.38 * (s ** 0.7),                   # x: peak(0.62) -> 1.0
        1.00 * (1.0 - s) ** 1.4,                    # y: peak -> 0, eased
    ] for s in np.linspace(0.0, 1.0, n)])
    outline = np.vstack([le, te[1:]])             # (2n-1, 2) open chain
    outline[:, 0] *= length
    outline[:, 1] *= h

    # Triangulate the closed loop (outline + base segment back to apex) as a
    # fan from the first vertex; sweep to a thin 3D prism.
    m = len(outline)
    t = thick / 2.0
    verts = np.vstack([
        np.column_stack([outline, np.full(m, -t)]),   # z = -t face
        np.column_stack([outline, np.full(m, +t)]),   # z = +t face
    ])
    faces = []
    for i in range(1, m - 1):                     # the two flat fin caps
        faces.append([0, i + 1, i])               # -t cap (CW seen from -z)
        faces.append([m, m + i, m + i + 1])       # +t cap
    for i in range(m):                            # the thin rim around the fin
        j = (i + 1) % m
        faces.append([i, j, m + j])
        faces.append([i, m + j, m + i])
    fin = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    fin.fix_normals()
    return fin


def _extrude_outline(outline: np.ndarray, thick: float) -> trimesh.Trimesh:
    """Thin-prism extrusion of a 2D (x,y) vane OUTLINE swept +/- thick/2 in z.

    Shared by the curved-planform builders (trapezoid, gothic, ...). `outline`
    is an open vertex chain in the (x,y) plane already scaled to meters; the
    closing base segment (last vertex back to the first) is implied. The two
    flat caps are fanned from vertex 0 and the rim is stitched as quads. Any
    sliver from the fan on a mildly non-convex loop is repaired by process=True.
    """
    m = len(outline)
    t = thick / 2.0
    verts = np.vstack([
        np.column_stack([outline, np.full(m, -t)]),
        np.column_stack([outline, np.full(m, +t)]),
    ])
    faces = []
    for i in range(1, m - 1):                     # the two flat caps
        faces.append([0, i + 1, i])
        faces.append([m, m + i, m + i + 1])
    for i in range(m):                            # the thin rim
        j = (i + 1) % m
        faces.append([i, j, m + j])
        faces.append([i, m + j, m + i])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def make_trapezoid_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One TRAPEZOIDAL (cropped-delta) vane in the make_vane frame.

    The geometry two studies closest to our setup converge on (ICAS 2020,
    fixed-wing V-22): a delta with the apex cropped flat, so the vane has a
    short raked LE rising to full height, a flat top, and a vertical TE. The
    flat top sustains the streamwise vortex farther downstream than a sharp
    triangle (~+10% vortex persistence in the literature). LE crop sits at
    ~25% of length; top runs flat from there to the TE.
    """
    crop = 0.25                                   # apex cropped at 25% length
    outline = np.array([
        [0.0, 0.0],                               # base LE (on the skin)
        [crop * length, h],                       # raked LE up to full height
        [length, h],                              # flat top to the TE
        [length, 0.0],                            # vertical trailing edge
    ])
    return _extrude_outline(outline, thick)


def make_gothic_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One GOTHIC-planform vane in the make_vane frame.

    A rectangle whose leading edge is swept back along a CONCAVE (gothic) arc
    that blends into a vertical trailing edge - area between a triangle and a
    rectangle. The curved swept LE sheds a tighter, more concentrated vortex
    that resists bursting better than a square LE. Built from a sampled arc:
    the LE rises from the base apex to full height following y = (s)^p with
    p<1 (concave-up, gothic), then a flat top and vertical TE.
    """
    n = 12
    p = 0.55                                       # <1 => concave gothic sweep
    le = np.array([[                               # swept concave leading edge
        0.45 * (s ** 1.6) * length,                # x: eases back, bulk near top
        (s ** p) * h,                              # y: rises gothic to full h
    ] for s in np.linspace(0.0, 1.0, n)])
    outline = np.vstack([
        le,
        [[length, h], [length, 0.0]],              # flat top, vertical TE
    ])
    return _extrude_outline(outline, thick)


def make_airfoil_vane(h: float, length: float, thick: float) -> trimesh.Trimesh:
    """One AIRFOIL-SECTION vane: a rectangular planform (length x h) whose
    CROSS-SECTION in the chordwise(x)-thickness(z) plane is a cambered airfoil
    instead of a flat plate. Same make_vane frame: base at y=0 on the skin,
    +x downstream, +y up; the section's chord runs along +x, camber bows in z.

    Research (DTU/KTH) shows a cambered-section vane makes a STRONGER downstream
    vortex than a flat plate of equal height while shedding LESS device drag
    (no sharp-edge self-separation) - the best cruise-vs-stall planform on
    paper. Modeled with a NACA-4-digit-style camber line + thickness so the
    incidence applied later (beta) sets the section's angle to the local flow.

    Built by lofting the same 2D airfoil section at y=0 and y=h (constant
    section, rectangular planform) and stitching the two loops into a solid.
    """
    # NACA-4-style section: max camber m at position pcm, thickness tc, all as
    # fractions of the section chord (= length). Modest camber for a vane.
    m, pcm, tc = 0.04, 0.40, max(0.12, thick / length)
    n = 18
    xc = (1 - np.cos(np.linspace(0.0, math.pi, n))) / 2.0   # cosine-clustered

    # Mean camber line z_c and its slope (NACA 4-digit piecewise).
    zc = np.where(xc < pcm,
                  m / pcm**2 * (2 * pcm * xc - xc**2),
                  m / (1 - pcm)**2 * ((1 - 2 * pcm) + 2 * pcm * xc - xc**2))
    # Half-thickness distribution (NACA 4-digit).
    yt = 5 * tc * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                   + 0.2843 * xc**3 - 0.1015 * xc**4)
    upper = np.column_stack([xc, zc + yt])          # (x, z) upper surface
    lower = np.column_stack([xc, zc - yt])          # (x, z) lower surface
    # Closed section loop: upper LE->TE then lower TE->LE. At the LE (xc=0) and
    # TE (xc=1) the two surfaces meet, so drop the duplicate endpoints from the
    # lower run or the loop has degenerate zero-length edges (breaks the solid).
    sect = np.vstack([upper, lower[::-1][1:-1]])
    sect[:, 0] *= length                            # scale x to vane length
    sect[:, 1] *= length                            # scale z by length too

    k = len(sect)
    # Two copies of the section at y=0 (root, on the skin) and y=h (tip).
    root = np.column_stack([sect[:, 0], np.zeros(k), sect[:, 1]])
    tip = np.column_stack([sect[:, 0], np.full(k, h), sect[:, 1]])
    verts = np.vstack([root, tip])
    faces = []
    for i in range(k):                              # side wall (root->tip loft)
        j = (i + 1) % k
        faces.append([i, j, k + j])
        faces.append([i, k + j, k + i])
    for i in range(1, k - 1):                       # root cap and tip cap
        faces.append([0, i, i + 1])
        faces.append([k, k + i + 1, k + i])
    vane = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    vane.fix_normals()
    return vane


def build_article(name: str, h_mm: float | None, alpha_deg: float,
                  ac, coords: np.ndarray, domain_span: float,
                  x_frac_override: float | None = None,
                  shape: str = "rect", toe: str = "out",
                  pitch_override: float | None = None,
                  count: str = "pair",
                  beta_deg_override: float | None = None) -> tuple[Path, Path | None]:
    """Build one wall article (wing [+ vane pair]), rotated to alpha.

    Returns (wall_stl, vanes_stl-or-None). The vanes-only STL is exported
    separately because snappy uses it purely as a distance-refinement
    source; the wall solid is the boolean union so inside/outside stays
    unambiguous.
    """
    chord = ac.wing.aileron.chord_at_mid_station          # 0.9022 m [DXF]
    # Pitch (VG-to-VG spacing) is the study default unless a spacing-sweep
    # case overrides it. NOTE the slice WIDTH stays = domain_span (one pitch
    # of the periodic array), so a pitch override changes both the vane z
    # offset AND the extruded slab width via the caller.
    pitch = pitch_override if pitch_override is not None \
        else ac.vg_defaults.wing.spacing_outboard         # 0.050 m  [IMP74]
    # Chordwise station of the vane row: study default from IMP74 unless a
    # matrix entry overrides it (the placement-sweep cases).
    x_frac = (x_frac_override if x_frac_override is not None
              else ac.vg_defaults.wing.chord_position_frac)  # 0.07   [IMP74]
    # Vane incidence to the local flow. Study default 15 deg (IMP74 +
    # Stolspeed sweet spot); the angle-sweep cases override it to bracket the
    # effective 10-20 deg band (and a shallow 5 deg low-end probe).
    beta = (math.radians(beta_deg_override) if beta_deg_override is not None
            else ac.vg_defaults.vane_incidence)           # radians
    l_per_h = ac.vg_defaults.vane_length_per_height       # 3.0

    stl_span = SPAN_OVERHANG * domain_span

    # Clean gapless wing solid - the "no aileron gap" article. The blunt-TE
    # resampled loop and extrusion come from the validated M0 toolkit.
    wing = extrude_section(coords, chord, stl_span)

    vanes: list[trimesh.Trimesh] = []
    if h_mm is not None:
        h = h_mm / 1000.0
        # Trapezoidal vanes use the research-recommended l=4h (longer flat top
        # sustains the vortex); all other planforms use the IMP74 l=3h default.
        vane_l = (4.0 if shape == "trap" else l_per_h) * h
        vane_t = max(0.0015, h / 8.0)                     # physical plate
        y_surf, slope = upper_surface_point(coords, x_frac)
        x_le, y_le = x_frac * chord, y_surf * chord

        # Planform builder per the matrix: rectangular plate (study default),
        # triangular delta ramp, Stolspeed swept-LE fin, trapezoid (cropped
        # delta, the ICAS/V-22 convergence shape), or gothic (concave swept LE).
        vane_builder = {
            "delta": make_delta_vane,
            "stol": make_stolspeed_vane,
            "trap": make_trapezoid_vane,
            "gothic": make_gothic_vane,
            "airfoil": make_airfoil_vane,
        }.get(shape, make_vane)
        # Toe sense: "out" splays the leading edges apart (each vane yawed so
        # its LE points away from the slice center - the IMP74 default);
        # "in" converges them. The yaw sign is toe_sign * (per-vane sign).
        toe_sign = -1.0 if toe == "in" else +1.0

        # Vane LAYOUT:
        #  - "pair"   : a counter-rotating pair in ONE pitch (IMP74). The slice
        #               is one pitch wide; vanes sit at z = +/- pitch/4, yawed
        #               +/-beta. This is the study baseline.
        #  - "single" : the Stolspeed "single alternating" pattern. One fin per
        #               pitch, the NEXT pitch's fin yawed the OTHER way. To make
        #               that alternation periodic the slice must span TWO
        #               pitches with two opposite-yawed fins (the caller widens
        #               the slab to 2*pitch for these cases).
        if count == "single":
            # Two fins, one per pitch, opposite yaw, centered in each pitch.
            placements = [(+1.0, -pitch / 2.0), (-1.0, +pitch / 2.0)]
        else:
            placements = [(+1.0, +pitch / 4.0), (-1.0, -pitch / 4.0)]

        for sgn, z_off in placements:
            v = vane_builder(h, vane_l, vane_t)
            v.apply_transform(trimesh.transformations.rotation_matrix(
                toe_sign * sgn * beta, (0.0, 1.0, 0.0)))
            v.apply_transform(trimesh.transformations.rotation_matrix(
                slope, (0.0, 0.0, 1.0)))
            v.apply_translation((x_le, y_le - 0.10 * h, z_off))
            vanes.append(v)

        wall = trimesh.boolean.union([wing] + vanes)
    else:
        wall = wing

    # Bake the angle of attack into the geometry: nose-up alpha = rotate
    # -alpha about z (x points downstream, y up), about the quarter chord
    # so the moment reference is trivially (c/4, 0, 0) in case coordinates.
    rot = trimesh.transformations.rotation_matrix(
        -math.radians(alpha_deg), (0.0, 0.0, 1.0), (0.25 * chord, 0.0, 0.0))
    wall.apply_transform(rot)

    ASSETS.mkdir(parents=True, exist_ok=True)
    wall_path = ASSETS / f"{name}_wall.stl"
    wall.export(wall_path)
    print(f"[articles] {wall_path.name}: watertight={wall.is_watertight} "
          f"faces={len(wall.faces)} alpha={alpha_deg:g}deg")

    vanes_path = None
    if vanes:
        vblob = trimesh.util.concatenate(vanes)
        vblob.apply_transform(rot)
        vanes_path = ASSETS / f"{name}_vanes.stl"
        vblob.export(vanes_path)
        print(f"[articles] {vanes_path.name}: refinement source, "
              f"faces={len(vblob.faces)}")
    return wall_path, vanes_path


# ---------------------------------------------------------------------------
#  OpenFOAM dictionaries (2.3-compatible syntax throughout, so the same
#  case feeds v2506 meshing utilities AND the RapidCFD 2.3-era solver)
# ---------------------------------------------------------------------------

def foam_header(cls: str, obj: str, location: str = "system") -> str:
    """Standard FoamFile banner; version 2.0 dict format suits both vintages."""
    return f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{location}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


# Domain box, meters. x: -5c .. +10.5c, y: +/-5.5c, z: one VG pitch.
XMIN, XMAX = -4.5, 9.5
YMIN, YMAX = -5.0, 5.0
BASE = 0.1               # background cell size in x,y (z = pitch, 1 cell)


def block_mesh_dict(pitch: float) -> str:
    nx = round((XMAX - XMIN) / BASE)
    ny = round((YMAX - YMIN) / BASE)
    zmin, zmax = -pitch / 2.0, pitch / 2.0
    return foam_header("dictionary", "blockMeshDict") + f"""
// Background hex box for snappy: base {BASE} m cells in x/y and a single
// cell across the one-pitch span (snappy's refinement subdivides z too,
// reaching sub-mm at the vane band).
//
// Side boundaries are CYCLIC: true span periodicity is exact for the
// infinite VG array, and it deliberately avoids RapidCFD's transform
// patches (slip/symmetryPlane) whose 2.3-era PISO/SIMPLE algebra injects
// spurious tangential wall friction on the GPU (proven by the uniform-flow
// reproducer; modern OpenFOAM fixed this class of artifact via
// constrainHbyA). Top/bottom are freestream for the same reason - plain
// inletOutlet machinery, no transform path, and less blockage than slip.
convertToMeters 1;

vertices
(
    ({XMIN} {YMIN} {zmin})  ({XMAX} {YMIN} {zmin})
    ({XMAX} {YMAX} {zmin})  ({XMIN} {YMAX} {zmin})
    ({XMIN} {YMIN} {zmax})  ({XMAX} {YMIN} {zmax})
    ({XMAX} {YMAX} {zmax})  ({XMIN} {YMAX} {zmax})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5)); }}
    top      {{ type patch; faces ((3 7 6 2)); }}
    bottom   {{ type patch; faces ((0 1 5 4)); }}
    // Sides are meshed as PLAIN patches (snappy's layer extrusion aborts
    // when wall geometry pierces a cyclic boundary) and converted to a
    // translational cyclic pair afterwards by createPatch - the geometry
    // is z-extruded, so the two planes castellate/snap/layer identically
    // and the faces match by translation.
    sideLeft {{ type patch; faces ((0 3 2 1)); }}
    sideRight{{ type patch; faces ((4 5 6 7)); }}
);

mergePatchPairs ();
"""


def snappy_dict(has_vanes: bool) -> str:
    vane_geom = """
    vanes
    {
        type triSurfaceMesh;
        file "vanes.stl";
    }""" if has_vanes else ""
    vane_refine = """
        // Sub-mm band hugging the vane plates: L8 ~ 0.39 mm within 1.5 mm,
        // L7 within 4 mm - resolves plate-edge vortex roll-up, the actual
        // VG mechanism the LBM arm could never reach.
        vanes
        {
            mode distance;
            levels ((0.0015 8) (0.004 7));
        }""" if has_vanes else ""
    return foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    wing
    {{
        type triSurfaceMesh;
        file "wing.stl";
    }}{vane_geom}

    // Near-field and wake boxes (axis-aligned because alpha is baked into
    // the rotated STL, never into the domain).
    boxNear {{ type searchableBox; min (-0.5 -0.6 -1); max ( 1.4  0.6 1); }}
    boxWake {{ type searchableBox; min (-0.9 -1.0 -1); max ( 3.6  1.0 1); }}
}}

castellatedMeshControls
{{
    maxLocalCells       4000000;
    maxGlobalCells      12000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
        {{ file "wing.eMesh"; level 6; }}
    );

    refinementSurfaces
    {{
        wing
        {{
            // L5-L6 on the skin: 3.1 down to 1.56 mm in x/y (half that in
            // z, the background cell is flatter); curvature picks L6.
            level (5 6);
            patchInfo {{ type wall; }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        wing
        {{
            // Distance bands off the skin: keeps the boundary-layer
            // neighbourhood at L6 and the near-field at L5.
            mode distance;
            levels ((0.012 6) (0.05 5));
        }}{vane_refine}
        boxNear {{ mode inside; levels ((1e15 4)); }}
        boxWake {{ mode inside; levels ((1e15 3)); }}
    }}

    locationInMesh (-2.0 -1.0 0.0001);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes   false;

    layers
    {{
        // Layers grow on every wing-derived patch (vanes included; snappy
        // collapses them locally where the thin plates pinch).
        "wing.*"
        {{
            nSurfaceLayers 5;
        }}
    }}

    // Absolute sizing: first cell 0.4 mm puts y+ ~ 35-45 at Re 2.2e6 -
    // squarely in the wall-function log-layer window.
    firstLayerThickness 4.0e-4;
    expansionRatio      1.25;
    minThickness        5.0e-5;

    nGrow               0;
    featureAngle        130;
    slipFeatureAngle    30;
    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    // Both spellings: 2.x reads minMedianAxisAngle, v2506 minMedialAxisAngle;
    // each vintage ignores the key it does not know.
    minMedianAxisAngle  90;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;

    relaxed
    {{
        maxNonOrtho 75;
    }}
}}

debug 0;
mergeTolerance 1e-6;
"""


def create_patch_dict(pitch: float) -> str:
    return foam_header("dictionary", "createPatchDict") + f"""
// Post-snappy conversion of the side planes into a translational cyclic
// pair (see blockMeshDict header for why they are meshed as plain patches).
pointSync false;

patches
(
    {{
        name sideL;
        patchInfo
        {{
            type            cyclicAMI;
            neighbourPatch  sideR;
            transform       translational;
            separationVector (0 0 {pitch});
            matchTolerance  1e-3;
        }}
        constructFrom patches;
        patches (sideLeft);
    }}

    {{
        name sideR;
        patchInfo
        {{
            type            cyclicAMI;
            neighbourPatch  sideL;
            transform       translational;
            separationVector (0 0 {-pitch});
            matchTolerance  1e-3;
        }}
        constructFrom patches;
        patches (sideRight);
    }}
);
"""


def surface_features_dict() -> str:
    return foam_header("dictionary", "surfaceFeatureExtractDict") + """
wing.stl
{
    extractionMethod    extractFromSurface;

    extractFromSurfaceCoeffs
    {
        // 150 deg keeps the blunt TE corners and vane plate edges while
        // ignoring the gentle skin curvature.
        includedAngle   150;
    }

    writeObj            no;
}
"""


def control_dict(u_inf: float, chord: float, pitch: float, n_iter: int) -> str:
    aref = chord * pitch
    return foam_header("dictionary", "controlDict") + f"""
application     simpleFoam;

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {n_iter};
deltaT          1;

writeControl    timeStep;
writeInterval   1000;
purgeWrite      2;

writeFormat     ascii;
writePrecision  7;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable false;

functions
{{
    forceCoeffs1
    {{
        // 2.3-era function-object syntax (RapidCFD vintage). Geometry is
        // rotated to alpha, freestream is +x, so lift/drag axes are fixed.
        type            forceCoeffs;
        functionObjectLibs ("libforces.so");
        outputControl   timeStep;
        outputInterval  25;

        patches         (wing);
        pName           p;
        UName           U;
        rhoName         rhoInf;
        rhoInf          1.0;        // kinematic pressure -> rho = 1
        log             true;

        CofR            ({0.25 * chord:.6f} 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);

        magUInf         {u_inf:.4f};
        lRef            {chord:.4f};
        Aref            {aref:.6f};  // chord x one-pitch span
    }}
}}
"""


def fv_schemes(momentum_div: str = "bounded Gauss limitedLinearV 1") -> str:
    return foam_header("dictionary", "fvSchemes") + f"""
ddtSchemes
{{
    default         steadyState;
}}

// RapidCFD's GPU port registers a reduced scheme menu: grad is Gauss-only
// (no cellLimited/leastSquares), and linearUpwind is absent - limitedLinear
// (TVD) is the second-order convection workhorse that survived the port.
gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    // Two-stage convection: the runner solves first on dissipative upwind
    // to kill the impulsive-start global oscillation, then restarts on the
    // TVD scheme for the reported numbers (this file is the stage marker).
    div(phi,U)      {momentum_div};
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear limited 0.33;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         limited 0.33;
}}

fluxRequired
{{
    default         no;
    p;
}}
"""


def fv_solution() -> str:
    return foam_header("dictionary", "fvSolution") + """
solvers
{
    // GPU pressure solve: Krylov + AINV is RapidCFD's massively-parallel
    // sweet spot (DIC-style preconditioners serialize on GPU). With the
    // SP build, the fp64-emulated reductions keep the CG recurrences
    // honest; tolerances sized for fp32 fields.
    p
    {
        solver          PCG;
        preconditioner  AINV;
        tolerance       1e-7;
        relTol          0.01;
        maxIter         1000;
    }

    "(U|k|omega)"
    {
        solver          PBiCGStab;
        preconditioner  AINV;
        tolerance       1e-7;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-5;
        U               1e-6;
        "(k|omega)"     1e-6;
    }
}

// potentialFoam pre-pass settings (initializes U/phi so SIMPLE starts from
// an attached-flow guess instead of the impulsive uniform field whose
// startup vortex was observed to pump a domain-scale Cl limit cycle).
potentialFlow
{
    nNonOrthogonalCorrectors 10;
}

relaxationFactors
{
    // Deliberately heavy damping: the slip-walled slice channel reflects
    // everything, so transients must be killed by relaxation alone.
    fields
    {
        p               0.2;
    }
    equations
    {
        U               0.5;
        k               0.4;
        omega           0.4;
    }
}
"""


def transport_properties() -> str:
    return foam_header("dictionary", "transportProperties", "constant") + f"""
transportModel  Newtonian;

nu              nu [0 2 -1 0 0 0 0] {NU};
"""


def ras_properties() -> str:
    return foam_header("dictionary", "RASProperties", "constant") + """
// kOmegaSST fully-turbulent: RapidCFD carries no transition model (the
// documented limitation that parked it for M1); acceptable here because
// the NASA anchors at Re 2.2e6 are TRIPPED data and the study question is
// the VG-on vs VG-off DELTA, not absolute transition location.
RASModel        kOmegaSST;

turbulence      on;

printCoeffs     on;
"""


def turbulence_properties() -> str:
    return foam_header("dictionary", "turbulenceProperties", "constant") + """
simulationType  RASModel;
"""


def field_file(name: str, dims: str, internal: str, bcs: dict[str, str]) -> str:
    body = foam_header("volScalarField" if name != "U" else "volVectorField",
                       name, "0")
    body += f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for patch, bc in bcs.items():
        body += f"    {patch}\n    {{\n{bc}    }}\n\n"
    body += "}\n"
    return body


def zero_dir(u_inf: float) -> dict[str, str]:
    """All 0/ fields. Freestream turbulence: I=0.5%, nut/nu ~ 8.

    No slip/symmetry anywhere: sides are cyclic (exact span periodicity)
    and top/bottom use the freestream (per-face inletOutlet) family -
    RapidCFD's transform-patch algebra is avoided entirely (GPU artifact,
    see blockMeshDict header).
    """
    k_inf = 1.5 * (u_inf * 0.005) ** 2          # ~0.049 m2/s2
    omega_inf = k_inf / (8.0 * NU)              # nut = k/omega = 8 nu
    cyc = "        type            cyclicAMI;\n"
    fields = {}

    fs_u = (f"        type            freestream;\n"
            f"        freestreamValue uniform ({u_inf:.4f} 0 0);\n")
    fs_k = (f"        type            freestream;\n"
            f"        freestreamValue uniform {k_inf:.6g};\n")
    fs_w = (f"        type            freestream;\n"
            f"        freestreamValue uniform {omega_inf:.6g};\n")

    fields["U"] = field_file(
        "U", "[0 1 -1 0 0 0 0]", f"uniform ({u_inf:.4f} 0 0)",
        {
            "inlet":   f"        type            fixedValue;\n        value           uniform ({u_inf:.4f} 0 0);\n",
            "outlet":  f"        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform ({u_inf:.4f} 0 0);\n",
            "top": fs_u, "bottom": fs_u,
            "sideL": cyc, "sideR": cyc,
            "wing":    "        type            fixedValue;\n        value           uniform (0 0 0);\n",
        })

    fields["p"] = field_file(
        "p", "[0 2 -2 0 0 0 0]", "uniform 0",
        {
            "inlet":   "        type            zeroGradient;\n",
            "outlet":  "        type            fixedValue;\n        value           uniform 0;\n",
            "top": "        type            zeroGradient;\n",
            "bottom": "        type            zeroGradient;\n",
            "sideL": cyc, "sideR": cyc,
            "wing":    "        type            zeroGradient;\n",
        })

    fields["k"] = field_file(
        "k", "[0 2 -2 0 0 0 0]", f"uniform {k_inf:.6g}",
        {
            "inlet":   f"        type            fixedValue;\n        value           uniform {k_inf:.6g};\n",
            "outlet":  f"        type            inletOutlet;\n        inletValue      uniform {k_inf:.6g};\n        value           uniform {k_inf:.6g};\n",
            "top": fs_k, "bottom": fs_k,
            "sideL": cyc, "sideR": cyc,
            "wing":    f"        type            kqRWallFunction;\n        value           uniform {k_inf:.6g};\n",
        })

    fields["omega"] = field_file(
        "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega_inf:.6g}",
        {
            "inlet":   f"        type            fixedValue;\n        value           uniform {omega_inf:.6g};\n",
            "outlet":  f"        type            inletOutlet;\n        inletValue      uniform {omega_inf:.6g};\n        value           uniform {omega_inf:.6g};\n",
            "top": fs_w, "bottom": fs_w,
            "sideL": cyc, "sideR": cyc,
            "wing":    f"        type            omegaWallFunction;\n        value           uniform {omega_inf:.6g};\n",
        })

    fields["nut"] = field_file(
        "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
        {
            "inlet":   "        type            calculated;\n        value           uniform 0;\n",
            "outlet":  "        type            calculated;\n        value           uniform 0;\n",
            "top": "        type            calculated;\n        value           uniform 0;\n",
            "bottom": "        type            calculated;\n        value           uniform 0;\n",
            "sideL": cyc, "sideR": cyc,
            "wing":    "        type            nutkWallFunction;\n        value           uniform 0;\n",
        })
    return fields


# ---------------------------------------------------------------------------
#  Case assembly + the WSL runner
# ---------------------------------------------------------------------------

def write_case(name: str, wall_stl: Path, vanes_stl: Path | None,
               u_inf: float, chord: float, pitch: float, n_iter: int) -> None:
    case = CASES / name
    if case.exists():
        shutil.rmtree(case)
    (case / "system").mkdir(parents=True)
    (case / "constant" / "triSurface").mkdir(parents=True)
    (case / "0").mkdir()

    # Geometry into the case (snappy reads constant/triSurface/wing.stl).
    shutil.copy2(wall_stl, case / "constant" / "triSurface" / "wing.stl")
    if vanes_stl is not None:
        shutil.copy2(vanes_stl, case / "constant" / "triSurface" / "vanes.stl")

    (case / "system" / "blockMeshDict").write_text(block_mesh_dict(pitch))
    (case / "system" / "snappyHexMeshDict").write_text(snappy_dict(vanes_stl is not None))
    (case / "system" / "createPatchDict").write_text(create_patch_dict(pitch))
    (case / "system" / "surfaceFeatureExtractDict").write_text(surface_features_dict())
    (case / "system" / "controlDict").write_text(control_dict(u_inf, chord, pitch, n_iter))
    # Stage 1 = dissipative upwind (startup damping), stage 2 = TVD scheme
    # for the reported window; the runner swaps the active fvSchemes.
    (case / "system" / "fvSchemes").write_text(
        fv_schemes("bounded Gauss upwind"))
    (case / "system" / "fvSchemes.stage1").write_text(
        fv_schemes("bounded Gauss upwind"))
    (case / "system" / "fvSchemes.stage2").write_text(
        fv_schemes("bounded Gauss limitedLinearV 1"))
    (case / "system" / "fvSolution").write_text(fv_solution())

    (case / "constant" / "transportProperties").write_text(transport_properties())
    (case / "constant" / "RASProperties").write_text(ras_properties())
    (case / "constant" / "turbulenceProperties").write_text(turbulence_properties())

    for fname, content in zero_dir(u_inf).items():
        (case / "0" / fname).write_text(content)

    print(f"[case] {name}: assembled at {case}")


RUNNER = r"""#!/bin/bash
# Mesh (OpenFOAM v2506, CPU) + solve (RapidCFD SP/sm_120, GPU) every case.
# Cases are staged into the WSL filesystem first: the solver writes far too
# often for the 9p-mounted Windows drive.
# (no set -u: the OpenFOAM/RapidCFD bashrc files reference unset vars freely)
set -o pipefail

# MESH_ONLY=1 stops after meshing - used to pre-mesh while RapidCFD compiles.
MESH_ONLY=${MESH_ONLY:-0}
# SKIP_MESH=1 solves an already-staged, already-meshed case in place.
SKIP_MESH=${SKIP_MESH:-0}

SRC="$(dirname "$(readlink -f "$0")")/cases"
STAGE=~/glasair_rapidcfd
RESULTS="$(dirname "$(readlink -f "$0")")/results"
mkdir -p "$STAGE" "$RESULTS"

OF=/usr/lib/openfoam/openfoam2506/etc/bashrc
RC=~/RapidCFD-dev/etc/bashrc

for case in "$@"; do
    src="$SRC/$case"
    dst="$STAGE/$case"

    if [ "$SKIP_MESH" != "1" ]; then
        [ -d "$src" ] || { echo "no such case: $case"; exit 1; }
        rm -rf "$dst"; cp -r "$src" "$dst"

        echo "=== [$case] meshing (v2506) ==="
        (
            source "$OF" || true
            cd "$dst"
            surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 \
                || surfaceFeatures > log.surfaceFeatureExtract 2>&1
            blockMesh           > log.blockMesh 2>&1            || exit 1
            snappyHexMesh -overwrite > log.snappyHexMesh 2>&1   || exit 1
            # Convert the plain side patches into the translational cyclic
            # pair (meshing directly with cyclic sides crashes the layer
            # extrusion when the wall geometry pierces the boundary).
            createPatch -overwrite > log.createPatch 2>&1       || exit 1
            # createPatch may emit the repatched mesh into 0/ depending on
            # startFrom; the solver and renumberMesh expect it in constant/.
            if [ -d 0/polyMesh ]; then
                rm -rf constant/polyMesh
                mv 0/polyMesh constant/polyMesh
            fi
            checkMesh           > log.checkMesh 2>&1
            # No renumberMesh: v2506's renumber rewrites the boundary file
            # before it renumbers fields, and it dies on the 2.3-dialect
            # cyclic field entries - leaving the mesh de-cycled. The GPU
            # solver does not need the bandwidth ordering badly enough to
            # risk that.
        ) || { echo "[$case] MESHING FAILED"
               for f in "$dst"/log.*; do echo "--- $f"; tail -20 "$f"; done
               continue; }
        grep -E "^ *cells:|Mesh OK|Failed" "$dst/log.checkMesh" | head -5
    fi

    [ "$MESH_ONLY" = "1" ] && { echo "=== [$case] mesh-only, skipping solve ==="; continue; }

    echo "=== [$case] solving (RapidCFD, GPU) ==="
    (
        export PATH=/usr/local/cuda/bin:$PATH
        source "$RC"
        cd "$dst"

        # Fresh pseudo-time history: drop any time dirs from earlier runs.
        for d in [0-9]*; do [ "$d" != "0" ] && rm -rf "$d"; done
        rm -rf postProcessing

        # Attached-flow initial guess (writes U/phi into 0/).
        potentialFoam -noFunctionObjects > log.potentialFoam 2>&1

        # Stage 1: first-order momentum, kills the startup transient.
        cp system/fvSchemes.stage1 system/fvSchemes
        sed -i "s/^endTime.*/endTime         ${STAGE1_END:-2000};/" system/controlDict
        simpleFoam > log.simpleFoam.stage1 2>&1

        # Stage 2: restart on the TVD scheme for the reported window.
        cp system/fvSchemes.stage2 system/fvSchemes
        sed -i "s/^endTime.*/endTime         ${STAGE2_END:-4500};/" system/controlDict
        simpleFoam > log.simpleFoam 2>&1
        true
    )
    # RapidCFD exits non-zero from a benign CUDA teardown bug even after a
    # clean run, so success is judged by the solver reaching its End banner.
    if grep -q "^End$" "$dst/log.simpleFoam" 2>/dev/null; then
        tail -5 "$dst/log.simpleFoam"
    else
        echo "[$case] SOLVE FAILED"
        for f in "$dst"/log.potentialFoam "$dst"/log.simpleFoam.stage1 "$dst"/log.simpleFoam; do
            echo "--- $f"; tail -15 "$f" 2>/dev/null
        done
    fi

    # Ship the numbers home (coefficients, logs, last mesh stats).
    out="$RESULTS/$case"
    mkdir -p "$out"
    cp -r "$dst/postProcessing" "$out/" 2>/dev/null
    cp "$dst"/log.* "$out/" 2>/dev/null
    echo "=== [$case] done; results in $out ==="
done
"""


def main() -> None:
    # Optional case filter: `--only nameA nameB` rebuilds just those matrix
    # entries (used to add sweep cases without touching ones already staged
    # or mid-solve in WSL - write_case wipes the case dir it rebuilds).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="CASE",
                    help="build only these CASE_MATRIX entries")
    opts = ap.parse_args()

    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station
    default_pitch = ac.vg_defaults.wing.spacing_outboard

    print(f"[build] chord={chord:.4f} m  pitch={default_pitch * 1000:.0f} mm  "
          f"(study Re {RE:.2g} -> U={RE * NU / chord:.2f} m/s)")

    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")

    selected = [row for row in CASE_MATRIX
                if not opts.only or row["name"] in opts.only]
    if opts.only:
        missing = set(opts.only) - {row["name"] for row in selected}
        if missing:
            raise SystemExit(f"[build] not in CASE_MATRIX: {sorted(missing)}")

    # Each row carries its own knobs; None fields fall back to study defaults.
    # The freestream speed is recomputed per case so the cruise pair
    # (Re 5.52e6) and the stall cases share one geometry pipeline but
    # different magUInf/BCs. Pitch overrides change BOTH the periodic slice
    # width (domain_span) and the vane z-offset so the array stays periodic.
    for row in selected:
        re_eff = row["re"] if row["re"] is not None else RE
        u_inf = re_eff * NU / chord
        pitch = (row["pitch_mm"] / 1000.0 if row["pitch_mm"] is not None
                 else default_pitch)
        # "single" alternating fins need a 2-pitch periodic cell so the cyclic
        # BCs see one fin yawed each way; the mesh slab + cyclic spacing use
        # this slab width while the vanes are still placed on the true pitch.
        slab = 2.0 * pitch if row["count"] == "single" else pitch
        beta_txt = f"{row['beta_deg']:g}" if row["beta_deg"] is not None else "15"
        print(f"[build] {row['name']}: alpha={row['alpha']:g}deg  "
              f"Re={re_eff:.2g}  U={u_inf:.2f} m/s  pitch={pitch*1000:.0f}mm  "
              f"slab={slab*1000:.0f}mm  shape={row['shape']}  "
              f"toe={row['toe']}  count={row['count']}  beta={beta_txt}deg")
        wall, vanes = build_article(row["name"], row["h_mm"], row["alpha"],
                                    ac, coords, slab,
                                    x_frac_override=row["x_frac"],
                                    shape=row["shape"], toe=row["toe"],
                                    pitch_override=pitch, count=row["count"],
                                    beta_deg_override=row["beta_deg"])
        write_case(row["name"], wall, vanes, u_inf, chord, slab, n_iter=4000)

    runner = HERE / "run_all.sh"
    runner.write_text(RUNNER, newline="\n")
    print(f"[build] runner: {runner}")
    print("[build] WSL:  bash gpu/rapidcfd/run_all.sh " +
          " ".join(row["name"] for row in selected))


if __name__ == "__main__":
    main()
