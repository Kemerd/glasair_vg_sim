# -*- coding: utf-8 -*-
"""Phase-1 2D validation case builder: C-grid generator + template filler.

Turns (AoA, Re, refinement level) plus aircraft.yaml into a complete,
runnable OpenFOAM case under cases/runs/, instantiated from the template
tree at cases/validation_2d/template/. Two jobs live here:

  1. A structured 2D C-grid blockMeshDict GENERATOR around the resampled
     LS(1)-0413 section (blunt TE -- the real 0.0055c base is kept; this is
     a transition study and the as-built TE matters; the XFOIL anchor runs
     sharp-TE because XFOIL requires it, and the comparison philosophy is
     "trust deltas, not absolutes"). Classic C-topology: far field 25 chords
     in every direction (spec minimum), wake block to 25c downstream, spline
     edges through the resampled surface points, one cell thick in z with
     empty front/back patches. The first wall-normal cell comes from the
     correlation chain in scripts/first_cell_height.py (imported, never
     duplicated), growth ratio capped at 1.2, and at least 30 cells inside
     the estimated boundary layer.

  2. Template instantiation: every @TOKEN@ in the template files is filled
     from aircraft.yaml + the CLI (freestream vector and forceCoeffs lift/
     drag directions rotated per AoA, nu set so Re is exact at c = 1 m,
     Mack/Langtry-Menter inlet turbulence chain, suction-surface BL sample
     lines), and the generated blockMeshDict replaces the template stub.

2D series convention (documented in every case README): chord c = 1.0 m and
U = Re * nu / c with nu from aircraft.yaml. At Re = 6e6 this gives U ~ 87.6
m/s, Mach ~ 0.26 -- above the 0.2 quasi-incompressible guideline. Decision:
KEEP c = 1.0 and run Re 6M anyway as a documented trend point (the gate
anchors at Re 3M / M 0.13; the ~3.5% Prandtl-Glauert bias on absolutes
cancels in the deltas this project trades in; one chord keeps one mesh
family for the whole Re sweep). Chord scaling was considered and rejected
for exactly that mesh-family reason.

Dry-run friendly by design: builds the full case directory with zero
OpenFOAM present; the case's Allrun executes later inside WSL2 Ubuntu with
ESI OpenFOAM v2506.

Run:  python scripts/build_validation_case.py --aoa 4 --re 3e6 --level 0
"""
from __future__ import annotations

import argparse
import math
import re as regex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python scripts/build_validation_case.py", Python puts scripts/
#  on sys.path rather than the repo root, which would hide the geometry
#  package. Inserting this file's grandparent mirrors conftest.py and keeps
#  the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geometry.airfoil import load_airfoil, resample_airfoil  # noqa: E402
from geometry.units import load_aircraft  # noqa: E402

# The y+ sizing chain is REUSED from the M0 calculator -- single home for the
# Schlichting correlation arithmetic (spec/house rule: never duplicate it).
from scripts.first_cell_height import first_cell_height, layers_to_cover  # noqa: E402

# =============================================================================
#  Fixed conventions of the 2D validation series
# =============================================================================

CHORD = 1.0                  # m; the 2D series convention (see module docstring)
FARFIELD_RADIUS = 25.0       # chords; spec Phase-1 minimum far-field distance
WAKE_LENGTH = 25.0           # chords of wake block downstream of the TE
Z_THICKNESS = 0.1            # m; one-cell extrusion depth (Aref = c * z)
GROWTH_CAP = 1.2             # spec section 4 wall-normal growth-ratio cap
MIN_BL_LAYERS = 30           # spec minimum cell count inside the BL
LEVEL_FACTOR = math.sqrt(2.0)  # per-level refinement step (GCI study)
N_AIRFOIL_POINTS = 241       # resample density feeding the spline edges

# Inlet disturbance environment: the XFOIL anchor runs e^N with Ncrit = 9;
# Mack's relation maps that onto the freestream Tu the transition model sees.
# The Mack value is the LE-INCIDENT target: k decays between the far boundary
# and the airfoil at the rate beta* * omega, so the boundary value of k is
# pre-boosted by the inverse analytic decay factor (see inlet_turbulence).
NCRIT = 9.0
TURB_VISC_RATIO = 10.0       # nut/nu at the far boundary; top of the
                             # Langtry-Menter 1..10 band = slowest Tu decay
BETA_STAR = 0.09             # k-omega beta*: freestream dk/dt = -beta* k omega

# Sea-level sound-speed constants for the Mach bookkeeping (dry air).
GAMMA_AIR = 1.4
R_AIR = 287.058              # J/(kg K)

# Baseline (level 0) discretisation knobs. Surface streamwise distribution
# is three geometric zones per surface: (length fraction of chord, share of
# the surface cell count, expansion last/first). Fine at the LE for the
# suction peak / transition region, fine again into the TE for the blunt
# base and the aft pressure recovery, uniform-ish over the mid chord.
N_SURF_BASE = 128            # cells per surface (upper, lower) at level 0
SURF_ZONES = (
    (0.20, 0.34, 10.0),      # LE cluster: grows away from the nose
    (0.55, 0.36, 1.0),       # mid chord: uniform
    (0.25, 0.30, 0.2),       # TE cluster: shrinks into the trailing edge
)

# Wall-normal BL sample stations for the Phase-1 delta/delta* extraction
# (suction surface). TWO lines per station (naming contract shared with
# scripts/extract_bl.py, which merges them before integrating):
#   * inner 'bl_x###_inner': dense near-wall line, BL_INNER_D99_FACTOR x the
#     flat-plate d99 estimate long at ~BL_INNER_SPACING point pitch -- the
#     resolved viscous sublayer (first cell ~1e-5 m at Re 3M) actually gets
#     sampled instead of falling between two coarse points;
#   * outer 'bl_x###': the original 0.15c / 600-point line that captures the
#     BL edge and the outer flow the edge-finder guard needs. 0.15c clears
#     the BL edge comfortably even at high AoA aft stations.
BL_STATIONS = (0.05, 0.07, 0.10, 0.15, 0.20, 0.30,
               0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
BL_LINE_LENGTH = 0.15        # chords, outer line, along the local outward normal
BL_LINE_POINTS = 600         # uniform samples on the outer line
BL_INNER_D99_FACTOR = 2.0    # inner-line length as a multiple of the d99 estimate
BL_INNER_SPACING = 1.0e-5    # m; inner-line point pitch (~the Re 3M first cell)
BL_WALL_OFFSET = 1.0e-6      # m; start a hair off the wall, inside cell 1

# Parallel decomposition: MPI ranks per refinement level (lvl 0/1/2). The 2D
# cell budget roughly doubles per level (~27k/54k/107k cells), so doubling the
# rank count holds the per-core load near ~13k cells at every level -- above
# the ~10k cells/core point where MPI overhead starts eating the speedup. A
# flat 8 ranks (the original choice) starved level 0 at ~3.4k cells/core.
SUBDOMAINS_BY_LEVEL = {0: 2, 1: 4, 2: 8}

# Template/output locations and the placeholder-token grammar shared with
# the tests (any surviving @NAME@ after instantiation is a build failure).
TEMPLATE_DIR = REPO / "cases" / "validation_2d" / "template"
RUNS_DIR = REPO / "cases" / "runs"
TOKEN_RE = regex.compile(r"@[A-Z][A-Z0-9_]*@")


# =============================================================================
#  Geometric-progression helpers (1D graded cell runs)
# =============================================================================

def _geom_total(first: float, ratio: float, n: int) -> float:
    """Total length of n geometrically growing cells starting at `first`."""
    # Degenerate ratio -> uniform run; the closed form divides by (r - 1).
    if abs(ratio - 1.0) < 1.0e-12:
        return first * n
    return first * (ratio ** n - 1.0) / (ratio - 1.0)


def _ratio_for_count(first: float, length: float, n: int) -> float:
    """Cell-to-cell ratio r so that n cells starting at `first` span `length`.

    The geometric sum is strictly increasing in r, so brentq on a generous
    bracket nails the root; callers guarantee n*first < length (otherwise a
    uniform run already overshoots and r = 1 is returned).
    """
    if n * first >= length:
        return 1.0
    return brentq(lambda r: _geom_total(first, r, n) - length,
                  1.0 + 1.0e-12, 10.0, xtol=1.0e-13)


def _count_for_ratio(first: float, length: float, ratio: float) -> int:
    """Smallest cell count covering `length` from `first` at ratio `ratio`."""
    if ratio <= 1.0 + 1.0e-12:
        return int(math.ceil(length / first))
    return int(math.ceil(
        math.log(1.0 + length * (ratio - 1.0) / first) / math.log(ratio)
    ))


def _solve_graded_run(first: float, length: float, ratio_cap: float
                      ) -> Tuple[int, float]:
    """Pick (n, r) so n cells from `first` span `length` exactly, r <= cap.

    Strategy: take the fewest cells admissible at the cap, then re-solve the
    ratio exactly for that integer count. The exact ratio can only come out
    AT or BELOW the cap (more cells than strictly needed never demand faster
    growth), so the cap is honored by construction -- and the first cell is
    exactly `first`, which is what the y+ < 1 budget was computed for.
    """
    n = _count_for_ratio(first, length, ratio_cap)
    r = _ratio_for_count(first, length, n)
    return n, r


def _ratio_for_layer_budget(first: float, target: float, n_layers: int) -> float:
    """Ratio at which exactly n_layers cells from `first` bury `target`.

    Used to translate the ">= 30 cells inside the BL" rule into a growth-
    ratio ceiling: any r at or below this value puts at least n_layers cells
    inside the BL thickness estimate.
    """
    if n_layers * first >= target:
        # Even a uniform run covers the BL with the layer budget; no ceiling.
        return float("inf")
    return brentq(lambda r: _geom_total(first, r, n_layers) - target,
                  1.0 + 1.0e-12, 10.0, xtol=1.0e-13)


# =============================================================================
#  Mesh plan: every number the blockMeshDict needs, solved up front
# =============================================================================

@dataclass(frozen=True)
class GradedRun:
    """One graded 1D cell run: count, cell-to-cell ratio, blockMesh grading
    (= total expansion last/first), first/last cell size, run length."""
    n: int
    ratio: float
    grading: float
    first: float
    last: float
    length: float


@dataclass(frozen=True)
class SurfZone:
    """One streamwise surface zone: chord-length fraction, integer cell
    count, total expansion (last/first), end cell sizes, |cell ratio|."""
    len_frac: float
    n_cells: int
    expansion: float
    d_start: float
    d_end: float
    ratio: float


@dataclass(frozen=True)
class MeshPlan:
    """Complete sizing record for one C-grid build (also the test contract:
    the unit tests audit these numbers against the spec rules)."""
    re: float
    u_inf: float
    mach: float
    level: int
    y1_correlation: float        # first-cell height straight from the chain
    h1: float                    # level-scaled first cell actually meshed
    d99: float                   # flat-plate BL thickness estimate at the TE
    normal: GradedRun            # wall-normal run (wall -> far field)
    layers_in_d99: int           # cells inside d99 at the solved ratio
    n_surf: int                  # streamwise cells per surface
    zones: Tuple[SurfZone, ...]  # the three streamwise zones
    d_le: float                  # finest LE streamwise spacing
    d_te: float                  # finest TE streamwise spacing
    wake: GradedRun              # wake streamwise run (TE -> outlet)
    base: GradedRun              # HALF of the blunt-TE base, corner -> midline
                                 # (symmetric two-zone multigrading; first = h1)
    n_base: int                  # total cells across the base (= 2 * base.n)
    n_cells_total: int           # whole-mesh cell count (one z layer)


def _solve_surface_zones(n_surf: int) -> Tuple[SurfZone, ...]:
    """Resolve SURF_ZONES into integer cell counts and end spacings.

    Zone shares are rounded to integers and the middle zone absorbs the
    rounding remainder so the counts always sum to n_surf exactly (blockMesh
    normalizes fractions, but integer bookkeeping keeps the printed spacing
    numbers honest). Spacings use the geometric closed form per zone.
    """
    counts = [max(2, int(round(share * n_surf))) for _, share, _ in SURF_ZONES]
    counts[1] += n_surf - sum(counts)          # middle zone takes the slack

    zones: List[SurfZone] = []
    for (len_frac, _share, expansion), n in zip(SURF_ZONES, counts):
        # Cell-to-cell ratio from the total expansion across n cells; the
        # first-cell closed form holds for growing AND shrinking zones.
        if abs(expansion - 1.0) < 1.0e-12:
            ratio, d_start = 1.0, len_frac / n
        else:
            ratio = expansion ** (1.0 / (n - 1))
            d_start = len_frac * (ratio - 1.0) / (ratio ** n - 1.0)
        d_end = d_start * expansion
        # Report the ratio as a magnitude >= 1 so the <= 1.2 audit reads the
        # same for shrinking zones (ratio < 1) as for growing ones.
        zones.append(SurfZone(len_frac, n, expansion, d_start, d_end,
                              max(ratio, 1.0 / ratio)))
    return tuple(zones)


def plan_cgrid(re_target: float, nu: float, rho: float, temperature: float,
               level: int, base_height: float) -> MeshPlan:
    """Solve every discretisation number for one (Re, level) C-grid.

    The wall-normal chain: first-cell height from the M0 correlation script
    (y+ = 1 target), then the growth ratio is the tightest of (a) the spec
    cap 1.2 and (b) the ratio that still leaves >= 30 cells inside the
    flat-plate d99 estimate; the cell count follows from spanning the 25c
    far-field distance, and the ratio is re-solved exactly for that count so
    the meshed first cell IS the correlation value.

    ``base_height`` is the MEASURED blunt-TE base opening (yu - yl of the
    resampled loop, ~0.0055c for the as-built LS(1)-0413), passed in rather
    than assumed so the base-slab grading below always matches the geometry
    actually being meshed.
    """
    if level not in (0, 1, 2):
        raise ValueError(f"refinement level must be 0, 1 or 2, got {level}")
    if base_height <= 0.0:
        raise ValueError(f"base_height must be positive, got {base_height}")

    # Freestream speed from the series convention U = Re * nu / c, and the
    # Mach bookkeeping for the compressibility note (sea-level sound speed).
    u_inf = re_target * nu / CHORD
    mach = u_inf / math.sqrt(GAMMA_AIR * R_AIR * temperature)

    # Correlation chain (REUSED, not duplicated): Schlichting Cf -> u_tau ->
    # y1 at y+ = 1, plus the d99 estimate that anchors the BL layer budget.
    chain = first_cell_height(u_inf, CHORD, rho, nu, y_plus=1.0)
    y1 = chain["y1"]
    d99 = chain["d99"]

    # Level-0 baseline COUNTS come from the caps (growth <= 1.2 and the
    # 30-layer BL budget); refinement then multiplies every direction's
    # count by sqrt(2) per level while the first cell shrinks by the same
    # factor and the ratios are RE-SOLVED exactly. Scaling counts (not just
    # h1) is what makes the levels a true uniform-refinement family: with a
    # fixed ratio cap, a smaller first cell alone would barely change the
    # outer mesh and the GCI premise would be broken.
    r_layers0 = _ratio_for_layer_budget(y1, d99, MIN_BL_LAYERS)
    n_norm0, _r0 = _solve_graded_run(y1, FARFIELD_RADIUS,
                                     min(GROWTH_CAP, r_layers0))
    zones0 = _solve_surface_zones(N_SURF_BASE)
    n_wake0, _rw0 = _solve_graded_run(zones0[-1].d_end, WAKE_LENGTH,
                                      GROWTH_CAP)

    scale = LEVEL_FACTOR ** level
    h1 = y1 / scale

    # Wall-normal run at this level: scaled count, exact ratio for the
    # scaled first cell. More cells over the same 25c can only LOWER the
    # ratio, so the caps hold at every level; verified defensively anyway.
    n_norm = int(math.ceil(n_norm0 * scale))
    r_norm = _ratio_for_count(h1, FARFIELD_RADIUS, n_norm)
    if r_norm > GROWTH_CAP + 1.0e-9:
        raise RuntimeError(
            f"normal growth ratio {r_norm:.4f} exceeds cap {GROWTH_CAP}")
    normal = GradedRun(
        n=n_norm, ratio=r_norm, grading=r_norm ** (n_norm - 1),
        first=h1, last=h1 * r_norm ** (n_norm - 1), length=FARFIELD_RADIUS,
    )
    layers, _covered = layers_to_cover(h1, r_norm, d99)

    # Streamwise surface zones at this level's cell count.
    n_surf = max(8, int(round(N_SURF_BASE * scale)))
    zones = _solve_surface_zones(n_surf)
    d_le = zones[0].d_start            # finest cell at the nose
    d_te = zones[-1].d_end             # finest cell into the TE

    # Wake streamwise run: first cell continues the TE surface spacing (no
    # size jump across the surface/wake block interface); scaled count,
    # exact ratio, same cap discipline as the normal run.
    n_wake = int(math.ceil(n_wake0 * scale))
    r_wake = _ratio_for_count(d_te, WAKE_LENGTH, n_wake)
    if r_wake > GROWTH_CAP + 1.0e-9:
        raise RuntimeError(
            f"wake growth ratio {r_wake:.4f} exceeds cap {GROWTH_CAP}")
    wake = GradedRun(
        n=n_wake, ratio=r_wake, grading=r_wake ** (n_wake - 1),
        first=d_te, last=d_te * r_wake ** (n_wake - 1), length=WAKE_LENGTH,
    )

    # Blunt-base slab (W_m), transverse direction: the W_u/W_l blocks meet
    # the wake-cut lines with the y+ = 1 first cell h1, so a uniform run
    # across the base would jump 57x..218x in wall-normal size at the two
    # shared corners (Re 1.5M..6M). Instead the base gets a SYMMETRIC
    # two-zone multigrading: each half runs from h1 at its corner toward a
    # coarse midline at a cell-to-cell ratio capped at GROWTH_CAP, solved
    # exactly so the edge cell IS h1 (same count-then-exact-ratio discipline
    # as the wall-normal run, including the sqrt(2) level scaling). The base
    # wall itself remains resolved by the wake's streamwise spacing (dead-air
    # recirculation; the yPlus function object writes the evidence).
    half_base = 0.5 * base_height
    n_bhalf0 = max(2, _count_for_ratio(y1, half_base, GROWTH_CAP))
    n_bhalf = int(math.ceil(n_bhalf0 * scale))
    r_base = _ratio_for_count(h1, half_base, n_bhalf)
    if r_base > GROWTH_CAP + 1.0e-9:
        raise RuntimeError(
            f"base growth ratio {r_base:.4f} exceeds cap {GROWTH_CAP}")
    base = GradedRun(
        n=n_bhalf, ratio=r_base, grading=r_base ** (n_bhalf - 1),
        first=h1, last=h1 * r_base ** (n_bhalf - 1), length=half_base,
    )
    n_base = 2 * n_bhalf

    # Whole-mesh budget: 2 surface blocks + 2 wake blocks + base block.
    n_cells = 2 * n_surf * n_norm + 2 * n_wake * n_norm + n_wake * n_base

    return MeshPlan(
        re=re_target, u_inf=u_inf, mach=mach, level=level,
        y1_correlation=y1, h1=h1, d99=d99, normal=normal,
        layers_in_d99=layers, n_surf=n_surf, zones=zones,
        d_le=d_le, d_te=d_te, wake=wake, base=base, n_base=n_base,
        n_cells_total=n_cells,
    )


# =============================================================================
#  blockMeshDict generation (5-block structured C-grid, blunt TE)
# =============================================================================
#
#  2D point layout (z extruded one cell to Z_THICKNESS):
#
#        3:F(-25,0)          4:Tt(1,25)              8:Otop(26,25)
#           *  . . arc . . . . *-----------------------*
#           |                  |                        |
#           |   S_u            |        W_u             |
#           |        1:TEu(1,yu)*----------------------* 6:Ou(26,yu)
#         0:LE(0,0)*           |          W_m           |
#           |        2:TEl(1,yl)*----------------------* 7:Ol(26,yl)
#           |   S_l            |        W_l             |
#           *  . . arc . . . . *-----------------------*
#        (mirror of F arc)   5:Bt(1,-25)             9:Obot(26,-25)
#
#  S_u/S_l: airfoil surface -> far field, split at the LE; the shared
#  LE->F line is an internal face (the C-grid "nose cut"). W_u/W_l: wake
#  above/below the cut lines y = yu / y = yl. W_m: the thin slab behind the
#  blunt base (its left face IS the base wall). Vertex ids: 2D id + 10 per
#  z-plane (0-9 at z = 0, 10-19 at z = Z_THICKNESS).

def _split_loop(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a resampled Selig loop into (upper, lower), both LE -> TE.

    Relies on the resample_airfoil contract: odd point count, single shared
    LE at index (N+1)//2 - 1, upper surface listed TE -> LE first. Verified
    here so a contract change upstream fails loudly instead of meshing junk.
    """
    n = coords.shape[0]
    if n % 2 == 0:
        raise ValueError("resampled loop must have an odd point count")
    n_side = (n + 1) // 2
    upper = coords[:n_side][::-1]      # LE -> TE
    lower = coords[n_side - 1:]        # LE -> TE (shares the LE row)
    if not np.allclose(upper[0], lower[0]):
        raise ValueError("loop does not share a single LE point")
    return upper, lower


def _fmt(x: float) -> str:
    """Compact float formatting for dictionary output (round-trip safe)."""
    return f"{x:.10g}"


def _vec(x: float, y: float, z: float) -> str:
    return f"({_fmt(x)} {_fmt(y)} {_fmt(z)})"


def _spline_edge(v0: int, v1: int, pts: np.ndarray, z: float) -> str:
    """One blockMesh spline edge with interior interpolation points."""
    body = "\n".join(f"        {_vec(p[0], p[1], z)}" for p in pts)
    return f"    spline {v0} {v1}\n    (\n{body}\n    )"


def _polyline_edge(v0: int, v1: int, pts: Sequence[Tuple[float, float]],
                   z: float) -> str:
    """One blockMesh polyLine edge (linear between points; used for the far
    boundary, whose exact shape is aerodynamically irrelevant at 25c)."""
    body = "\n".join(f"        {_vec(x, y, z)}" for x, y in pts)
    return f"    polyLine {v0} {v1}\n    (\n{body}\n    )"


def _surf_grading(zones: Tuple[SurfZone, ...], reverse: bool) -> str:
    """Multigrading spec for a surface edge, optionally direction-reversed.

    blockMesh multigrading triples are (length frac, cell frac, expansion);
    reversing a direction reverses the zone order AND inverts each expansion
    (a zone growing along +x1 shrinks along -x1). Used verbatim for S_u
    (LE -> TE) and reversed for S_l (whose x1 runs TE -> LE to keep the hex
    right-handed).
    """
    zs = list(zones)
    if reverse:
        zs = [SurfZone(z.len_frac, z.n_cells, 1.0 / z.expansion,
                       z.d_end, z.d_start, z.ratio) for z in reversed(zs)]
    parts = " ".join(
        f"({_fmt(z.len_frac)} {z.n_cells} {_fmt(z.expansion)})" for z in zs
    )
    return f"( {parts} )"


def _arc_points(theta_start_deg: float, theta_end_deg: float,
                n: int) -> List[Tuple[float, float]]:
    """Sample the far-field circle (radius FARFIELD_RADIUS about the LE)
    between two polar angles, endpoints excluded (they are block vertices
    or repeated knots supplied by the caller)."""
    th = np.linspace(math.radians(theta_start_deg),
                     math.radians(theta_end_deg), n + 2)[1:-1]
    return [(FARFIELD_RADIUS * math.cos(t), FARFIELD_RADIUS * math.sin(t))
            for t in th]


def generate_blockmeshdict(coords: np.ndarray, plan: MeshPlan) -> str:
    """Render the full blockMeshDict text for one resampled loop + plan."""
    upper, lower = _split_loop(coords)

    # Blunt-TE corner ordinates; the resampler guarantees both surfaces end
    # at x = 1 (cosine targets), so only y differs across the open base.
    yu = float(upper[-1, 1])
    yl = float(lower[-1, 1])
    if yu <= yl:
        raise ValueError("expected an open blunt TE (upper above lower)")
    # The base-slab grading in the plan was solved for a specific base
    # opening; meshing a loop with a different one would silently break the
    # h1 corner match, so a mismatch is a hard error, not a warning.
    if not math.isclose(yu - yl, 2.0 * plan.base.length, rel_tol=1.0e-6):
        raise ValueError(
            f"plan base height {2.0 * plan.base.length:.6e} does not match "
            f"the loop's TE opening {yu - yl:.6e}")

    R = FARFIELD_RADIUS
    x_out = CHORD + WAKE_LENGTH
    z0, z1 = 0.0, Z_THICKNESS

    # ---- vertices: 2D table extruded to both z planes ----------------------
    pts2d = [
        (0.0, 0.0),        # 0  LE
        (CHORD, yu),       # 1  TE upper corner
        (CHORD, yl),       # 2  TE lower corner
        (-R, 0.0),         # 3  F: far field straight upstream of the LE
        (CHORD, R),        # 4  Tt: far field above the TE
        (CHORD, -R),       # 5  Bt: far field below the TE
        (x_out, yu),       # 6  Ou: outlet on the upper wake-cut line
        (x_out, yl),       # 7  Ol: outlet on the lower wake-cut line
        (x_out, R),        # 8  Otop: outlet top corner
        (x_out, -R),       # 9  Obot: outlet bottom corner
    ]
    vert_lines = []
    for plane, z in ((0, z0), (1, z1)):
        for i, (x, y) in enumerate(pts2d):
            vert_lines.append(f"    {_vec(x, y, z)}    // {i + 10 * plane}")
    vertices = "\n".join(vert_lines)

    # ---- blocks -------------------------------------------------------------
    g_norm = _fmt(plan.normal.grading)
    g_wake = _fmt(plan.wake.grading)
    g_wake_rev = _fmt(1.0 / plan.wake.grading)
    n_srf, n_nrm = plan.n_surf, plan.normal.n
    n_wke, n_bse = plan.wake.n, plan.n_base
    # W_m transverse multigrading: lower half grows from h1 at the W_l
    # corner toward the midline, upper half mirrors it back down to h1 at
    # the W_u corner -- both shared faces continue the y+ = 1 first cell of
    # the neighbouring wake blocks with no size jump.
    g_base = _fmt(plan.base.grading)
    g_base_rev = _fmt(1.0 / plan.base.grading)
    n_bh = plan.base.n
    base_grading = (f"( (0.5 {n_bh} {g_base}) (0.5 {n_bh} {g_base_rev}) )")
    blocks = f"""    // S_u: upper surface -> far field. x1 = LE->TE along the wall
    // (multigraded: LE/TE clusters), x2 = wall->far (first cell from the
    // y+ chain, geometric growth, ratio {_fmt(plan.normal.ratio)}).
    hex (0 1 4 3 10 11 14 13) ({n_srf} {n_nrm} 1)
    simpleGrading ( {_surf_grading(plan.zones, reverse=False)} {g_norm} 1 )

    // S_l: lower surface -> far field. x1 runs TE->LE (right-handedness),
    // so the surface multigrading is reversed/inverted.
    hex (2 0 3 5 12 10 13 15) ({n_srf} {n_nrm} 1)
    simpleGrading ( {_surf_grading(plan.zones, reverse=True)} {g_norm} 1 )

    // W_u: wake above the upper cut line. x1 = TE->outlet (first cell
    // continues the TE surface spacing to within the ~2% arc-length vs
    // chord-fraction difference of the multigraded surface edge), x2 =
    // cut->far (same normal run as the surface blocks so the shared face
    // matches cell-for-cell).
    hex (1 6 8 4 11 16 18 14) ({n_wke} {n_nrm} 1)
    simpleGrading ( {g_wake} {g_norm} 1 )

    // W_l: wake below the lower cut line. x1 = outlet->TE (right-handed),
    // hence the inverted wake grading; x2 = cut->far.
    hex (7 2 5 9 17 12 15 19) ({n_wke} {n_nrm} 1)
    simpleGrading ( {g_wake_rev} {g_norm} 1 )

    // W_m: slab behind the blunt base (left face = the base wall);
    // streamwise identical to W_u. Transverse (x2 = lower cut -> upper cut):
    // symmetric two-zone multigrading, edge cells = h1 at BOTH corners so
    // the shared wake-cut faces match W_l/W_u cell-for-cell (ratio
    // {_fmt(plan.base.ratio)}, cap {_fmt(GROWTH_CAP)}, {n_bse} cells across the base).
    hex (2 7 6 1 12 17 16 11) ({n_wke} {n_bse} 1)
    simpleGrading ( {g_wake} {base_grading} 1 )"""

    # ---- edges ---------------------------------------------------------------
    # Airfoil surface: spline edges through the resampled points (interior
    # points only; the block vertices supply the endpoints). The lower edge
    # belongs to S_l whose x1 runs TE -> LE, so its points are reversed.
    up_int = upper[1:-1]
    lo_int = lower[1:-1][::-1]
    # Far boundary: composite quarter-arc + straight, rendered as polyLines
    # (dense arc sampling; exact far-boundary shape is irrelevant at 25c).
    far_u = _arc_points(180.0, 90.0, 45) + [(0.0, R), (0.5 * CHORD, R)]
    far_l = [(0.5 * CHORD, -R), (0.0, -R)] + _arc_points(270.0, 180.0, 45)
    edge_lines = []
    for z in (z0, z1):
        off = 0 if z == z0 else 10
        edge_lines.append(_spline_edge(0 + off, 1 + off, up_int, z))
        edge_lines.append(_spline_edge(2 + off, 0 + off, lo_int, z))
        edge_lines.append(_polyline_edge(3 + off, 4 + off, far_u, z))
        edge_lines.append(_polyline_edge(5 + off, 3 + off, far_l, z))
    edges = "\n".join(edge_lines)

    # ---- boundary -------------------------------------------------------------
    boundary = """    airfoil
    {
        // Viscous wall: both surface block wall faces plus the blunt-TE
        // base face of the W_m slab.
        type wall;
        faces
        (
            (0 1 11 10)     // S_u wall face (upper surface)
            (2 0 10 12)     // S_l wall face (lower surface)
            (1 2 12 11)     // W_m left face (TE base)
        );
    }

    farfield
    {
        // C-arc plus the straight top/bottom far boundaries; freestream
        // mixed conditions in 0/ handle inflow/outflow per face.
        type patch;
        faces
        (
            (3 4 14 13)     // S_u far face (arc + upper straight)
            (5 3 13 15)     // S_l far face (arc + lower straight)
            (4 8 18 14)     // W_u top
            (5 9 19 15)     // W_l bottom
        );
    }

    outlet
    {
        // x = 26c plane: pure outflow at every AoA in the sweep.
        type patch;
        faces
        (
            (6 8 18 16)     // W_u outlet
            (7 6 16 17)     // W_m outlet
            (7 9 19 17)     // W_l outlet
        );
    }

    frontAndBack
    {
        // The OpenFOAM 2D convention: empty z faces, no equations normal
        // to them; one cell thick by construction.
        type empty;
        faces
        (
            (0 1 4 3)   (10 11 14 13)    // S_u
            (2 0 3 5)   (12 10 13 15)    // S_l
            (1 6 8 4)   (11 16 18 14)    // W_u
            (7 2 5 9)   (17 12 15 19)    // W_l
            (2 7 6 1)   (12 17 16 11)    // W_m
        );
    }"""

    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
|  blockMeshDict -- structured 2D C-grid, LS(1)-0413 blunt TE                 |
|  GENERATED by scripts/build_validation_case.py -- do not hand-edit          |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Re = {_fmt(plan.re)}, refinement level {plan.level}. Chord c = 1 m, far
// field {_fmt(R)} c, wake {_fmt(WAKE_LENGTH)} c, z thickness {_fmt(Z_THICKNESS)} m (one cell, empty).
// First wall-normal cell {_fmt(plan.h1)} m (y+ = 1 chain in
// scripts/first_cell_height.py, scaled 1/sqrt(2) per level), wall-normal
// ratio {_fmt(plan.normal.ratio)} over {plan.normal.n} cells ({plan.layers_in_d99} inside the
// d99 estimate {_fmt(plan.d99)} m; spec floor {MIN_BL_LAYERS}). Wake ratio {_fmt(plan.wake.ratio)}
// over {plan.wake.n} cells from the TE spacing {_fmt(plan.d_te)} m. Blunt base:
// {plan.n_base} cells, symmetric two-zone grading from h1 at both corners
// (ratio {_fmt(plan.base.ratio)}). Total {plan.n_cells_total} cells.

scale   1.0;

vertices
(
{vertices}
);

blocks
(
{blocks}
);

edges
(
{edges}
);

boundary
(
{boundary}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
    return header


# =============================================================================
#  AoA frame rotation, inlet turbulence chain, BL sample lines
# =============================================================================

def aoa_directions(alpha_rad: float) -> Dict[str, Tuple[float, float, float]]:
    """Wind-frame unit vectors for a freestream rotated by alpha.

    The mesh stays at zero incidence; the freestream vector tilts. Drag is
    measured ALONG the freestream and lift 90 deg counter-clockwise from it
    (z up out of the 2D plane), so:
        dragDir = ( cos a,  sin a, 0)
        liftDir = (-sin a,  cos a, 0)
    Orthonormal by construction; tests/test_case_gen.py verifies.
    """
    ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)
    return {
        "drag": (ca, sa, 0.0),
        "lift": (-sa, ca, 0.0),
    }


def mack_tu_from_ncrit(ncrit: float) -> float:
    """Freestream Tu (fraction) from Mack's e^N correlation.

    Mack (1977): Ncrit = -8.43 - 2.4 ln(Tu) with Tu as a FRACTION; inverted
    here so the RANS inlet sees the same disturbance environment as the
    XFOIL e^N anchor at the matching Ncrit (9 -> Tu ~ 0.0007 = 0.07%).
    """
    return math.exp(-(8.43 + ncrit) / 2.4)


def langtry_rethetat(tu_percent: float) -> float:
    """Freestream Re_theta_t from the Langtry-Menter empirical correlation.

    Zero-pressure-gradient branch (Langtry & Menter, AIAA J. 2009), Tu in
    PERCENT, with the published low/high-Tu split at 1.3% and the model's
    own floor of 20.
    """
    if tu_percent <= 0.0:
        raise ValueError("Tu must be positive")
    if tu_percent <= 1.3:
        re_t = 1173.51 - 589.428 * tu_percent + 0.2196 / tu_percent ** 2
    else:
        re_t = 331.5 * (tu_percent - 0.5658) ** -0.671
    return max(re_t, 20.0)


def inlet_turbulence(u_inf: float, nu: float) -> Dict[str, float]:
    """Full inlet chain: Ncrit -> Tu -> (k, omega, ReThetat), decay-compensated.

    The Mack Tu is the LE-INCIDENT target, but k set at the far boundary
    decays in the freestream before it reaches the airfoil: with U uniform
    along the 25c approach the k transport equation reduces to

        dk/dx = -beta* k omega / U   ->   k(LE) = k(0) exp(-beta* omega L / U)

    so the BOUNDARY value is the target divided by that analytic decay
    factor, evaluated with omega at the target state (k_target at the
    nut/nu = 10 eddy-viscosity ratio -- top of the Langtry-Menter 1..10
    band, the slowest-decay end). The boundary omega is then recomputed
    from the boosted k so the inlet eddy-viscosity ratio stays at the
    documented 10. Returned values:

        k_target / omega_target -- the LE-incident pair the decay model
                                   integrates toward
        decay                   -- exp(-beta* omega_target 25c / U)
        k / omega               -- the boundary-condition pair actually
                                   written into 0/k and 0/omega
        rethetat                -- Langtry-Menter correlation at the TARGET
                                   (LE-incident) Tu, i.e. the Mack value
    """
    tu = mack_tu_from_ncrit(NCRIT)
    k_target = 1.5 * (tu * u_inf) ** 2
    omega_target = k_target / (nu * TURB_VISC_RATIO)

    # Analytic freestream decay over the approach length (25 chords).
    approach = FARFIELD_RADIUS * CHORD
    decay = math.exp(-BETA_STAR * omega_target * approach / u_inf)

    # Pre-boost the boundary k so the decayed LE value lands on the target;
    # boundary omega keeps the nut/nu = 10 definition on the boosted k.
    k_inlet = k_target / decay
    omega_inlet = k_inlet / (nu * TURB_VISC_RATIO)
    return {
        "tu": tu,
        "tu_percent": 100.0 * tu,
        "k_target": k_target,
        "omega_target": omega_target,
        "decay": decay,
        "k": k_inlet,
        "omega": omega_inlet,
        "rethetat": langtry_rethetat(100.0 * tu),
    }


def sampling_set_entries(coords: np.ndarray, alpha_rad: float,
                         d99: float) -> str:
    """Render the wall-normal BL sample-line entries for system/sampling.

    Lines anchor on the SUCTION surface: upper for alpha >= 0, lower for
    negative AoA (the spec sweep dips to -4 deg). Each station gets TWO
    lines along the local outward surface normal (computed from a spline of
    the same resampled loop the mesh was built from), both starting a hair
    off the wall:

      * inner 'bl_x###_inner': BL_INNER_D99_FACTOR * d99 long at
        ~BL_INNER_SPACING pitch. The outer line's 600 points over 0.15c put
        ~2.5e-4 m between samples while the y+ = 1 first cell is ~8e-6 m;
        the entire resolved sublayer would fall between the first two
        samples and the integral thicknesses would lean on the synthetic
        wall anchor alone. The inner line samples it for real.
      * outer 'bl_x###': BL_LINE_LENGTH chords / BL_LINE_POINTS points,
        capturing the BL edge and the outer flow the edge finder needs.

    Set-naming CONTRACT with scripts/extract_bl.py (its module docstring is
    the authority): station as PERCENT digits ('bl_x007' = 7% chord), with
    an optional '_inner' suffix; the extractor MERGES the pair (sorted by
    wall distance, deduplicated) before integrating. ``d99`` is the
    flat-plate TE estimate from the mesh plan -- an upper bound over the
    stations, so every inner line clears its local BL edge.
    """
    upper, lower = _split_loop(coords)
    on_upper = alpha_rad >= 0.0
    surf = upper if on_upper else lower
    side = "upper" if on_upper else "lower"

    # Inner-line discretisation: ~BL_INNER_SPACING pitch over 2x the BL
    # thickness estimate (+1 turns interval count into point count).
    inner_len = BL_INNER_D99_FACTOR * d99
    inner_pts = int(round(inner_len / BL_INNER_SPACING)) + 1

    # y(x) spline of the chosen surface; x is strictly increasing along a
    # resampled surface, so the derivative is well-defined at each station.
    spl = CubicSpline(surf[:, 0], surf[:, 1])
    z_mid = 0.5 * Z_THICKNESS

    entries: List[str] = []
    for xc in BL_STATIONS:
        y = float(spl(xc))
        dy = float(spl(xc, 1))
        # Outward unit normal: rotate the tangent (1, y') by +90 deg for the
        # upper surface, -90 deg for the lower (suction side faces away from
        # the section either way).
        norm = math.hypot(1.0, dy)
        nx, ny = (-dy / norm, 1.0 / norm) if on_upper else (dy / norm, -1.0 / norm)
        sx, sy = xc + BL_WALL_OFFSET * nx, y + BL_WALL_OFFSET * ny
        base_name = f"bl_x{int(round(xc * 100)):03d}"
        # (name suffix, line length in m, point count, role comment)
        for suffix, length, npts, role in (
            ("_inner", inner_len, inner_pts, "inner sublayer line"),
            ("", BL_LINE_LENGTH, BL_LINE_POINTS, "outer line"),
        ):
            ex, ey = xc + length * nx, y + length * ny
            entries.append(
                f"        {base_name}{suffix}\n"
                f"        {{\n"
                f"            // x/c = {xc:.2f} on the {side} (suction) surface -- {role}\n"
                f"            type    uniform;\n"
                f"            axis    distance;\n"
                f"            start   {_vec(sx, sy, z_mid)};\n"
                f"            end     {_vec(ex, ey, z_mid)};\n"
                f"            nPoints {npts};\n"
                f"        }}"
            )
    return "\n".join(entries)


# =============================================================================
#  Naming, summary, template instantiation
# =============================================================================

def subdomains_for_level(level: int) -> int:
    """MPI rank count for one refinement level (2/4/8 at lvl 0/1/2).

    Single authority for the decomposition width: the value lands in
    system/decomposeParDict via the @N_SUBDOMAINS@ token, and Allrun reads
    it back with foamDictionary, so builder/dictionary/mpirun can never
    drift apart. Rationale for the doubling lives at SUBDOMAINS_BY_LEVEL.
    """
    if level not in SUBDOMAINS_BY_LEVEL:
        raise ValueError(f"refinement level must be 0, 1 or 2, got {level}")
    return SUBDOMAINS_BY_LEVEL[level]


def _fmt_re_tag(re_target: float) -> str:
    """Compact Re tag for names/READMEs: 3e6, 1.5e6; plain below 1e6.

    Stays parseable by argparse's float() so the rebuild command printed in
    each case README round-trips.
    """
    if re_target >= 1.0e6:
        return f"{re_target / 1.0e6:g}e6"
    return f"{re_target:g}"


def case_name(aoa_deg: float, re_target: float, level: int) -> str:
    """Directory name val2d_aoa{A}_re{R}_lvl{L}; '-' -> 'm', '.' -> 'p' so
    the name stays filesystem- and shell-safe on every platform."""
    a = f"{aoa_deg:g}".replace("-", "m").replace(".", "p")
    return f"val2d_aoa{a}_re{_fmt_re_tag(re_target)}_lvl{level}"


def mesh_summary(plan: MeshPlan) -> str:
    """ASCII sizing report: the 'achieved expansion/grading numbers' the
    milestone requires printed, reused verbatim in the case README."""
    z = plan.zones
    lines = [
        f"level {plan.level} C-grid, Re = {_fmt_re_tag(plan.re)}, "
        f"U = {plan.u_inf:.3f} m/s, Mach = {plan.mach:.3f}",
        f"first wall cell      : {plan.h1:.4e} m "
        f"(correlation y1 = {plan.y1_correlation:.4e} m / sqrt(2)^{plan.level})",
        f"wall-normal run      : {plan.normal.n} cells, ratio "
        f"{plan.normal.ratio:.4f} (cap {GROWTH_CAP}), grading "
        f"{plan.normal.grading:.1f}",
        f"BL layer budget      : {plan.layers_in_d99} cells inside d99 = "
        f"{plan.d99 * 1000:.2f} mm (spec floor {MIN_BL_LAYERS})",
        f"surface cells/side   : {plan.n_surf}  zones "
        f"[{z[0].n_cells}/{z[1].n_cells}/{z[2].n_cells}], "
        f"d_LE = {plan.d_le:.2e} m, d_TE = {plan.d_te:.2e} m, "
        f"max zone ratio {max(zz.ratio for zz in z):.4f}",
        f"wake run             : {plan.wake.n} cells, ratio "
        f"{plan.wake.ratio:.4f}, grading {plan.wake.grading:.1f}, "
        f"first = TE spacing",
        f"blunt-TE base        : {plan.n_base} cells across "
        f"{2.0 * plan.base.length:.4g} m, symmetric two-zone grading, "
        f"corner cells = h1, ratio {plan.base.ratio:.4f} (cap {GROWTH_CAP})",
        f"total cells          : {plan.n_cells_total}",
    ]
    return "\n".join(lines)


def fill_tokens(text: str, tokens: Dict[str, str], path: Path) -> str:
    """Replace every @NAME@ in `text`; unknown or leftover tokens are build
    failures, never warnings (a half-filled dictionary must not run)."""
    def _sub(m: regex.Match) -> str:
        key = m.group(0)[1:-1]
        if key not in tokens:
            raise KeyError(f"{path}: template token @{key}@ has no value")
        return tokens[key]

    out = TOKEN_RE.sub(_sub, text)
    leftover = TOKEN_RE.findall(out)
    if leftover:
        raise ValueError(f"{path}: unresolved tokens after fill: {leftover}")
    return out


def instantiate_case(case_dir: Path, tokens: Dict[str, str],
                     blockmesh_text: str, force: bool = False) -> List[Path]:
    """Copy the template into `case_dir`, filling tokens and swapping the
    blockMeshDict stub for the generated mesh dictionary.

    Every file is written with LF endings (the case runs under WSL; bash in
    particular rejects CRLF Allrun scripts). Returns the written paths.
    """
    if case_dir.exists():
        if not force:
            raise FileExistsError(
                f"{case_dir} already exists (use --force to rebuild)")
        shutil.rmtree(case_dir)

    written: List[Path] = []
    for src in sorted(TEMPLATE_DIR.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(TEMPLATE_DIR)
        dst = case_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if rel.as_posix() == "system/blockMeshDict":
            # The stub is replaced wholesale by the generated mesh.
            content = blockmesh_text
        else:
            content = fill_tokens(src.read_text(encoding="utf-8"),
                                  tokens, src)

        # newline="\n" pins LF on Windows; OpenFOAM tolerates CRLF but bash
        # does not, and mixed endings make diffs noisy across the WSL line.
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        written.append(dst)
    return written


# =============================================================================
#  Top-level build
# =============================================================================

def build_case(aoa_deg: float, re_target: float, level: int,
               yaml_path: Path, out_root: Path,
               force: bool = False) -> Tuple[Path, MeshPlan]:
    """Build one complete validation case; returns (case_dir, plan)."""
    ac = load_aircraft(yaml_path)
    nu = ac.atmosphere.kinematic_viscosity
    rho = ac.atmosphere.density
    temperature = ac.atmosphere.temperature

    # Geometry: the committed LS(1)-0413 set, blunt TE kept (real 0.0055c
    # base -- transition study, the as-built TE matters; see module + case
    # README for the XFOIL sharp-TE / trust-deltas note).
    airfoil_path = REPO / ac.wing.airfoil_file
    coords = resample_airfoil(load_airfoil(airfoil_path),
                              n_points=N_AIRFOIL_POINTS, te="blunt")

    # Blunt-TE base opening MEASURED from the loop being meshed (~0.0055c
    # for the as-built section) -- the base-slab grading is solved against
    # this exact value, never a hard-coded nominal.
    upper, lower = _split_loop(coords)
    base_height = float(upper[-1, 1] - lower[-1, 1])

    # Mesh plan + generated dictionary.
    plan = plan_cgrid(re_target, nu, rho, temperature, level,
                      base_height=base_height)
    blockmesh_text = generate_blockmeshdict(coords, plan)

    # Freestream vector and wind-frame coefficient axes for this AoA.
    alpha = math.radians(aoa_deg)
    dirs = aoa_directions(alpha)
    ux, uy = plan.u_inf * dirs["drag"][0], plan.u_inf * dirs["drag"][1]

    # Inlet turbulence chain (Ncrit 9 -> Tu -> k/omega/ReThetat).
    turb = inlet_turbulence(plan.u_inf, nu)

    name = case_name(aoa_deg, re_target, level)
    tokens = {
        "CASE_NAME": name,
        "AOA_DEG": f"{aoa_deg:g}",
        "RE": _fmt_re_tag(re_target),
        "LEVEL": str(level),
        "U_MAG": f"{plan.u_inf:.6g}",
        "UX": f"{ux:.8g}",
        "UY": f"{uy:.8g}",
        "NU": f"{nu:.6g}",
        "RHO": f"{rho:.6g}",
        "MACH": f"{plan.mach:.3f}",
        "TU_PERCENT": f"{turb['tu_percent']:.4g}",
        # Boundary-condition pair (decay-compensated) plus the audit trail:
        # the LE-incident targets and the analytic decay factor are written
        # into the 0/k and 0/omega comments so a reviewer can re-derive
        # K_INF = K_LE / DECAY_FACTOR from the case files alone.
        "K_INF": f"{turb['k']:.6g}",
        "OMEGA_INF": f"{turb['omega']:.6g}",
        "K_LE": f"{turb['k_target']:.6g}",
        "OMEGA_LE": f"{turb['omega_target']:.6g}",
        "DECAY_FACTOR": f"{turb['decay']:.6g}",
        "RETHETAT_INF": f"{turb['rethetat']:.6g}",
        "DRAGX": f"{dirs['drag'][0]:.8g}",
        "DRAGY": f"{dirs['drag'][1]:.8g}",
        "LIFTX": f"{dirs['lift'][0]:.8g}",
        "LIFTY": f"{dirs['lift'][1]:.8g}",
        "AREF": f"{CHORD * Z_THICKNESS:.6g}",
        "N_SUBDOMAINS": str(subdomains_for_level(level)),
        "MESH_SUMMARY": mesh_summary(plan),
        "SAMPLING_SETS": sampling_set_entries(coords, alpha, plan.d99),
    }

    case_dir = out_root / name
    instantiate_case(case_dir, tokens, blockmesh_text, force=force)
    return case_dir, plan


def main() -> int:
    """CLI entry: build one case and print the sizing report (ASCII only)."""
    p = argparse.ArgumentParser(
        description="Phase-1 2D validation case builder (C-grid + template)")
    # '--alpha' is the spelling the M1 driver (scripts/run_validation.py)
    # uses in its sibling-script contract; both names land on the same dest.
    p.add_argument("--aoa", "--alpha", dest="aoa", type=float, required=True,
                   help="angle of attack in degrees (freestream rotated; "
                        "mesh stays at zero incidence)")
    p.add_argument("--re", type=float, required=True,
                   help="chord Reynolds number (c = 1 m, U = Re*nu/c)")
    p.add_argument("--level", type=int, default=0, choices=(0, 1, 2),
                   help="refinement level: counts x sqrt(2)^level (default 0)")
    p.add_argument("--yaml", type=Path, default=REPO / "aircraft.yaml",
                   help="aircraft parameter file (default: repo aircraft.yaml)")
    # '--out' is the M1 driver's spelling for the same knob (see the
    # sibling-script contract in scripts/run_validation.py).
    p.add_argument("--out-root", "--out", dest="out_root", type=Path,
                   default=RUNS_DIR,
                   help="case output root (default: cases/runs)")
    p.add_argument("--force", action="store_true",
                   help="rebuild over an existing case directory")
    a = p.parse_args()

    case_dir, plan = build_case(a.aoa, a.re, a.level, a.yaml, a.out_root,
                                force=a.force)

    # Sizing report: the achieved expansion/grading numbers, per milestone.
    print("=" * 72)
    print(f"case built: {case_dir}")
    print("=" * 72)
    print(mesh_summary(plan))
    # Compressibility bookkeeping for the Re 6M trend point: the documented
    # decision is to KEEP c = 1 m (see module docstring + case README).
    if plan.mach > 0.2:
        print(f"NOTE: Mach {plan.mach:.3f} exceeds the 0.2 quasi-"
              f"incompressible guideline; kept as a documented trend point "
              f"(see case README).")
    print("=" * 72)
    print("dry-run ready: execute ./Allrun inside WSL2 with ESI OpenFOAM "
          "v2506 (openfoam2506)")
    # Sibling-script CONTRACT with scripts/run_validation.py: the created
    # case directory is echoed as the LAST stdout line so the driver can
    # pick it up verbatim instead of re-deriving the naming scheme.
    print(case_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
