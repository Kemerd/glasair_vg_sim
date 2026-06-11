# -*- coding: utf-8 -*-
"""Boundary-layer extraction from wall-normal sample sets (spec Phase-1 #5).

Consumes the wall-normal velocity profiles that the 2D validation case
template writes via a 'sets' function object at fixed suction-surface
stations, and turns them into the table that sizes the vortex generators:
per station x/c it computes

    delta99   -- wall distance where |U| first reaches 0.99 * U_edge
    delta*    -- displacement thickness, integral (1 - u/Ue) dn
    theta     -- momentum thickness, integral (u/Ue)(1 - u/Ue) dn
    H         -- shape factor delta*/theta

and writes results/bl_table_{case}.csv plus a delta99-vs-x/c figure with
one curve per AoA. Integrals use trapezoid quadrature on the sampled
profile; |U| stands in for the wall-parallel velocity component, which is
exact in 2D up to the (small inside the BL) wall-normal component -- the
sample lines are wall-normal by construction.

SAMPLING-LAYOUT CONTRACT (coordinated with the case template in
cases/validation_2d/template/system/): each station is a line set named
'bl_x<station>' sampling U from the wall outward, where <station> encodes
the chord fraction either as percent digits ('bl_x007' = 7% c) or with 'p'
or '.' as the decimal mark ('bl_x0p07', 'bl_x0.07'). Both OpenFOAM
setFormats the template may emit are read:

    raw  ->  <set>_U.xy        whitespace columns: distance + Ux Uy Uz
                               (axis distance) or x y z + Ux Uy Uz (axis xyz);
                               multi-field files like <set>_p_U.xy are
                               handled by deriving the column layout from
                               the field tokens in the file name
    csv  ->  <set>_U.csv       self-describing comma header (x,y,z,U_x,...
                               or distance,U_x,...)

Profiles are read from the LATEST time directory of the sample function
object -- the converged end state is what sizes the VGs. When a station is
present in both formats the csv is preferred (self-describing header beats
filename heuristics).

CASE-SCALE CONVENTION (M1, documented here because the aircraft-scale
column depends on it; mirrored in the case README): the 2D validation case
runs at unit chord c = 1.0 m with the freestream speed chosen to hit the
target Reynolds number, U = Re * nu / c, using nu from aircraft.yaml
(ISA sea level, 1.4607e-5 m^2/s):

    Re 1.5e6 -> U = 21.9 m/s  (M ~ 0.064)
    Re 3.0e6 -> U = 43.8 m/s  (M ~ 0.129)
    Re 6.0e6 -> U = 87.6 m/s  (M ~ 0.258)   <-- exceeds the M < 0.2
                                                 quasi-incompressible guide

Decision: Re 6e6 is KEPT at c = 1 m as a trend point rather than scaling
the chord. simpleFoam is strictly incompressible, so the solver itself is
Mach-free -- the nominal U only labels how the real (compressible) flow at
that Re would deviate from the incompressible answer, a few-percent
Prandtl-Glauert-level effect at M 0.26. The VG design conditions (approach,
Re 1.5-3e6) sit safely below M 0.2; Re 6e6 exists to anchor the Re trend
against XFOIL/NASA, which is standard practice for simpleFoam airfoil
studies. Scaling the chord to lower the nominal Mach would change nothing
in the incompressible solution at fixed Re.

AIRCRAFT-SCALE COLUMN: at matched Re and AoA the nondimensional profile
delta99/c at a given x/c carries over from the c = 1 m case to the real
wing, so the aileron-station boundary-layer thickness is

    delta99_aircraft = delta99_case * (c_aileron_mid / c_case)

with c_aileron_mid = wing.aileron.chord_at_mid_station from aircraft.yaml
(DXF-measured, 0.9022 m) and c_case = 1.0 m. The VG heights are then
h = {0.2, 0.5, 1.0} * delta99_aircraft per the Lin micro-VG guidance
(vg_defaults.height_per_delta in aircraft.yaml).

Library API:
    extract_case(case_dir, ...)   -> list[BLStation]
    compute_bl_metrics(n, umag)   -> BLMetrics
    write_bl_table(...)           -> Path to results/bl_table_{case}.csv
    plot_delta99(...)             -> Path to the figure

CLI:  python scripts/extract_bl.py CASE_DIR [CASE_DIR ...]
          [--out-dir results] [--case-chord 1.0] [--aircraft aircraft.yaml]
          [--no-plot]
One table per case directory; one combined figure (curve per AoA, parsed
from case names like 'aoa_+04' / 'alpha6'). Exit 0 on success, 2 when a
case yields no stations. Plain-ASCII console output only.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python scripts/extract_bl.py", Python puts scripts/ on
#  sys.path rather than the repo root, hiding the geometry package. Inserting
#  this file's grandparent mirrors conftest.py and keeps the script runnable
#  from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geometry.units import load_aircraft  # noqa: E402

# numpy renamed trapz -> trapezoid in 2.x; this shim keeps the module working
# on the pinned 1.26 environment and on any future upgrade without edits.
_TRAPZ = getattr(np, "trapezoid", np.trapz)

# The M1 unit-chord convention (see module docstring): all delta values come
# out of the case in meters AT this chord; the aircraft-scale column rescales
# by the YAML-measured aileron-station chord.
CASE_CHORD_M_DEFAULT = 1.0

# Station token inside set/file names: 'bl' then 'x' then digits with an
# optional 'p' or '.' decimal mark. Examples matched: bl_x007_U.xy,
# bl_x0p07_U.csv, blProfiles_x0.35_U.xy, BL-x100_U.xy.
STATION_RE = re.compile(r"(?i)(?:^|[_\-])bl[a-z]*[_\-]?x(?P<tok>\d+(?:[p.]\d+)?)")

# Velocity must be among the sampled fields; raw set files always carry the
# field names in the stem ('<set>_U.xy', '<set>_p_U.xy', ...).
_HAS_U_RE = re.compile(r"_U(?:_|$)")

# AoA token in case-directory names, e.g. 'aoa_+04', 'aoa_p04', 'aoa-m2',
# 'alpha6p5'. Filesystem-safe sign encodings: 'm'/'n' mean minus, 'p' means
# plus; a 'p' BETWEEN digits is the decimal mark ('6p5' = 6.5), which the
# sign/number split below disambiguates naturally (sign is consumed only
# when it precedes the first digit).
AOA_RE = re.compile(
    r"(?i)(?:aoa|alpha)[_\-]?(?P<sign>[mnp+-]?)(?P<num>\d+(?:[p.]\d+)?)"
)


# =============================================================================
#  Result containers
# =============================================================================
@dataclass
class BLMetrics:
    """Boundary-layer integral quantities for one wall-normal profile (SI)."""

    u_edge: float        # edge velocity from the guarded max-|U| search [m/s]
    delta99: float       # 0.99*u_edge crossing height [m]
    delta_star: float    # displacement thickness [m]
    theta: float         # momentum thickness [m]
    H: float             # shape factor delta_star / theta
    edge_index: int      # profile index used as the BL edge / integral bound
    n_points: int        # sample count after cleanup (incl. wall anchor)


@dataclass
class BLStation:
    """One suction-surface station: location plus its profile metrics."""

    x_over_c: float
    metrics: BLMetrics
    source: Path         # the sample-set file the profile came from


# =============================================================================
#  Profile math
# =============================================================================
def find_edge(umag: np.ndarray, drop_tol: float = 0.02) -> Tuple[int, float]:
    """Locate the boundary-layer edge: max |U| with a monotonicity guard.

    The naive edge estimator is the global maximum of |U| along the
    wall-normal line. That fails when the line leaves the boundary layer
    and keeps sampling the outer inviscid flow: near the suction peak |U|
    DECREASES outboard of the BL edge and can rise again further out from
    domain/curvature effects, so a far-field overshoot would drag the edge
    (and every integral bound) away from the physical layer.

    Guard: walk outward tracking the running maximum; the first point that
    falls more than ``drop_tol`` (default 2%) below the running maximum
    marks the start of the outer-flow decay, and the search is truncated
    there. The edge is the (first) maximum of |U| within the untruncated
    inner region. A profile that never violates the guard (flat-plate-like
    monotone rise to a plateau) uses its global maximum directly.
    """
    running_max = np.maximum.accumulate(umag)
    below = umag < (1.0 - drop_tol) * running_max
    if below.any():
        # Truncate at the first guarded drop; the edge lives at or before it.
        cut = int(np.argmax(below))            # first True index
        search = umag[:cut]
    else:
        search = umag
    if search.size == 0:
        # Degenerate: first sample already violates (cannot happen with the
        # wall anchor at u=0, kept for safety against pathological input).
        search = umag[:1]
    i_edge = int(np.argmax(search))            # FIRST occurrence of the max
    return i_edge, float(search[i_edge])


def compute_bl_metrics(n: np.ndarray, umag: np.ndarray,
                       drop_tol: float = 0.02) -> BLMetrics:
    """Compute delta99 / delta* / theta / H from one wall-normal profile.

    ``n`` is wall distance [m], ``umag`` is |U| [m/s], both wall -> outward.
    Cleanup performed here so every caller gets identical treatment:
      * sort by n and drop duplicate heights (set writers occasionally emit
        the cell-center and face point at the same distance);
      * enforce the no-slip anchor: when the first sample sits off the wall
        (sets often start at the first cell center, not the face) a (0, 0)
        point is prepended so the trapezoid does not lose the near-wall
        area -- without it delta* / theta bias low by up to one cell.

    Integrals run from the wall to the detected edge index: beyond the edge
    u = Ue makes the integrands vanish, while including the outer inviscid
    region (where |U| drops below Ue again) would re-inflate (1 - u/Ue) with
    flow that is not boundary layer.
    """
    # ---- cleanup: ascending, unique heights -----------------------------------
    order = np.argsort(n, kind="stable")
    n = np.asarray(n, dtype=float)[order]
    umag = np.asarray(umag, dtype=float)[order]
    keep = np.ones(n.size, dtype=bool)
    keep[1:] = np.diff(n) > 0.0          # drop exact-duplicate heights
    n, umag = n[keep], umag[keep]
    if n.size < 4:
        raise ValueError(f"profile too short after cleanup: {n.size} points")

    # ---- no-slip wall anchor ---------------------------------------------------
    if n[0] > 1e-12:
        n = np.concatenate(([0.0], n))
        umag = np.concatenate(([0.0], umag))

    # ---- edge detection (guarded max |U|) -------------------------------------
    i_edge, u_e = find_edge(umag, drop_tol=drop_tol)
    if u_e <= 0.0:
        raise ValueError("profile edge velocity is non-positive; bad sample")

    # ---- delta99: first 0.99*Ue crossing, linearly interpolated ---------------
    target = 0.99 * u_e
    above = np.nonzero(umag[: i_edge + 1] >= target)[0]
    if above.size == 0:
        # Cannot happen (u_e itself satisfies the target) but guard anyway.
        raise ValueError("0.99*Ue crossing not found inside the profile")
    k = int(above[0])
    if k == 0:
        delta99 = float(n[0])
    else:
        # Linear interpolation between the bracketing samples gives sub-cell
        # resolution; the profile is locally smooth at the 99% level.
        frac = (target - umag[k - 1]) / (umag[k] - umag[k - 1])
        delta99 = float(n[k - 1] + frac * (n[k] - n[k - 1]))

    # ---- integral thicknesses (trapezoid to the edge) --------------------------
    nn = n[: i_edge + 1]
    ratio = umag[: i_edge + 1] / u_e
    delta_star = float(_TRAPZ(1.0 - ratio, nn))
    theta = float(_TRAPZ(ratio * (1.0 - ratio), nn))
    if theta <= 0.0:
        raise ValueError("non-positive momentum thickness; degenerate profile")

    return BLMetrics(u_edge=u_e, delta99=delta99, delta_star=delta_star,
                     theta=theta, H=delta_star / theta,
                     edge_index=i_edge, n_points=int(n.size))


# =============================================================================
#  Sample-set file readers (raw .xy and csv)
# =============================================================================
def parse_station_token(name: str) -> Optional[float]:
    """Decode the x/c station from a set or file name; None when absent.

    Encodings accepted (see module docstring): explicit decimals via '.' or
    'p' ('x0p07' -> 0.07), or plain digits read as PERCENT of chord when the
    raw value exceeds 1 ('x007' -> 7 -> 0.07, 'x100' -> 1.0). Values that
    remain outside [0, 1] after the percent fallback are rejected -- a
    station off the chord is a naming bug, not data.
    """
    m = STATION_RE.search(name)
    if not m:
        return None
    value = float(m.group("tok").replace("p", "."))
    if value > 1.0 + 1e-9:
        value /= 100.0                    # percent-encoded digits
    if not (0.0 <= value <= 1.0 + 1e-9):
        return None
    return min(value, 1.0)


def _fields_from_stem(stem: str) -> List[str]:
    """Recover the sampled-field tokens from a raw set file name.

    The sets writer names raw files '<setName>_<field1>_<field2>...'. The
    set name itself contains underscores ('bl_x007'), so the split point is
    located via the station token: everything after the token's match is
    the field list ('_U' -> ['U'], '_p_U' -> ['p', 'U']).
    """
    m = STATION_RE.search(stem)
    if not m:
        return []
    rest = stem[m.end():].strip("_")
    return [tok for tok in rest.split("_") if tok]


def _read_raw_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read one raw-format set file -> (wall distance, |U|), both SI.

    Column layout is derived, not assumed: a field list parsed from the file
    name (U occupies 3 columns, scalars 1) determines how many leading
    coordinate columns remain -- 1 (axis distance) or 3 (axis xyz). Files
    whose names defeat the parser fall back to the two unambiguous layouts
    (4 cols = distance+U, 6 cols = xyz+U).
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    ncols = data.shape[1]

    fields = _fields_from_stem(path.stem)
    widths = {f: (3 if f == "U" else 1) for f in fields}
    n_coord = ncols - sum(widths.values()) if fields else -1

    if fields and "U" in fields and n_coord in (1, 3):
        # Coordinates first, then fields in file-name order; U's offset is
        # the coordinate width plus every preceding field's width.
        u_off = n_coord + sum(widths[f] for f in fields[: fields.index("U")])
    elif ncols == 4:
        n_coord, u_off = 1, 1             # distance + Ux Uy Uz
    elif ncols == 6:
        n_coord, u_off = 3, 3             # x y z + Ux Uy Uz
    else:
        raise ValueError(f"{path}: cannot infer column layout "
                         f"({ncols} columns, fields={fields})")

    # Wall distance: the distance column verbatim, or arc distance from the
    # first sampled point when xyz coordinates were written (the line is
    # straight and wall-normal, so |r - r0| IS the wall-normal coordinate).
    if n_coord == 1:
        n_wall = data[:, 0]
    else:
        xyz = data[:, :3]
        n_wall = np.linalg.norm(xyz - xyz[0], axis=1)
    umag = np.linalg.norm(data[:, u_off:u_off + 3], axis=1)
    return n_wall, umag


def _read_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read one csv-format set file -> (wall distance, |U|), both SI.

    The csv writer emits a self-describing header; columns are located by
    NAME so the reader is indifferent to ordering and to extra sampled
    fields. Velocity components match 'U_x'/'Ux'/'U_0' style names;
    coordinates come from a 'distance' column or from x/y/z triples.
    """
    with open(path, "r", encoding="utf-8") as fh:
        header_line = fh.readline()
    names = [t.strip().strip('"').lstrip("#").strip()
             for t in header_line.split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {len(names)} header names but "
                         f"{data.shape[1]} data columns")
    idx = {name.lower(): j for j, name in enumerate(names)}

    # Velocity component columns: tolerate U_x / Ux / U:x / U_0 spellings.
    comp_re = re.compile(r"(?i)^u[_:.]?([xyz012])$")
    u_cols = [j for name, j in idx.items() if comp_re.match(name)]
    if not u_cols:
        raise ValueError(f"{path}: no velocity columns in header {names}")
    umag = np.linalg.norm(data[:, sorted(u_cols)], axis=1)

    # Wall distance: explicit distance column wins; otherwise reconstruct
    # from xyz exactly as the raw reader does.
    if "distance" in idx:
        n_wall = data[:, idx["distance"]]
    elif all(c in idx for c in ("x", "y", "z")):
        xyz = data[:, [idx["x"], idx["y"], idx["z"]]]
        n_wall = np.linalg.norm(xyz - xyz[0], axis=1)
    else:
        raise ValueError(f"{path}: no 'distance' or x/y/z columns in {names}")
    return n_wall, umag


def read_profile(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Format dispatch: one sample-set file -> (wall distance, |U|)."""
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    return _read_raw_xy(path)


# =============================================================================
#  Case-level discovery and extraction
# =============================================================================
def _latest_time_dir(fo_dir: Path) -> Optional[Path]:
    """Pick the numerically-largest time directory under a function-object dir.

    The BL table must describe the CONVERGED state, so only the final write
    is read; intermediate writes are solver history, not geometry input.
    """
    best, best_t = None, None
    for child in fo_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            t = float(child.name)
        except ValueError:
            continue
        if best_t is None or t > best_t:
            best, best_t = child, t
    return best


def discover_station_files(case_dir: Path) -> Dict[float, Path]:
    """Map x/c station -> sample file for one case (latest time, csv first).

    Walks every function-object directory under postProcessing/, keeps the
    latest time directory of each, and collects files that (a) carry a
    parseable station token and (b) sample U. When both formats exist for a
    station the csv wins (self-describing header beats filename heuristics);
    across function objects the first (sorted) provider wins deterministically.
    """
    post = case_dir / "postProcessing"
    if not post.is_dir():
        raise FileNotFoundError(f"{case_dir}: no postProcessing/ directory")

    stations: Dict[float, Path] = {}
    for fo_dir in sorted(p for p in post.iterdir() if p.is_dir()):
        time_dir = _latest_time_dir(fo_dir)
        if time_dir is None:
            continue
        # csv listed before xy so the dict-setdefault below prefers csv.
        candidates = sorted(time_dir.glob("*.csv")) + sorted(time_dir.glob("*.xy"))
        for f in candidates:
            x_over_c = parse_station_token(f.stem)
            if x_over_c is None:
                continue
            # Raw files must name U among their fields; csv headers are
            # checked at read time, but the cheap stem test filters obvious
            # non-velocity samples (e.g. bl_x007_p.xy) for both formats.
            if f.suffix.lower() == ".xy" and not _HAS_U_RE.search(f.stem):
                continue
            stations.setdefault(round(x_over_c, 6), f)
    return stations


def extract_case(case_dir: Union[str, Path],
                 drop_tol: float = 0.02) -> List[BLStation]:
    """Extract every BL station of one case, sorted by x/c."""
    case_dir = Path(case_dir)
    stations = discover_station_files(case_dir)
    results: List[BLStation] = []
    for x_over_c in sorted(stations):
        f = stations[x_over_c]
        n_wall, umag = read_profile(f)
        metrics = compute_bl_metrics(n_wall, umag, drop_tol=drop_tol)
        results.append(BLStation(x_over_c=x_over_c, metrics=metrics, source=f))
    return results


# =============================================================================
#  Outputs: VG-sizing table and delta99 figure
# =============================================================================
def write_bl_table(stations: Sequence[BLStation], case_name: str,
                   out_dir: Union[str, Path],
                   aircraft_yaml: Union[str, Path],
                   case_chord_m: float = CASE_CHORD_M_DEFAULT) -> Path:
    """Write results/bl_table_{case}.csv -- THE table that sizes the VGs.

    Columns (mm for the human-facing thicknesses; x/c and H dimensionless):
        x_over_c, delta99_mm, delta_star_mm, theta_mm, H, delta99_aircraft_mm
    The last column rescales delta99 from the unit-chord case to the wing
    chord at the aileron mid-span station (Re-similarity argument in the
    module docstring); h/delta in {0.2, 0.5, 1.0} of THAT number is the VG
    height sweep.
    """
    # Aircraft-scale chord comes from the master file through the single
    # unit-conversion choke point -- never hard-coded.
    ac = load_aircraft(aircraft_yaml)
    chord_ratio = ac.wing.aileron.chord_at_mid_station / case_chord_m

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bl_table_{case_name}.csv"

    lines = ["x_over_c,delta99_mm,delta_star_mm,theta_mm,H,delta99_aircraft_mm"]
    for st in stations:
        m = st.metrics
        lines.append(
            f"{st.x_over_c:.4f},{m.delta99 * 1e3:.4f},"
            f"{m.delta_star * 1e3:.4f},{m.theta * 1e3:.4f},"
            f"{m.H:.4f},{m.delta99 * chord_ratio * 1e3:.4f}"
        )
    # utf-8 with trailing newline; ASCII content keeps it diff- and
    # Excel-friendly on Windows.
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_aoa_label(case_name: str) -> str:
    """Human label for a case's AoA, parsed from its directory name.

    Recognizes 'aoa'/'alpha' tokens with filesystem-safe sign encodings
    ('m'/'n' = minus) and 'p' decimal marks. Falls back to the raw case
    name so unlabeled cases still plot, just without a pretty legend.
    """
    m = AOA_RE.search(case_name)
    if not m:
        return case_name
    value = float(m.group("num").replace("p", "."))
    if m.group("sign").lower() in ("m", "n", "-"):
        value = -value
    return f"AoA {value:+.1f} deg"


def _aoa_sort_key(case_name: str) -> Tuple[int, float, str]:
    """Sort key putting parseable AoA cases in numeric order, others last.

    String-sorting the labels would order '+10.0' before '+4.0'; sorting on
    the parsed numeric value keeps legends and tables in sweep order.
    """
    m = AOA_RE.search(case_name)
    if not m:
        return (1, 0.0, case_name)
    value = float(m.group("num").replace("p", "."))
    if m.group("sign").lower() in ("m", "n", "-"):
        value = -value
    return (0, value, case_name)


def plot_delta99(per_case: Dict[str, Sequence[BLStation]],
                 out_dir: Union[str, Path],
                 case_chord_m: float = CASE_CHORD_M_DEFAULT) -> Path:
    """Figure: delta99 vs x/c, one curve per case (i.e. per AoA).

    matplotlib is imported lazily with the Agg backend forced so the script
    runs headless (WSL batch post-processing) and never touches a display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    # Sort cases by their parsed numeric AoA so the legend reads in sweep order.
    for case_name in sorted(per_case, key=_aoa_sort_key):
        stations = per_case[case_name]
        x = [st.x_over_c for st in stations]
        d99 = [st.metrics.delta99 * 1e3 for st in stations]
        ax.plot(x, d99, marker="o", linewidth=1.4,
                label=parse_aoa_label(case_name))
    ax.set_xlabel("x/c")
    ax.set_ylabel(f"delta99 [mm]  (case scale, c = {case_chord_m:g} m)")
    ax.set_title("Suction-surface boundary-layer growth (Phase-1 clean section)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bl_delta99_vs_xc.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
#  CLI
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Extract BL tables (and the combined figure) for one or more cases."""
    p = argparse.ArgumentParser(
        description="Boundary-layer extraction from wall-normal sample sets "
                    "(Phase-1 item 5; produces the VG-sizing table)")
    p.add_argument("case_dirs", nargs="+", type=Path,
                   help="OpenFOAM case directories (one per AoA)")
    p.add_argument("--out-dir", type=Path, default=REPO / "results",
                   help="output directory for tables and the figure")
    p.add_argument("--case-chord", type=float, default=CASE_CHORD_M_DEFAULT,
                   help="2D case chord in meters (M1 convention: 1.0)")
    p.add_argument("--aircraft", type=Path, default=REPO / "aircraft.yaml",
                   help="master parameter file (aircraft-scale chord source)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the delta99 figure (tables only)")
    args = p.parse_args(argv)

    ac = load_aircraft(args.aircraft)
    chord_ratio = ac.wing.aileron.chord_at_mid_station / args.case_chord

    per_case: Dict[str, List[BLStation]] = {}
    failures = 0
    for case_dir in args.case_dirs:
        try:
            stations = extract_case(case_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR [{case_dir.name}]: {exc}")
            failures += 1
            continue
        if not stations:
            print(f"ERROR [{case_dir.name}]: no BL sample stations found "
                  f"under postProcessing/ (expected sets named bl_x<station>)")
            failures += 1
            continue
        per_case[case_dir.name] = stations
        table = write_bl_table(stations, case_dir.name, args.out_dir,
                               args.aircraft, case_chord_m=args.case_chord)

        # Console table mirroring the CSV so a remote shell session shows
        # the sizing numbers without opening files.
        print(f"\ncase {case_dir.name}  ({parse_aoa_label(case_dir.name)})  "
              f"-> {table}")
        hdr = (f"  {'x/c':>6} {'d99_mm':>9} {'dstar_mm':>9} {'theta_mm':>9} "
               f"{'H':>7} {'d99_ac_mm':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for st in stations:
            m = st.metrics
            print(f"  {st.x_over_c:>6.3f} {m.delta99 * 1e3:>9.3f} "
                  f"{m.delta_star * 1e3:>9.3f} {m.theta * 1e3:>9.3f} "
                  f"{m.H:>7.3f} {m.delta99 * chord_ratio * 1e3:>10.3f}")

        # VG-height preview at the station nearest the sweep-center chordwise
        # position (vg_defaults.wing.chord_position_frac): the h/delta ladder
        # from aircraft.yaml applied to the aircraft-scale delta99. Purely
        # informational -- the sweep YAML remains the source of record.
        x_vg = ac.vg_defaults.wing.chord_position_frac
        nearest = min(stations, key=lambda s: abs(s.x_over_c - x_vg))
        d99_ac = nearest.metrics.delta99 * chord_ratio
        ladder = ", ".join(
            f"h/d={f:g}: {f * d99_ac * 1e3:.2f} mm"
            for f in ac.vg_defaults.height_per_delta
        )
        print(f"  VG height ladder @ x/c={nearest.x_over_c:.3f} "
              f"(nearest {x_vg:g}c): {ladder}")

    if per_case and not args.no_plot:
        fig_path = plot_delta99(per_case, args.out_dir,
                                case_chord_m=args.case_chord)
        print(f"\nfigure written: {fig_path}")

    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
