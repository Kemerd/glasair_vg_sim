# -*- coding: utf-8 -*-
"""Suction-surface skin-friction extraction for the transition gate (M1).

This is the DATA PRODUCER for gate (c) of the Phase-1 validation
(validation/compare_gate.py, transition_xtr): it converts the wall shear
stress that the case template's ``wallShearStress`` function object dumps on
the airfoil patch into the upper-surface skin-friction curve

    Cf(x/c) = |tau_w| / (0.5 * U_inf^2)

written as ``CASE_DIR/postProcessing/cf_upper.csv`` (columns ``x_c,cf``) --
exactly the file scripts/run_validation.py globs for and feeds to the shared
transition detector ``validation.compare_gate.detect_transition_from_cf``.

RHO CONVENTION (read this before touching the formula): simpleFoam /
pimpleFoam are incompressible solvers working with KINEMATIC quantities, so
the ``wallShearStress`` function object writes tau_w / rho in m^2/s^2, not a
true stress in Pa. The matching kinematic dynamic pressure is q/rho =
0.5 * U_inf^2, hence Cf = |tau_w/rho| / (0.5 * U_inf^2) with NO density
anywhere -- rho cancels exactly. (A dimensional check: [m^2/s^2] over
[m^2/s^2] = dimensionless, as Cf must be.)

INPUT CONTRACT (coordinated with the template's system/wallShearStressObj):
the ``wallCf`` surfaces function object writes the airfoil-patch wall shear
at face centres, ``surfaceFormat raw``, under
``postProcessing/wallCf/<time>/wallShearStress_*.raw``. The raw surface
writer emits '#'-prefixed header lines followed by whitespace-separated
numeric rows ``x y z tau_x tau_y tau_z`` (one row per face centre). Only the
LATEST time directory is read -- Cf must describe the converged end state.
A VTK surface dump (.vtk/.vtp with a ``wallShearStress`` cell or point
array) is accepted as a fallback through a GUARDED pyvista import, so a
template switched to ``surfaceFormat vtk`` keeps working where pyvista is
installed without making pyvista a hard dependency of the pipeline.

SUCTION-SIDE SELECTION: the patch dump carries upper surface, lower surface
and the blunt-TE base together. Base faces sit on the vertical x = c plane
and are dropped by their x coordinate; the remaining faces are split into
upper/lower against a binned midline computed FROM THE FACE CENTRES
THEMSELVES (per x-bin midpoint of the y extremes), which tracks the camber
line without needing the geometry files. The suction side is the upper
surface for alpha >= 0 and the lower surface for negative AoA -- the same
convention as the template's BL sample lines -- with both U_inf and alpha
read from the case's own 0/U freestream vector, never hard-coded.

CLI:  python scripts/extract_cf.py CASE_DIR [CASE_DIR ...]
Exit 0 when every case produced a curve, 2 when any failed (missing dump,
unparseable 0/U, ...). Plain-ASCII console output; CSV written UTF-8/LF.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python scripts/extract_cf.py", Python puts scripts/ on
#  sys.path rather than the repo root, which would hide the validation
#  package. Inserting this file's grandparent mirrors conftest.py and keeps
#  the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Output-file contract with scripts/run_validation.py (its extract_transition
# glob looks for cf*upper*.csv under postProcessing/).
CF_CSV_NAME = "cf_upper.csv"

# Surface-dump extensions in preference order: raw first (the template's
# format, parseable with numpy alone), VTK flavors as the pyvista fallback.
SURFACE_EXTS = (".raw", ".vtk", ".vtp")

# Blunt-TE base faces live on the vertical x = chord plane; anything within
# this distance of the maximum x of the dump is treated as base, not surface.
# The finest TE surface spacing is O(1e-3) chord, so 1e-6 cannot eat a
# genuine surface face while still catching every base-face centre exactly.
BASE_X_TOL = 1.0e-6

# x-bin count for the camber-midline split. Both surfaces share the same
# streamwise clustering (the mesh generator grades them identically), so
# every populated bin sees faces from both sides at any practical count.
N_SPLIT_BINS = 32

# 0/U freestream vector, as written by the case builder:
#   internalField   uniform (43.7 3.06 0);
_U_INLINE_RE = re.compile(
    r"internalField\s+uniform\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s*\)")


# =============================================================================
#  Case freestream (U_inf magnitude + AoA) from the case's own 0/U
# =============================================================================
def parse_freestream(case_dir: Path) -> Tuple[float, float]:
    """Read (U_inf, alpha_rad) from CASE_DIR/0/U.

    The builder writes the freestream as a uniform internalField vector
    rotated by the AoA (mesh fixed at zero incidence), so the magnitude IS
    U_inf and atan2(Uy, Ux) IS alpha. Reading the case's own dictionary
    keeps this script free of any Re/speed assumption.
    """
    u_path = Path(case_dir) / "0" / "U"
    if not u_path.is_file():
        raise FileNotFoundError(f"{case_dir}: no 0/U file (case incomplete?)")
    m = _U_INLINE_RE.search(u_path.read_text(encoding="utf-8",
                                             errors="replace"))
    if not m:
        raise ValueError(f"{u_path}: no 'internalField uniform (..)' vector")
    ux, uy = float(m.group(1)), float(m.group(2))
    u_inf = math.hypot(ux, uy)
    if u_inf <= 0.0:
        raise ValueError(f"{u_path}: zero freestream magnitude")
    return u_inf, math.atan2(uy, ux)


# =============================================================================
#  Surface-dump discovery (latest time of the wall-shear function object)
# =============================================================================
def _latest_time_dir(fo_dir: Path) -> Optional[Path]:
    """Numerically-largest time directory under one function-object dir.

    Same converged-end-state rule as the BL extraction: only the final write
    is data; intermediate writes are solver history.
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


def find_wall_shear_file(case_dir: Path) -> Path:
    """Locate the latest-time wallShearStress surface dump of one case.

    Function-object directories named 'wallCf' (the template's writer) are
    searched first; any other postProcessing/ directory holding a
    wallShearStress surface file is accepted as a fallback so a renamed
    writer does not silently disable the transition gate. Within one time
    directory the extension preference is raw > vtk > vtp.
    """
    post = Path(case_dir) / "postProcessing"
    if not post.is_dir():
        raise FileNotFoundError(f"{case_dir}: no postProcessing/ directory")

    # Template writer first, then everything else in deterministic order.
    fo_dirs = sorted((p for p in post.iterdir() if p.is_dir()),
                     key=lambda p: (p.name != "wallCf", p.name))
    for fo_dir in fo_dirs:
        time_dir = _latest_time_dir(fo_dir)
        if time_dir is None:
            continue
        for ext in SURFACE_EXTS:
            hits = sorted(time_dir.glob(f"wallShearStress*{ext}"))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"{case_dir}: no wallShearStress surface dump under postProcessing/ "
        f"(expected from the wallCf function object, formats "
        f"{'/'.join(SURFACE_EXTS)})")


# =============================================================================
#  Surface-dump readers (raw primary, VTK via guarded pyvista fallback)
# =============================================================================
def read_raw_surface(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a raw-format surface dump -> (points Nx3, tau Nx3).

    ESI raw surface-writer layout (the documented FO format this reader is
    unit-tested against): '#' header/comment lines, then one whitespace-
    separated row per face centre: x y z tau_x tau_y tau_z. Extra trailing
    columns (a second sampled field) would change the width, so exactly the
    leading 6 are consumed and anything narrower is rejected loudly.
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 6:
        raise ValueError(
            f"{path}: expected >= 6 columns (x y z tau_x tau_y tau_z), "
            f"got {data.shape[1]}")
    return data[:, :3], data[:, 3:6]


def read_vtk_surface(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a VTK surface dump -> (points Nx3, tau Nx3) via pyvista.

    GUARDED import: pyvista is in requirements.txt but this reader is a
    fallback path -- the raw format needs numpy only -- so its absence must
    fail with instructions, not at module import time. Cell data wins over
    point data (the template writes face-centre values, interpolate false);
    cell centres then stand in as the sample coordinates.
    """
    try:
        import pyvista as pv
    except ImportError as exc:                      # pragma: no cover
        raise RuntimeError(
            f"{path}: VTK surface dump found but pyvista is not importable; "
            f"pip install pyvista (see requirements.txt) or switch the "
            f"wallCf function object back to surfaceFormat raw") from exc

    mesh = pv.read(str(path))

    def _pick(arrays) -> Optional[str]:
        # The writer names the array after the field; tolerate decorated
        # names like 'wallShearStress_airfoilSurface' case-insensitively.
        for name in arrays.keys():
            if name.lower().startswith("wallshearstress"):
                return name
        return None

    name = _pick(mesh.cell_data)
    if name is not None:
        tau = np.asarray(mesh.cell_data[name], dtype=float)
        pts = np.asarray(mesh.cell_centers().points, dtype=float)
    else:
        name = _pick(mesh.point_data)
        if name is None:
            raise ValueError(f"{path}: no wallShearStress array in cell or "
                             f"point data")
        tau = np.asarray(mesh.point_data[name], dtype=float)
        pts = np.asarray(mesh.points, dtype=float)
    if tau.ndim != 2 or tau.shape[1] != 3:
        raise ValueError(f"{path}: wallShearStress array is not Nx3 "
                         f"(shape {tau.shape})")
    return pts, tau


def read_surface(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Format dispatch on extension: .raw -> numpy, .vtk/.vtp -> pyvista."""
    if path.suffix.lower() == ".raw":
        return read_raw_surface(path)
    return read_vtk_surface(path)


# =============================================================================
#  Suction-side selection (base-face drop + binned-midline surface split)
# =============================================================================
def suction_side(points: np.ndarray, tau: np.ndarray, alpha_rad: float,
                 n_bins: int = N_SPLIT_BINS
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Filter one patch dump down to the suction surface.

    Returns (x_c, tau_rows) for the suction side only, x_c normalized to
    [0, 1] by the LE/TE x extent of the SURFACE faces (identity at the M1
    unit-chord convention; kept general so a rescaled case still reads
    correctly). Steps, each commented at the code:

      1. drop blunt-TE base faces (vertical x = chord plane);
      2. split upper/lower against a binned midline of the face centres
         (per-bin midpoint of the y extremes ~ the local camber line);
      3. keep the upper side for alpha >= 0, the lower side otherwise
         (matching the BL sample-line convention).
    """
    pts = np.asarray(points, dtype=float)
    tw = np.asarray(tau, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] != tw.shape[0]:
        raise ValueError("points/tau arrays do not describe one surface")
    x, y = pts[:, 0], pts[:, 1]

    # ---- 1. base-face drop ----------------------------------------------------
    #  Base face centres sit exactly on the max-x plane of the patch; surface
    #  face centres stop a TE half-spacing short of it (O(1e-3) chord), so an
    #  absolute 1e-6 band removes the base and nothing else.
    surf = x < (float(np.max(x)) - BASE_X_TOL)
    if int(surf.sum()) < 8:
        raise ValueError("too few surface faces after base-face removal")
    x, y, tw = x[surf], y[surf], tw[surf]

    # ---- chord normalization ----------------------------------------------------
    x_le, x_te = float(np.min(x)), float(np.max(x))
    if x_te <= x_le:
        raise ValueError("degenerate chordwise extent in surface dump")
    x_c = (x - x_le) / (x_te - x_le)

    # ---- 2. upper/lower split against the binned midline ----------------------
    #  Per x-bin midpoint between the highest and lowest face centre tracks
    #  the camber line closely enough to classify every face: at any station
    #  the two surfaces are separated by the full local thickness while the
    #  midline sits centrally between them. Bins seeing fewer than 2 faces
    #  cannot define a midpoint and borrow it by interpolation from the
    #  populated bins instead of misclassifying their lone occupant.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(x_c, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mid = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if int(sel.sum()) >= 2:
            mid[b] = 0.5 * (float(np.min(y[sel])) + float(np.max(y[sel])))
    good = np.isfinite(mid)
    if not good.any():
        raise ValueError("midline split failed (no populated x bins)")
    mid_full = np.interp(centers, centers[good], mid[good])
    upper = y >= mid_full[idx]

    # ---- 3. suction side per AoA sign ------------------------------------------
    #  alpha >= 0 -> upper surface is the suction side; negative AoA (the
    #  sweep dips to -4 deg) -> lower. Same rule as sampling_set_entries in
    #  the case builder, so the Cf curve and the BL profiles describe the
    #  same surface.
    keep = upper if alpha_rad >= 0.0 else ~upper
    if int(keep.sum()) < 8:
        raise ValueError("too few suction-side faces after the surface split")
    order = np.argsort(x_c[keep])
    return x_c[keep][order], tw[keep][order]


# =============================================================================
#  Cf computation + CSV output
# =============================================================================
def compute_cf(tau: np.ndarray, u_inf: float) -> np.ndarray:
    """Cf = |tau_w/rho| / (0.5 U_inf^2): KINEMATIC stress over kinematic q.

    See the module docstring's rho-convention note: the wallShearStress
    function object of an incompressible solver writes tau/rho [m^2/s^2],
    so dividing by 0.5*U^2 (= q/rho) yields the dimensionless Cf with rho
    cancelled exactly -- introducing a density here would be a unit bug.
    """
    if u_inf <= 0.0:
        raise ValueError("U_inf must be positive")
    return np.linalg.norm(np.asarray(tau, dtype=float), axis=1) \
        / (0.5 * u_inf ** 2)


def write_cf_csv(path: Path, x_c: np.ndarray, cf: np.ndarray,
                 source: Path, u_inf: float, alpha_deg: float) -> None:
    """Write the cf_upper.csv contract file (header 'x_c,cf', '#' comments).

    Provenance comments make every curve auditable back to its dump without
    rerunning anything; the column header is what compare_gate's tolerant
    CSV reader keys on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# suction-surface skin friction, "
                 f"Cf = |tau_w/rho| / (0.5*U_inf^2)  [kinematic tau from "
                 f"the incompressible wallShearStress FO]\n"
                 f"# source: {source.as_posix()}\n"
                 f"# U_inf = {u_inf:.6g} m/s, alpha = {alpha_deg:+.2f} deg "
                 f"(both read from 0/U)\n"
                 f"# written by scripts/extract_cf.py\n"
                 f"x_c,cf\n")
        for xx, ff in zip(x_c, cf):
            fh.write(f"{xx:.6f},{ff:.8e}\n")


def extract_case(case_dir: Path) -> Path:
    """Full chain for one case: dump -> suction-side Cf -> cf_upper.csv.

    Returns the written CSV path; raises FileNotFoundError/ValueError (or
    RuntimeError from the guarded pyvista path) with a precise reason when
    any stage cannot deliver -- the caller decides whether that is fatal.
    """
    case_dir = Path(case_dir)
    u_inf, alpha = parse_freestream(case_dir)
    dump = find_wall_shear_file(case_dir)
    points, tau = read_surface(dump)
    x_c, tau_side = suction_side(points, tau, alpha)
    cf = compute_cf(tau_side, u_inf)
    out = case_dir / "postProcessing" / CF_CSV_NAME
    write_cf_csv(out, x_c, cf, dump, u_inf, math.degrees(alpha))
    return out


# =============================================================================
#  CLI
# =============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    """Convert wall-shear dumps to cf_upper.csv for one or more cases."""
    parser = argparse.ArgumentParser(
        description="Suction-surface Cf(x/c) extraction from the case "
                    "template's wallShearStress surface dump (transition-"
                    "gate data producer)")
    parser.add_argument("case_dirs", nargs="+", type=Path,
                        help="OpenFOAM case directories (one per AoA)")
    args = parser.parse_args(argv)

    failures = 0
    for case_dir in args.case_dirs:
        try:
            out = extract_case(case_dir)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"ERROR [{case_dir.name}]: {exc}")
            failures += 1
            continue
        print(f"  [cf] {case_dir.name} -> {out}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
