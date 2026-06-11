# -*- coding: utf-8 -*-
"""OpenFOAM postProcessing reader and convergence gate for the M1 pipeline.

Reads the two time series every Phase-1 case writes under postProcessing/:

  * force coefficients  -- forceCoeffs function object; ESI releases name the
    file ``coefficient.dat`` (v2012+) while older/foundation builds write
    ``forceCoeffs.dat``. Both layouts are accepted; columns are located by
    NAME from the last '# ...' header line, never by position, so extra
    columns (Cs, CmRoll, front/rear splits like ``Cl(f)``) are harmless.
  * residuals           -- ESI ``solverInfo`` (per-field ``<U>_initial`` /
    ``<U>_final`` / ``<U>_iters`` columns, plus non-numeric solver-name
    columns that this parser tolerates) or the foundation-style ``residuals``
    function object (plain ``# Time p Ux Uy ...`` numeric table).

  * wall y+           -- the yPlus function object (system/yPlusObj in the
    case template) writes the y+ volField into each write-time directory
    and echoes per-patch min/max/average to the solver log and to
    postProcessing/<yPlus FO>/<time>/yPlus.dat. The y+ gate below consumes
    the per-face field when present and falls back to the scalar stats.

On top of those it implements BOTH spec section-4 case gates VERBATIM:

    residuals < 1e-5  AND  force coefficients flat
    (< 0.5% peak-to-peak oscillation over the last 500 iterations)

    y+ < 1 on all viscous walls is the design target; print a y+ histogram
    per case and FAIL the case if y+ > 2 over more than 1% of wall faces

Interpretation choices, documented once here and enforced below:

  * "residuals < 1e-5" is judged on the INITIAL residual of every monitored
    equation at the final recorded iteration -- the standard steady-state
    reading (the initial residual is the true measure of how far the field
    is from satisfying its equation; the final residual only reflects the
    linear-solver tolerance). The per-window maximum is also reported for
    transparency but does not gate, so a residual still descending through
    the averaging window is not punished retroactively.
  * "flat over the last 500 iterations" assumes the function object writes
    one sample per iteration (writeInterval 1, the case-template setting);
    the window is counted in SAMPLES. A history shorter than the window
    cannot certify flatness and FAILS with an explicit note -- the spec
    demands failing cases be auto-flagged, never silently included.
  * peak-to-peak is normalized by |mean| over the window. A coefficient
    sitting near zero (Cm crossing zero at some AoA) would blow that ratio
    up on numerical dust, so the normalizer is floored at MEAN_FLOOR; below
    the floor the test effectively becomes an absolute pk-pk bound of
    flat_limit * MEAN_FLOOR.

Restart handling: a restarted run writes a NEW time directory under the
function-object folder (postProcessing/forceCoeffs/0/, .../1500/, ...) whose
iteration range overlaps the tail of the previous segment. Segments are
concatenated in start-time order and duplicate iteration values keep the
LAST occurrence: the post-restart rows were recomputed from the restart
fields and belong to the trajectory the final solution actually followed,
while the pre-restart tail belongs to an abandoned (often diverging) segment
that the restart deliberately discarded.

Library API (imported by the M1 orchestrator):

    load_force_coeffs(case_dir) -> ForceHistory     (numpy arrays)
    load_residuals(case_dir)    -> ResidualHistory  (numpy arrays per field)
    convergence_verdict(forces, residuals, ...) -> ConvergenceVerdict
    yplus_verdict(case_dir, patch="airfoil", ...) -> YPlusVerdict

Driver contract (scripts/run_validation.py imports these two entry points;
exact signatures are part of the cross-script contract and must not drift):

    gate_case(case_dir) -> ConvergenceVerdict
        residual + force-flatness gate; .passed / .cl_mean / .cd_mean /
        .cm_mean consumed by the polar aggregation.
    gate_case_yplus(case_dir) -> YPlusVerdict
        spec section-4 wall-y+ gate at default thresholds (y+ > 2 on more
        than 1% of airfoil wall faces = FAIL). Never raises: missing or
        stats-only evidence degrades to an INDETERMINATE verdict whose
        .passed is False, so the driver can flag without try/except.
        .report() renders the per-case ASCII histogram the spec requires.

CLI:  python scripts/parse_forces.py CASE_DIR [--window 500]
          [--residual-limit 1e-5] [--flat-limit 0.005] [--yplus]
Exit code 0 = gate PASS, 1 = gate FAIL, 2 = data missing/unreadable.
With --yplus the wall-y+ gate (histogram report included) runs after the
convergence gate and both must PASS for exit code 0.
Plain-ASCII output only (safe on the default Windows console).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python scripts/parse_forces.py", Python puts scripts/ on
#  sys.path rather than the repo root, which would hide sibling packages.
#  Inserting this file's grandparent directory mirrors conftest.py and keeps
#  the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# =============================================================================
#  Gate constants -- straight from spec section 4 ("Engineering Rules").
#  These are SPEC thresholds, not aircraft parameters, so they live here
#  rather than in aircraft.yaml; the CLI can still override per invocation.
# =============================================================================
RESIDUAL_LIMIT_DEFAULT = 1.0e-5   # all equations' initial residuals below this
FLAT_LIMIT_DEFAULT = 0.005        # < 0.5% pk-pk oscillation on each coefficient
WINDOW_DEFAULT = 500              # averaging/flatness window, in iterations
MEAN_FLOOR = 1.0e-3               # normalizer floor for near-zero coefficients

# File names the two history loaders search for under postProcessing/.
# Order matters only for documentation; discovery merges all hits.
FORCE_FILE_NAMES = ("coefficient.dat", "forceCoeffs.dat")
RESIDUAL_FILE_NAMES = ("solverInfo.dat", "residuals.dat")

# --- wall-y+ gate (spec section 4: "y+ < 1 on all viscous walls; print a
# y+ histogram per case and fail the case if max y+ > 2 over more than 1%
# of wall faces"). Thresholds are SPEC values, same policy as above.
YPLUS_LIMIT_DEFAULT = 2.0          # per-face y+ threshold
YPLUS_MAX_FRACTION_DEFAULT = 0.01  # tolerated exceedance fraction (1% faces)
YPLUS_PATCH_DEFAULT = "airfoil"    # viscous-wall patch name (case template)
# Fixed histogram bins. The interesting physics lives below 2 (design target
# y+ < 1, hard fail threshold 2); everything past 10 is one overflow bucket.
YPLUS_BIN_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf"))
# Solver logs scanned for the yPlus FO echo, steady first then the
# documented pseudo-transient fallback (template README).
YPLUS_LOG_NAMES = ("log.simpleFoam", "log.pimpleFoam")


# =============================================================================
#  History containers -- plain numpy arrays so the orchestrator can slice,
#  plot, or window them without re-parsing anything.
# =============================================================================
@dataclass
class ForceHistory:
    """Concatenated, deduplicated force-coefficient time series for one case."""

    time: np.ndarray                    # iteration/pseudo-time, strictly increasing
    cl: np.ndarray                      # lift coefficient
    cd: np.ndarray                      # drag coefficient
    cm: np.ndarray                      # pitching-moment coefficient (CmPitch)
    sources: List[Path] = field(default_factory=list)   # files, segment order
    n_duplicates_dropped: int = 0       # restart-overlap rows replaced

    def __len__(self) -> int:
        return int(self.time.size)


@dataclass
class ResidualHistory:
    """Concatenated, deduplicated residual time series for one case.

    ``fields`` maps equation name -> initial-residual array aligned with
    ``time``. For solverInfo input the ``_initial`` suffix is stripped so the
    keys read 'Ux', 'p', 'omega', ... regardless of which function object
    produced the file.
    """

    time: np.ndarray
    fields: Dict[str, np.ndarray]
    sources: List[Path] = field(default_factory=list)
    n_duplicates_dropped: int = 0

    def __len__(self) -> int:
        return int(self.time.size)


# =============================================================================
#  Low-level table parsing -- shared by both loaders
# =============================================================================
def _read_of_table(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Parse one OpenFOAM postProcessing table into named numpy columns.

    Robustness rules, each earned by a real OpenFOAM quirk:
      * Column names come from the LAST '#' comment line seen BEFORE the
        first data row ('# Time ...' and '#Time ...' both occur in the
        wild); earlier banner lines ('# Force coefficients', dragDir echo
        lines, blank '#') are skipped.
      * Rows whose token count differs from the header are dropped -- a run
        killed mid-write leaves a truncated final line and that must not
        poison the whole history.
      * Columns that fail float conversion (solverInfo's '<U>_solver' name
        strings, 'converged' true/false flags) are silently omitted; only
        numeric columns are returned.
    """
    header: Optional[List[str]] = None
    rows: List[List[str]] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Candidate header: tokens after the '#'. Only update while
                # no data has been read -- a stray comment mid-file must not
                # rename columns under the data that already used them.
                tokens = stripped.lstrip("#").split()
                if tokens and not rows:
                    header = tokens
                continue
            # Data row: parentheses are stripped defensively so a vector
            # written as '(fx fy fz)' (older forces.dat style) still splits
            # into clean numeric tokens.
            rows.append(stripped.replace("(", " ").replace(")", " ").split())

    if header is None:
        raise ValueError(f"{path}: no '# ...' header line found")
    if not rows:
        raise ValueError(f"{path}: header present but no data rows")

    # Keep only rows matching the header width (truncated-line guard).
    width = len(header)
    rows = [r for r in rows if len(r) == width]
    if not rows:
        raise ValueError(f"{path}: no data row matches the {width}-column header")

    # Column-wise float conversion; non-numeric columns are dropped wholesale
    # so string-valued solverInfo columns never raise.
    columns: Dict[str, np.ndarray] = {}
    names: List[str] = []
    for j, name in enumerate(header):
        try:
            col = np.array([float(r[j]) for r in rows], dtype=float)
        except ValueError:
            continue
        # Duplicate header names (rare, malformed files): first wins, the
        # repeat is ignored rather than silently overwriting.
        if name not in columns:
            columns[name] = col
            names.append(name)
    if not columns:
        raise ValueError(f"{path}: no numeric columns could be parsed")
    return names, columns


def _find_time_column(names: Sequence[str]) -> str:
    """Locate the time/iteration column by case-insensitive name match."""
    for name in names:
        if name.lower() == "time":
            return name
    raise ValueError(f"no 'Time' column among parsed columns: {list(names)}")


def _discover_segments(case_dir: Path, file_names: Sequence[str],
                       fo_hint: str) -> List[Path]:
    """Find one function object's history files, sorted by segment start time.

    Layout searched:  <case>/postProcessing/<funcObject>/<startTime>/<file>.
    When several function objects match (e.g. a user added a second forces
    block), the one whose directory name matches ``fo_hint`` (substring,
    case-insensitive) is preferred; ties resolve alphabetically so behavior
    stays deterministic. Segments are ordered by float(startTime) -- the
    concatenation order the keep-LAST duplicate rule depends on.
    """
    post = case_dir / "postProcessing"
    if not post.is_dir():
        raise FileNotFoundError(f"{case_dir}: no postProcessing/ directory")

    # Gather every matching file, grouped by its function-object directory
    # (two levels up: <fo>/<time>/<file>).
    by_fo: Dict[Path, List[Path]] = {}
    for fname in file_names:
        for hit in post.rglob(fname):
            by_fo.setdefault(hit.parent.parent, []).append(hit)
    if not by_fo:
        raise FileNotFoundError(
            f"{case_dir}: none of {list(file_names)} found under postProcessing/"
        )

    # Prefer the function object whose name carries the hint; deterministic
    # alphabetical fallback otherwise.
    fo_dirs = sorted(by_fo, key=lambda p: (fo_hint not in p.name.lower(), p.name))
    files = by_fo[fo_dirs[0]]

    # Segment order = numeric start-time of the enclosing time directory.
    # Non-numeric directory names (should not happen) sort last by name so
    # nothing is ever silently dropped.
    def seg_key(p: Path) -> Tuple[float, str]:
        try:
            return (float(p.parent.name), p.parent.name)
        except ValueError:
            return (float("inf"), p.parent.name)

    return sorted(files, key=seg_key)


def _dedupe_keep_last(time: np.ndarray,
                      arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """Sort by time and keep the LAST occurrence of each duplicated value.

    A stable argsort preserves concatenation order among equal times, so
    within each run of equal time values the final element is the one from
    the latest restart segment -- exactly the row that supersedes the
    abandoned pre-restart tail (see module docstring for the rationale).
    """
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    keep = np.ones(t_sorted.size, dtype=bool)
    # Element i survives only if the next sorted time differs -> last of run.
    keep[:-1] = t_sorted[1:] != t_sorted[:-1]
    dropped = int(t_sorted.size - keep.sum())
    kept_idx = order[keep]
    return t_sorted[keep], [a[kept_idx] for a in arrays], dropped


# =============================================================================
#  Force-coefficient loader
# =============================================================================
def _match_column(columns: Dict[str, np.ndarray],
                  candidates: Sequence[str]) -> Optional[str]:
    """Return the first column whose name equals a candidate (case-insens).

    Exact-name matching is deliberate: 'Cl' must NOT match 'Cl(f)' or
    'Cl(r)' (the front/rear body splits ESI appends), so substring matching
    would grab the wrong physics.
    """
    lower = {name.lower(): name for name in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_force_coeffs(case_dir: Union[str, Path]) -> ForceHistory:
    """Load Cl/Cd/Cm history for one case as deduplicated numpy arrays.

    Accepts both the ESI ``coefficient.dat`` and the legacy
    ``forceCoeffs.dat`` layouts; concatenates restart segments in start-time
    order and keeps the LAST row at duplicated times. Cm maps to 'CmPitch'
    when present (ESI 6-component output) falling back to plain 'Cm'
    (legacy 3-column output).
    """
    case_dir = Path(case_dir)
    files = _discover_segments(case_dir, FORCE_FILE_NAMES, fo_hint="coeff")

    t_parts: List[np.ndarray] = []
    cl_parts: List[np.ndarray] = []
    cd_parts: List[np.ndarray] = []
    cm_parts: List[np.ndarray] = []

    for f in files:
        names, cols = _read_of_table(f)
        t_name = _find_time_column(names)
        # Name-based mapping; every miss is a hard error naming the file so
        # a renamed function object fails loudly, not with garbage physics.
        cl_name = _match_column(cols, ("Cl",))
        cd_name = _match_column(cols, ("Cd",))
        cm_name = _match_column(cols, ("CmPitch", "Cm"))
        missing = [lbl for lbl, n in
                   (("Cl", cl_name), ("Cd", cd_name), ("Cm/CmPitch", cm_name))
                   if n is None]
        if missing:
            raise ValueError(f"{f}: missing coefficient column(s): {missing}; "
                             f"parsed columns = {names}")
        t_parts.append(cols[t_name])
        cl_parts.append(cols[cl_name])
        cd_parts.append(cols[cd_name])
        cm_parts.append(cols[cm_name])

    time = np.concatenate(t_parts)
    arrays = [np.concatenate(cl_parts), np.concatenate(cd_parts),
              np.concatenate(cm_parts)]
    time, (cl, cd, cm), dropped = _dedupe_keep_last(time, arrays)
    return ForceHistory(time=time, cl=cl, cd=cd, cm=cm,
                        sources=files, n_duplicates_dropped=dropped)


# =============================================================================
#  Residual loader
# =============================================================================
def load_residuals(case_dir: Union[str, Path]) -> ResidualHistory:
    """Load per-equation initial-residual history for one case.

    solverInfo files carry '<field>_initial' / '_final' / '_iters' triplets
    plus string columns; only the '_initial' columns gate convergence (the
    initial residual measures how far the field is from its equation; the
    final residual just echoes the linear-solver tolerance). Foundation
    'residuals' files carry plain per-field columns, all of which are taken
    as initial residuals. Restart segments concatenate keep-LAST exactly
    like the force history.
    """
    case_dir = Path(case_dir)
    files = _discover_segments(case_dir, RESIDUAL_FILE_NAMES, fo_hint="solver")

    t_parts: List[np.ndarray] = []
    field_parts: Dict[str, List[np.ndarray]] = {}

    for f in files:
        names, cols = _read_of_table(f)
        t_name = _find_time_column(names)

        # solverInfo flavor: any '_initial' column present -> use only those,
        # stripping the suffix for clean field names. Otherwise (plain
        # residuals.dat) every numeric non-Time column is a field.
        initial_names = [n for n in names if n.endswith("_initial")]
        if initial_names:
            picked = {n[: -len("_initial")]: cols[n] for n in initial_names}
        else:
            picked = {n: cols[n] for n in names if n != t_name}
        if not picked:
            raise ValueError(f"{f}: no residual columns found (columns={names})")

        # Field sets must agree across segments -- a restart with a changed
        # equation set would silently misalign arrays otherwise.
        if field_parts and set(picked) != set(field_parts):
            raise ValueError(
                f"{f}: residual fields {sorted(picked)} differ from earlier "
                f"segments {sorted(field_parts)}"
            )
        t_parts.append(cols[t_name])
        for name, arr in picked.items():
            field_parts.setdefault(name, []).append(arr)

    time = np.concatenate(t_parts)
    names_sorted = sorted(field_parts)
    arrays = [np.concatenate(field_parts[n]) for n in names_sorted]
    time, deduped, dropped = _dedupe_keep_last(time, arrays)
    return ResidualHistory(
        time=time,
        fields={n: a for n, a in zip(names_sorted, deduped)},
        sources=files,
        n_duplicates_dropped=dropped,
    )


# =============================================================================
#  Convergence gate
# =============================================================================
@dataclass
class ConvergenceVerdict:
    """Machine- and human-readable outcome of the section-4 convergence gate."""

    passed: bool                         # overall verdict (both sub-gates)
    residual_pass: bool                  # all final initial-residuals < limit
    flatness_pass: bool                  # all coefficients flat over window
    window: int                          # requested window length (iterations)
    window_used: int                     # samples actually available/used
    # --- means over the averaging window (the reportable polar point) ---
    cl_mean: float
    cd_mean: float
    cm_mean: float
    # --- residual evidence ---
    final_residuals: Dict[str, float]    # field -> value at last iteration
    window_max_residuals: Dict[str, float]  # field -> max over window (info)
    worst_residual: float                # max of final_residuals
    worst_residual_field: str
    residual_limit: float
    # --- flatness evidence: coefficient -> relative pk-pk over window ---
    oscillation: Dict[str, float]
    flat_limit: float
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        """Render the verdict as a fixed-width ASCII block for logs/console."""
        lines: List[str] = []
        verdict = "PASS" if self.passed else "FAIL"
        lines.append("=" * 72)
        lines.append(f"convergence gate verdict: {verdict}")
        lines.append("=" * 72)
        # Window means first -- these are the numbers the polar consumes.
        lines.append(f"averaging window        : last {self.window_used} of "
                     f"{self.window} requested iterations")
        lines.append(f"mean Cl                 : {self.cl_mean:+.6f}")
        lines.append(f"mean Cd                 : {self.cd_mean:+.6f}")
        lines.append(f"mean Cm                 : {self.cm_mean:+.6f}")
        # Residual sub-gate with the per-field evidence table.
        res = "PASS" if self.residual_pass else "FAIL"
        lines.append(f"residual gate (<{self.residual_limit:.0e})   : {res}  "
                     f"(worst {self.worst_residual:.3e} on "
                     f"'{self.worst_residual_field}')")
        for name in sorted(self.final_residuals):
            lines.append(f"  {name:<12} final {self.final_residuals[name]:.3e}"
                         f"   window-max {self.window_max_residuals[name]:.3e}")
        # Flatness sub-gate with per-coefficient relative pk-pk.
        flat = "PASS" if self.flatness_pass else "FAIL"
        lines.append(f"flatness gate (<{self.flat_limit * 100:.2f}% pk-pk): {flat}")
        for name in ("Cl", "Cd", "Cm"):
            if name in self.oscillation:
                lines.append(f"  {name:<12} pk-pk/|mean| = "
                             f"{self.oscillation[name] * 100:.4f} %")
        for note in self.notes:
            lines.append(f"note: {note}")
        lines.append("=" * 72)
        return "\n".join(lines)


def convergence_verdict(forces: ForceHistory,
                        residuals: Optional[ResidualHistory],
                        window: int = WINDOW_DEFAULT,
                        residual_limit: float = RESIDUAL_LIMIT_DEFAULT,
                        flat_limit: float = FLAT_LIMIT_DEFAULT,
                        mean_floor: float = MEAN_FLOOR) -> ConvergenceVerdict:
    """Apply the spec section-4 gate to one case's histories.

    PASS requires BOTH sub-gates:
      1. residuals: every monitored equation's initial residual at the final
         recorded iteration is below ``residual_limit`` (1e-5 per spec). A
         missing residual history cannot certify and therefore FAILS.
      2. flatness: over the last ``window`` force samples (500 iterations
         per spec, one sample per iteration assumed), each of Cl/Cd/Cm has
         peak-to-peak / max(|mean|, mean_floor) below ``flat_limit`` (0.5%).
         A history shorter than the window cannot certify and FAILS.

    Means are computed over the same window so the reported polar point and
    the flatness evidence describe identical samples.
    """
    notes: List[str] = []

    # ---- averaging / flatness window ----------------------------------------
    n = len(forces)
    window_used = min(window, n)
    short_history = n < window
    if short_history:
        notes.append(f"force history has only {n} samples; cannot certify "
                     f"flatness over the required {window}-iteration window")
    sl = slice(n - window_used, n)

    # Window means -- always computed (even on FAIL the numbers are wanted
    # for the auto-flag report).
    cl_mean = float(np.mean(forces.cl[sl]))
    cd_mean = float(np.mean(forces.cd[sl]))
    cm_mean = float(np.mean(forces.cm[sl]))

    # Per-coefficient relative peak-to-peak with the near-zero floor.
    oscillation: Dict[str, float] = {}
    for name, arr in (("Cl", forces.cl), ("Cd", forces.cd), ("Cm", forces.cm)):
        seg = arr[sl]
        pkpk = float(np.max(seg) - np.min(seg))
        denom = max(abs(float(np.mean(seg))), mean_floor)
        oscillation[name] = pkpk / denom
    flatness_pass = (not short_history) and all(
        v < flat_limit for v in oscillation.values()
    )

    # ---- residual sub-gate ----------------------------------------------------
    final_res: Dict[str, float] = {}
    window_max_res: Dict[str, float] = {}
    if residuals is None or len(residuals) == 0:
        notes.append("no residual history available; residual gate cannot "
                     "be certified and is marked FAIL")
        residual_pass = False
        worst, worst_field = float("nan"), "(none)"
    else:
        # Window over the residual history sized like the force window so the
        # informational window-max describes the averaging interval.
        m = len(residuals)
        rsl = slice(max(0, m - window_used), m)
        for fname, arr in residuals.fields.items():
            final_res[fname] = float(arr[-1])
            window_max_res[fname] = float(np.max(arr[rsl]))
        worst_field = max(final_res, key=lambda k: final_res[k])
        worst = final_res[worst_field]
        residual_pass = worst < residual_limit

    if forces.n_duplicates_dropped:
        notes.append(f"restart overlap: {forces.n_duplicates_dropped} "
                     f"duplicated force samples superseded (kept last)")

    return ConvergenceVerdict(
        passed=residual_pass and flatness_pass,
        residual_pass=residual_pass,
        flatness_pass=flatness_pass,
        window=window,
        window_used=window_used,
        cl_mean=cl_mean, cd_mean=cd_mean, cm_mean=cm_mean,
        final_residuals=final_res,
        window_max_residuals=window_max_res,
        worst_residual=worst,
        worst_residual_field=worst_field,
        residual_limit=residual_limit,
        oscillation=oscillation,
        flat_limit=flat_limit,
        notes=notes,
    )


def gate_case(case_dir: Union[str, Path],
              window: int = WINDOW_DEFAULT,
              residual_limit: float = RESIDUAL_LIMIT_DEFAULT,
              flat_limit: float = FLAT_LIMIT_DEFAULT) -> ConvergenceVerdict:
    """Convenience one-shot: load both histories and gate a case directory.

    Residuals are loaded best-effort: a case that never wrote a solverInfo/
    residuals file still gets a verdict (FAIL on the residual sub-gate with
    an explanatory note) instead of an exception, because the orchestrator
    needs to flag such cases, not crash on them.
    """
    forces = load_force_coeffs(case_dir)
    try:
        residuals: Optional[ResidualHistory] = load_residuals(case_dir)
    except (FileNotFoundError, ValueError):
        residuals = None
    return convergence_verdict(forces, residuals, window=window,
                               residual_limit=residual_limit,
                               flat_limit=flat_limit)


# =============================================================================
#  Wall-y+ gate -- spec section 4
# =============================================================================
#  Evidence sources, strongest first:
#    1. the yPlus volField written into the latest time directory (per-face
#       values on the wall patch -> true histogram + exact exceedance
#       fraction);
#    2. the scalar min/max/average stats: postProcessing yPlus.dat rows or
#       the 'patch <name> y+ : min = .. max = .. average = ..' echo in the
#       solver log. Stats alone cannot count faces, so the documented
#       conservative rule applies: PASS only when max <= limit, otherwise
#       INDETERMINATE (treated as FAIL) with instructions to write the field.
# -----------------------------------------------------------------------------

# v2506 yPlus FO log echo, e.g.
#   patch airfoil y+ : min = 0.31, max = 1.42, average = 0.87
# The separator alternation ('=' vs ':') also swallows the foundation-build
# variant so a log produced by either lineage parses identically.
_YPLUS_LOG_RE = re.compile(
    r"patch\s+(\S+)\s+y\+\s*:\s*"
    r"min\s*[=:]\s*([-+0-9.eE]+)\s*,?\s*"
    r"max\s*[=:]\s*([-+0-9.eE]+)\s*,?\s*"
    r"average\s*[=:]\s*([-+0-9.eE]+)"
)


def _strip_foam_comments(text: str) -> str:
    """Remove C/C++ comments from an OpenFOAM dictionary/field file.

    The FoamFile banner is a '/* ... */' block and every file ends with a
    '// ***' footer; both must vanish before brace matching so a brace-like
    character inside ASCII art can never derail the parser.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def _brace_block(text: str, open_idx: int) -> Optional[str]:
    """Return the contents of the brace block opening at ``open_idx``.

    Plain depth counter -- OpenFOAM field files contain no string literals
    with braces, so no quote tracking is needed. Returns None on an
    unbalanced block (truncated write) so callers degrade instead of crash.
    """
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


def _parse_yplus_field_patch(path: Path, patch: str
                             ) -> Tuple[Optional[np.ndarray], List[str]]:
    """Extract the wall-patch y+ values from a written yPlus field file.

    Handles the ASCII volScalarField boundaryField entry in both forms:
      value uniform <s>;                          -> one representative value
      value nonuniform List<scalar> N ( ... );    -> per-face list
    Returns (values, notes); values is None when the patch cannot be read
    (binary format, missing patch, truncated file), with the reason noted.
    """
    notes: List[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, [f"yPlus field unreadable: {exc}"]

    # Binary-format fields cannot be text-parsed; tell the user how to fix
    # rather than mis-reading raw bytes as zero faces.
    if re.search(r"\bformat\s+binary\s*;", text):
        return None, [f"{path.name} is binary format; set writeFormat ascii "
                      f"(or reconstruct with -fields) to enable the per-face "
                      f"y+ histogram"]

    text = _strip_foam_comments(text)

    # boundaryField { ... } first, then the named patch's sub-dictionary.
    m = re.search(r"\bboundaryField\b", text)
    if not m:
        return None, [f"{path.name}: no boundaryField block"]
    open_idx = text.find("{", m.end())
    if open_idx < 0:
        return None, [f"{path.name}: malformed boundaryField block"]
    bf = _brace_block(text, open_idx)
    if bf is None:
        return None, [f"{path.name}: unbalanced boundaryField block "
                      f"(truncated write?)"]

    pm = re.search(r"(?:^|\s)" + re.escape(patch) + r"\s*\{", bf)
    if not pm:
        return None, [f"{path.name}: patch '{patch}' not in boundaryField"]
    pblock = _brace_block(bf, bf.find("{", pm.start()))
    if pblock is None:
        return None, [f"{path.name}: unbalanced '{patch}' patch entry"]

    # nonuniform per-face list: the payload between the parentheses is pure
    # scalars (no nested parens possible for List<scalar>), newlines included.
    nm = re.search(r"\bvalue\s+nonuniform\s+List<scalar>\s*\d*\s*"
                   r"\(([^()]*)\)\s*;", pblock, flags=re.DOTALL)
    if nm:
        tokens = nm.group(1).split()
        if not tokens:
            return None, [f"{path.name}: patch '{patch}' has an empty y+ "
                          f"value list"]
        try:
            return np.array([float(t) for t in tokens], dtype=float), notes
        except ValueError:
            return None, [f"{path.name}: non-numeric token in '{patch}' "
                          f"y+ value list"]

    # uniform fallback: a single value standing for every face. The face
    # count is unrecoverable, but the exceedance fraction is degenerate
    # (0 or 1) so the gate still decides exactly.
    um = re.search(r"\bvalue\s+uniform\s+([-+0-9.eE]+)\s*;", pblock)
    if um:
        notes.append(f"patch '{patch}' wrote a uniform y+ value; face count "
                     f"unavailable, single representative value histogrammed")
        return np.array([float(um.group(1))], dtype=float), notes

    return None, [f"{path.name}: patch '{patch}' carries no parsable "
                  f"'value' entry"]


def _find_latest_yplus_field(case_dir: Path) -> Optional[Path]:
    """Locate the yPlus field file in the LATEST numeric time directory.

    Write-time fields land in <case>/<time>/yPlus (reconstructPar
    -latestTime reassembles the parallel pieces there). Only directories
    that actually contain the field compete, so a 0/ directory or a write
    time predating the FO's addition never shadows real data.
    """
    best: Optional[Tuple[float, Path]] = None
    try:
        children = list(case_dir.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        try:
            t = float(child.name)
        except ValueError:
            continue
        f = child / "yPlus"
        if f.is_file() and (best is None or t > best[0]):
            best = (t, f)
    return best[1] if best else None


def _yplus_stats_from_dat(path: Path, patch: str
                          ) -> Optional[Tuple[float, float, float]]:
    """Read (min, max, average) for one patch from a yPlus.dat table.

    v2506 writeFile layout: '# Time patch min max average' with one row per
    (write time, patch). The patch column is a string, so the generic table
    reader cannot row-filter -- parsed manually here. The LAST matching row
    wins: latest write time describes the final solution.
    """
    last: Optional[Tuple[float, float, float]] = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        # Row shape: time, patch name, min, max, average.
        if len(tok) >= 5 and tok[1] == patch:
            try:
                last = (float(tok[2]), float(tok[3]), float(tok[4]))
            except ValueError:
                continue
    return last


def _yplus_stats_from_log(path: Path, patch: str
                          ) -> Optional[Tuple[float, float, float]]:
    """Scan one solver log for the yPlus FO echo; LAST occurrence wins."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last: Optional[Tuple[float, float, float]] = None
    for m in _YPLUS_LOG_RE.finditer(text):
        if m.group(1) == patch:
            last = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return last


@dataclass
class YPlusVerdict:
    """Outcome of the spec section-4 wall-y+ gate for one case.

    ``status`` is three-valued: PASS / FAIL on solid evidence, INDETERMINATE
    when only scalar stats exist and max > limit (face fraction unknowable)
    or when no y+ evidence exists at all. INDETERMINATE always carries
    ``passed = False`` -- the spec auto-flags, it never benefit-of-doubts.
    """

    passed: bool                       # overall verdict (False covers FAIL
                                       # and INDETERMINATE alike)
    status: str                        # "PASS" | "FAIL" | "INDETERMINATE"
    source: str                        # "field" (per-face) | "stats" | "none"
    patch: str                         # wall patch evaluated
    n_faces: int                       # per-face sample count (0 for stats)
    y_min: float                       # min y+ (nan when unknown)
    y_max: float                       # max y+ (nan when unknown)
    y_mean: float                      # mean/average y+ (nan when unknown)
    fraction_above: float              # faces with y+ > limit / total
                                       # (nan when per-face data missing)
    limit: float                       # per-face threshold (spec: 2.0)
    max_fraction: float                # tolerated fraction (spec: 0.01)
    bin_counts: List[int] = field(default_factory=list)   # per YPLUS_BIN_EDGES
    bin_edges: Tuple[float, ...] = YPLUS_BIN_EDGES
    evidence: Optional[Path] = None    # file the verdict was read from
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        """Render the verdict + spec-required histogram as fixed-width ASCII."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append(f"wall y+ gate verdict: {self.status}  "
                     f"(patch '{self.patch}', evidence: {self.source})")
        lines.append("=" * 72)
        if np.isfinite(self.y_max):
            lines.append(f"y+ min / mean / max     : {self.y_min:.4f} / "
                         f"{self.y_mean:.4f} / {self.y_max:.4f}")
        if self.evidence is not None:
            lines.append(f"evidence file           : {self.evidence}")
        if self.bin_counts:
            # Exceedance line first -- the number the gate actually trips on.
            lines.append(f"faces with y+ > {self.limit:g}      : "
                         f"{int(round(self.fraction_above * self.n_faces))} "
                         f"of {self.n_faces} "
                         f"({self.fraction_above * 100:.3f}%; "
                         f"limit {self.max_fraction * 100:.3f}%)")
            lines.append(f"y+ histogram ({self.n_faces} faces):")
            peak = max(self.bin_counts) or 1     # bar scale; avoid div-by-0
            for k, count in enumerate(self.bin_counts):
                lo, hi = self.bin_edges[k], self.bin_edges[k + 1]
                hi_txt = "inf " if hi == float("inf") else f"{hi:4.1f}"
                bar = "#" * int(round(40.0 * count / peak))
                pct = 100.0 * count / max(self.n_faces, 1)
                lines.append(f"  [{lo:4.1f}, {hi_txt})  {count:8d} "
                             f"|{bar:<40s}| {pct:6.2f}%")
        for note in self.notes:
            lines.append(f"note: {note}")
        lines.append("=" * 72)
        return "\n".join(lines)


def yplus_verdict(case_dir: Union[str, Path],
                  patch: str = YPLUS_PATCH_DEFAULT,
                  limit: float = YPLUS_LIMIT_DEFAULT,
                  max_fraction: float = YPLUS_MAX_FRACTION_DEFAULT
                  ) -> YPlusVerdict:
    """Evaluate the spec section-4 wall-y+ gate for one case directory.

    Evidence search order (strongest wins, see section banner above):
      1. <case>/<latestTime>/yPlus field, '<patch>' boundaryField values --
         exact histogram, FAIL iff fraction(y+ > limit) > max_fraction;
      2. postProcessing/**/yPlus.dat row or solver-log echo for '<patch>' --
         min/max/average only, so the conservative rule applies: PASS only
         when max <= limit, otherwise INDETERMINATE (passed = False) with a
         note telling the user to enable/recover the field write;
      3. nothing found -> INDETERMINATE (passed = False).

    Total function: never raises on missing/garbled data -- the orchestrator
    must auto-flag broken cases, not crash on them (spec section 4).
    """
    case_dir = Path(case_dir)
    notes: List[str] = []

    # ---- 1. per-face field at the latest write time --------------------------
    field_file = _find_latest_yplus_field(case_dir)
    if field_file is not None:
        values, parse_notes = _parse_yplus_field_patch(field_file, patch)
        notes.extend(parse_notes)
        if values is not None and values.size:
            counts, _ = np.histogram(values, bins=np.array(YPLUS_BIN_EDGES))
            frac = float(np.count_nonzero(values > limit)) / float(values.size)
            failed = frac > max_fraction
            if failed:
                notes.append(f"{frac * 100:.3f}% of '{patch}' faces exceed "
                             f"y+ = {limit:g} (allowed {max_fraction * 100:.3f}%)")
            return YPlusVerdict(
                passed=not failed,
                status="FAIL" if failed else "PASS",
                source="field",
                patch=patch,
                n_faces=int(values.size),
                y_min=float(np.min(values)),
                y_max=float(np.max(values)),
                y_mean=float(np.mean(values)),
                fraction_above=frac,
                limit=limit, max_fraction=max_fraction,
                bin_counts=[int(c) for c in counts],
                evidence=field_file,
                notes=notes,
            )
        # Field file present but unreadable: fall through to the stats path,
        # keeping the parse failure visible in the notes.

    # ---- 2. scalar stats: yPlus.dat table, then the solver-log echo ----------
    stats: Optional[Tuple[float, float, float]] = None
    stats_src: Optional[Path] = None
    post = case_dir / "postProcessing"
    if post.is_dir():
        # Several restart segments may each carry a yPlus.dat; scanning in
        # NUMERIC start-time order and keeping the last hit mirrors the
        # keep-LAST rule the history loaders apply to restarts. Lexical
        # sorting would put segment '500' after '1000' and resurrect stale
        # pre-restart stats, so the parent time directory is compared as a
        # float (non-numeric parents sort first, i.e. weakest evidence).
        def _seg_time(p: Path) -> Tuple[float, str]:
            try:
                return (float(p.parent.name), str(p))
            except ValueError:
                return (float("-inf"), str(p))

        for dat in sorted(post.rglob("yPlus.dat"), key=_seg_time):
            got = _yplus_stats_from_dat(dat, patch)
            if got is not None:
                stats, stats_src = got, dat
    if stats is None:
        for log_name in YPLUS_LOG_NAMES:
            log_path = case_dir / log_name
            if log_path.is_file():
                got = _yplus_stats_from_log(log_path, patch)
                if got is not None:
                    stats, stats_src = got, log_path
                    break

    if stats is not None:
        y_min, y_max, y_avg = stats
        if y_max <= limit:
            # Conservative PASS: if even the worst face is under the limit,
            # the 1%-fraction rule is satisfied by construction.
            notes.append(f"per-face y+ field unavailable; PASS granted on "
                         f"the conservative rule max y+ = {y_max:g} <= "
                         f"{limit:g} (every face below the limit)")
            status, passed = "PASS", True
        else:
            # Stats cannot count exceeding faces -> cannot certify either
            # way. Spec auto-flags: INDETERMINATE is treated as FAIL.
            notes.append(f"max y+ = {y_max:g} exceeds {limit:g} but only "
                         f"min/max/average stats are available, so the "
                         f">{max_fraction * 100:.0f}%-of-faces rule cannot be "
                         f"evaluated; treated as FAIL. Enable the yPlus "
                         f"field write (system/yPlusObj, writeControl "
                         f"writeTime) and reconstruct the latest time so "
                         f"<case>/<time>/yPlus exists, then re-gate")
            status, passed = "INDETERMINATE", False
        return YPlusVerdict(
            passed=passed, status=status, source="stats", patch=patch,
            n_faces=0, y_min=y_min, y_max=y_max, y_mean=y_avg,
            fraction_above=float("nan"),
            limit=limit, max_fraction=max_fraction,
            bin_counts=[], evidence=stats_src, notes=notes,
        )

    # ---- 3. no evidence at all ------------------------------------------------
    notes.append(f"no y+ evidence found for patch '{patch}': no "
                 f"<time>/yPlus field, no postProcessing yPlus.dat, no "
                 f"'patch {patch} y+ :' line in "
                 f"{'/'.join(YPLUS_LOG_NAMES)}; run the case with the "
                 f"yPlus function object enabled (system/yPlusObj)")
    return YPlusVerdict(
        passed=False, status="INDETERMINATE", source="none", patch=patch,
        n_faces=0, y_min=float("nan"), y_max=float("nan"),
        y_mean=float("nan"), fraction_above=float("nan"),
        limit=limit, max_fraction=max_fraction,
        bin_counts=[], evidence=None, notes=notes,
    )


# CONTRACT(run_validation.py): after reconstructPar in solve_case, call
# scripts.parse_forces.gate_case_yplus(case_dir) -> YPlusVerdict, print
# verdict.report() (the spec's per-case histogram) and treat
# verdict.passed == False as a case FAIL alongside the convergence gate.
def gate_case_yplus(case_dir: Union[str, Path]) -> YPlusVerdict:
    """Driver entry point: wall-y+ gate at spec defaults for one case.

    Thin wrapper so the orchestrator depends on one stable name with one
    argument (mirroring gate_case); thresholds stay pinned to the spec
    values unless a caller deliberately reaches for yplus_verdict.
    """
    return yplus_verdict(case_dir)


# =============================================================================
#  CLI
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Gate one case from the command line; exit code mirrors the verdict."""
    p = argparse.ArgumentParser(
        description="OpenFOAM force/residual reader + spec section-4 "
                    "convergence gate (PASS/FAIL)")
    p.add_argument("case_dir", type=Path,
                   help="OpenFOAM case directory containing postProcessing/")
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                   help="flatness/averaging window in iterations (spec: 500)")
    p.add_argument("--residual-limit", type=float, default=RESIDUAL_LIMIT_DEFAULT,
                   help="residual threshold (spec: 1e-5)")
    p.add_argument("--flat-limit", type=float, default=FLAT_LIMIT_DEFAULT,
                   help="relative pk-pk threshold (spec: 0.005 = 0.5%%)")
    p.add_argument("--yplus", action="store_true",
                   help="also run the spec section-4 wall-y+ gate and print "
                        "the per-case y+ histogram; both gates must PASS "
                        "for exit code 0")
    args = p.parse_args(argv)

    # Loading errors are environment problems (missing files), not gate
    # failures -- distinct exit code 2 so the sweep runner can tell them apart.
    try:
        verdict = gate_case(args.case_dir, window=args.window,
                            residual_limit=args.residual_limit,
                            flat_limit=args.flat_limit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"case: {args.case_dir}")
    print(verdict.report())
    passed = verdict.passed

    # Opt-in y+ gate keeps the historical exit-code contract intact for
    # callers that only want convergence (the driver imports the library
    # entry points directly; this flag serves manual triage).
    if args.yplus:
        yp = gate_case_yplus(args.case_dir)
        print(yp.report())
        passed = passed and yp.passed

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
