# -*- coding: utf-8 -*-
"""Phase-1 validation gate evaluator (spec Phase-1 item 4, milestone M1).

Compares the OpenFOAM clean-section polar against the XFOIL baseline and the
NASA experimental anchor, renders a PASS/FAIL/SKIPPED verdict per gate, and
append-updates ``validation/report.md`` with the full configuration table and
the numbers behind every verdict. Nothing is ever silently passed: a missing
input downgrades the affected gate to SKIPPED with an explicit reason.

Gates (thresholds straight from the spec):

  (a) cl_alpha_slope   dCl/dalpha from a linear fit over alpha in [-2, +6] deg
                       must agree with the XFOIL fit within 5 % (relative).
  (b) clmax            Clmax within 10 % of the NASA experimental polar, AND
      stall_aoa        stall AoA (alpha at Clmax) within 2 deg of NASA.
                       NASA digitized CSVs land in validation/nasa/digitized/;
                       until they exist these two gates report SKIPPED.
  (c) transition_xtr   upper-surface transition x/c from the RANS solution
                       within 0.10 chord of XFOIL's e^N xtr_top at matching
                       Ncrit, compared at matching AoA points.

Transition detector (gate c, RANS side)
---------------------------------------
``kOmegaSSTLM`` does not output a single "transition point"; the physically
robust pickup is the wall-Cf signature: laminar Cf decays monotonically aft
of the leading edge, bottoms out just before transition onset, then rises
steeply to the turbulent level. We therefore define the RANS transition
location as the x/c where Cf first rises through its post-minimum inflection,
i.e. the FIRST LOCAL MAXIMUM of dCf/d(x/c) downstream of the Cf minimum (the
steepest point of the turbulent recovery). The leading-edge stagnation spike
is excluded by an x/c cutoff, and the rise must be a real recovery (Cf must
exceed ``rise_factor`` times its minimum) so a separating laminar layer at
the trailing edge is not mistaken for transition. The detector lives here
(``detect_transition_from_cf``) so the orchestrator and the test suite share
one implementation.

Input contracts (all CSV, '#' comments allowed, header row required)
--------------------------------------------------------------------
  OpenFOAM polar    alpha,cl,cd,cm           one row per converged AoA case,
                                             written by scripts/run_validation.py
                                             from the parse_forces means.
  XFOIL polar       alpha,cl,cd,cm,xtr_top   validation/xfoil/polar_*.csv from
                                             validation/xfoil_polar.py; Re and
                                             Ncrit are parsed from the filename
                                             (e.g. polar_Re3e6_N9_free.csv).
  NASA digitized    alpha,cl[,cd]            validation/nasa/digitized/*.csv,
                                             hand-digitized from the reports in
                                             validation/nasa/ (may not exist yet).
  RANS transition   alpha,xtr_top            results/validation/transition_*.csv,
                                             built by the orchestrator from the
                                             extract_bl Cf curves through the
                                             detector above.

Column names are matched case-insensitively through an alias table, so minor
header spelling differences between producers do not break the gate.

CLI:  python validation/compare_gate.py [--re 3e6] [--level 0] [--ncrit 9]
          [--of-csv PATH] [--xfoil-dir DIR] [--nasa-dir DIR]
          [--transition-csv PATH] [--report PATH] [--yaml PATH]

Exit status: 1 if any gate FAILED, else 0 (SKIPPED gates do not fail the
process, but they are loudly visible in the report and on the console).
ASCII-only console output; report written UTF-8 with LF endings.
"""

from __future__ import annotations

import argparse
import math
import re as _regex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python validation/compare_gate.py", Python puts validation/ on
#  sys.path rather than the repo root, hiding the geometry package. Inserting
#  this file's grandparent mirrors conftest.py and keeps the script runnable
#  from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geometry.units import load_aircraft  # noqa: E402

# =============================================================================
#  Gate thresholds and physical constants
# =============================================================================
#  Thresholds are the spec Phase-1 item-4 numbers; they live here as named
#  module constants so the report can print the exact criterion next to each
#  verdict and the tests can reference the same values.
SLOPE_TOL_REL = 0.05        # (a) lift-curve slope agreement, relative
CLMAX_TOL_REL = 0.10        # (b) Clmax agreement vs NASA, relative
STALL_TOL_DEG = 2.0         # (b) stall-AoA agreement vs NASA, degrees
XTR_TOL_XC = 0.10           # (c) transition-location agreement, fraction of chord
FIT_ALPHA_LO = -2.0         # linear-range fit window, degrees ...
FIT_ALPHA_HI = 6.0          # ... per the M1 work order
XTR_ALPHA_MAX = 10.0        # transition compared only pre-stall, where the
                            # e^N vs gamma-Re_theta comparison is meaningful
XTR_ALPHA_MATCH = 0.51      # max AoA mismatch when pairing RANS/XFOIL rows
                            # (covers a 1-deg RANS grid vs 0.5-deg XFOIL grid)

#  Dry-air constants for the Mach bookkeeping in the configuration table.
#  These are physical constants, not aircraft parameters, so they may live
#  here; temperature comes from aircraft.yaml (atmosphere block).
GAMMA_AIR = 1.4             # ratio of specific heats
R_AIR = 287.05              # specific gas constant, J/(kg K)

#  2D-case chord convention for this milestone: unit chord, with the
#  freestream speed set from the target Reynolds number (U = Re * nu / c).
#  See the M1 validation section of the README for the Mach discussion.
CHORD_2D_M = 1.0


# =============================================================================
#  Tolerant CSV ingestion
# =============================================================================
#  Producers on both sides of the gate (XFOIL wrapper, the orchestrator, hand
#  digitization) are separate scripts; matching their headers through one
#  alias table here keeps the gate robust to cosmetic spelling differences
#  without ever guessing about the physics columns.
_COLUMN_ALIASES: Dict[str, str] = {
    # angle of attack (degrees in every polar file in this repo)
    "alpha": "alpha", "a": "alpha", "aoa": "alpha", "alfa": "alpha",
    "alpha_deg": "alpha", "alpha(deg)": "alpha",
    # force/moment coefficients
    "cl": "cl", "cd": "cd", "cdp": "cdp", "cm": "cm",
    # XFOIL e^N transition locations (top = upper/suction surface)
    "xtr_top": "xtr_top", "xtrtop": "xtr_top", "xtr_upper": "xtr_top",
    "top_xtr": "xtr_top", "xtr_t": "xtr_top", "xtr": "xtr_top",
    "xtr_bot": "xtr_bot", "xtrbot": "xtr_bot", "xtr_lower": "xtr_bot",
    "bot_xtr": "xtr_bot",
    # wall-Cf curve columns (extract_bl output consumed by the detector)
    "x_c": "x_c", "x/c": "x_c", "xc": "x_c", "x": "x_c",
    "cf": "cf", "cfx": "cf",
}


def _canon_column(name: str) -> str:
    """Map one raw header token to its canonical column name.

    Unknown headers pass through lower-cased rather than erroring: extra
    columns (Re, iteration counts, ...) are harmless and must not break the
    reader.
    """
    key = name.strip().lower().replace(" ", "")
    return _COLUMN_ALIASES.get(key, key)


def read_csv_columns(path: Path) -> Dict[str, np.ndarray]:
    """Read a small numeric CSV into {canonical_column: float array}.

    Format rules (the cross-producer contract): UTF-8 text, '#' lines are
    comments, the first non-comment line is the header, fields separated by
    commas (whitespace-separated accepted as a fallback for raw XFOIL-style
    dumps). Non-numeric cells become NaN so a stray '-' marker cannot crash
    the gate; short rows are NaN-padded, long rows truncated.
    """
    path = Path(path)
    header: Optional[List[str]] = None
    rows: List[List[float]] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            # Skip blanks and comment lines anywhere in the file.
            if not text or text.startswith("#"):
                continue
            # Choose the delimiter per line: comma when present, else any
            # whitespace run. Mixed files therefore still parse.
            tokens = [t.strip() for t in (text.split(",") if "," in text
                                          else text.split())]
            if header is None:
                header = [_canon_column(t) for t in tokens]
                continue
            # Numeric row: coerce each cell, NaN on failure, pad/truncate to
            # the header width so ragged rows cannot misalign columns.
            vals: List[float] = []
            for tok in tokens[: len(header)]:
                try:
                    vals.append(float(tok))
                except ValueError:
                    vals.append(math.nan)
            while len(vals) < len(header):
                vals.append(math.nan)
            rows.append(vals)

    if header is None:
        raise ValueError(f"{path}: no header row found (empty file?)")

    # Column-major float arrays keyed by canonical name. Duplicate headers
    # keep the FIRST occurrence (later duplicates would be aliases anyway).
    table = np.array(rows, dtype=float) if rows else np.zeros((0, len(header)))
    out: Dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        if name not in out:
            out[name] = table[:, j]
    return out


# =============================================================================
#  Polar-file discovery (filename metadata matching)
# =============================================================================
def _re_from_name(name: str) -> Optional[float]:
    """Parse a Reynolds-number token out of a polar filename.

    Accepted spellings (case-insensitive): Re3e6 / Re1.5e6 / Re_3.0e+06,
    Re3000000 (plain integer, >= 5 digits), Re3M / Re1.5M. Returns None when
    no token is recognized so callers can apply their single-file fallback.
    """
    m = _regex.search(r"[rR]e[_\-]?([0-9]*\.?[0-9]+)[eE]([+\-]?[0-9]+)", name)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    m = _regex.search(r"[rR]e[_\-]?([0-9]{5,})", name)
    if m:
        return float(m.group(1))
    m = _regex.search(r"[rR]e[_\-]?([0-9]*\.?[0-9]+)[mM](?![a-zA-Z])", name)
    if m:
        return float(m.group(1)) * 1.0e6
    return None


def _ncrit_from_name(name: str) -> Optional[int]:
    """Parse an Ncrit token (N9 / Ncrit9 / _n9_) out of a polar filename."""
    m = _regex.search(r"(?:^|[_\-])[nN](?:crit)?[_\-]?([0-9]+)", name)
    return int(m.group(1)) if m else None


def find_xfoil_polar(xfoil_dir: Path, re_value: float, ncrit: int = 9,
                     free_transition: bool = True
                     ) -> Tuple[Optional[Path], str]:
    """Pick the XFOIL polar CSV matching (Re, Ncrit, free/tripped).

    Returns (path, note); path is None when nothing usable exists and the
    note then carries the human-readable reason for the SKIPPED verdict.
    Selection order: exact Re match (1 % tolerance) filtered by transition
    type, with an Ncrit-match preference; if no file carries a parsable Re
    token but exactly one candidate exists, that file is used with a loud
    assumption note rather than silently.
    """
    xfoil_dir = Path(xfoil_dir)
    if not xfoil_dir.is_dir():
        return None, f"XFOIL polar directory not present: {xfoil_dir}"
    candidates = sorted(xfoil_dir.glob("polar_*.csv"))
    if not candidates:
        return None, f"no polar_*.csv files in {xfoil_dir}"

    # Transition-type filter first: tripped polars carry 'trip' in the name
    # by the xfoil_polar.py contract; free-transition files do not.
    typed = [p for p in candidates
             if (("trip" in p.name.lower()) != free_transition)]
    pool = typed if typed else candidates

    # Re-matched subset (1 % relative tolerance covers 3e6 vs 3.0e+06 noise).
    matched = [p for p in pool
               if (_re_from_name(p.name) is not None
                   and abs(_re_from_name(p.name) - re_value)
                   <= 0.01 * re_value)]
    if matched:
        # Prefer the requested Ncrit when several Re-matched files exist.
        ncrit_hits = [p for p in matched if _ncrit_from_name(p.name) == ncrit]
        chosen = (ncrit_hits or matched)[0]
        return chosen, f"matched {chosen.name}"

    # No Re token matched: a single untagged candidate is accepted with an
    # explicit assumption note; multiple untagged files are ambiguous and
    # therefore refused (never guess between polars).
    untagged = [p for p in pool if _re_from_name(p.name) is None]
    if len(untagged) == 1:
        return untagged[0], (f"WARNING: no Re token in {untagged[0].name}; "
                             f"assumed it matches Re={re_value:.3g}")
    return None, (f"no XFOIL polar matching Re={re_value:.3g} "
                  f"in {xfoil_dir} ({len(candidates)} file(s) present)")


def find_nasa_polar(nasa_dir: Path, re_value: float
                    ) -> Tuple[Optional[Path], str]:
    """Pick the digitized NASA experimental polar for this Re.

    The digitized files may not exist yet (digitization is follow-on manual
    work from the PDFs in validation/nasa/); absence is reported as a note so
    the Clmax/stall gates can be marked SKIPPED, never silently passed.
    """
    nasa_dir = Path(nasa_dir)
    if not nasa_dir.is_dir():
        return None, (f"NASA digitized data not present ({nasa_dir} missing) "
                      f"- digitize from validation/nasa/ PDFs")
    candidates = sorted(nasa_dir.glob("*.csv"))
    if not candidates:
        return None, f"no digitized CSVs in {nasa_dir}"

    # Same Re-token logic as the XFOIL picker; experimental data is sparse,
    # so a single untagged file is accepted with an assumption note.
    matched = [p for p in candidates
               if (_re_from_name(p.name) is not None
                   and abs(_re_from_name(p.name) - re_value)
                   <= 0.01 * re_value)]
    if matched:
        return matched[0], f"matched {matched[0].name}"
    if len(candidates) == 1:
        return candidates[0], (f"WARNING: no Re token in {candidates[0].name};"
                               f" assumed it matches Re={re_value:.3g}")
    return None, (f"no NASA CSV matching Re={re_value:.3g} in {nasa_dir} "
                  f"({len(candidates)} file(s) present)")


# =============================================================================
#  Numerics: slope fit, Clmax pickup, transition detector
# =============================================================================
def fit_cl_alpha_slope(alpha_deg: np.ndarray, cl: np.ndarray,
                       lo: float = FIT_ALPHA_LO, hi: float = FIT_ALPHA_HI
                       ) -> Optional[Tuple[float, int]]:
    """Least-squares dCl/dalpha (per degree) over the linear-range window.

    Returns (slope, n_points) or None when fewer than 3 finite samples fall
    inside [lo, hi] - a fit through 2 points is exact but meaningless, so the
    caller downgrades the gate to SKIPPED instead.
    """
    a = np.asarray(alpha_deg, dtype=float)
    c = np.asarray(cl, dtype=float)
    mask = np.isfinite(a) & np.isfinite(c) & (a >= lo - 1e-9) & (a <= hi + 1e-9)
    if int(mask.sum()) < 3:
        return None
    slope = float(np.polyfit(a[mask], c[mask], 1)[0])
    return slope, int(mask.sum())


def clmax_and_stall(alpha_deg: np.ndarray, cl: np.ndarray
                    ) -> Optional[Tuple[float, float]]:
    """(Clmax, stall AoA) from a polar; None when no finite samples exist.

    Stall AoA is defined as the alpha at the Cl maximum of the swept range -
    the same definition is applied to both the RANS and NASA polars so the
    comparison is self-consistent even on a coarse AoA grid.
    """
    a = np.asarray(alpha_deg, dtype=float)
    c = np.asarray(cl, dtype=float)
    mask = np.isfinite(a) & np.isfinite(c)
    if not mask.any():
        return None
    a, c = a[mask], c[mask]
    i = int(np.argmax(c))
    return float(c[i]), float(a[i])


def detect_transition_from_cf(x_c, cf, x_min_cut: float = 0.02,
                              rise_factor: float = 1.3) -> Optional[float]:
    """Upper-surface transition x/c from a wall-Cf distribution (gate c).

    Definition (documented in the module docstring): the x/c where Cf first
    rises through its post-minimum inflection - implemented as the first
    local maximum of dCf/d(x/c) downstream of the Cf minimum, i.e. the
    steepest point of the laminar-to-turbulent Cf recovery.

    Guards that make the pickup robust on real solver output:
      * x/c < ``x_min_cut`` is discarded (LE stagnation spike);
      * a 3-point moving average suppresses cell-to-cell sampling noise;
      * the minimum must not sit at the trailing edge (no recovery = either
        fully laminar or separated -> None, caller reports "not detected");
      * the recovery must be real: max post-minimum Cf >= ``rise_factor`` x
        the minimum, otherwise trailing-edge pressure-gradient droop would
        masquerade as transition;
      * derivative bumps below 25 % of the strongest rise are skipped so a
        tiny numerical wiggle just aft of the minimum cannot win.

    Returns the transition x/c, or None when no transition signature exists.
    """
    x = np.asarray(x_c, dtype=float)
    f = np.asarray(cf, dtype=float)
    mask = np.isfinite(x) & np.isfinite(f)
    x, f = x[mask], f[mask]
    if x.size == 0:
        return None

    # Sort by arc position and drop the stagnation region; everything below
    # needs monotone x for the gradient to mean what we think it means.
    order = np.argsort(x)
    x, f = x[order], f[order]
    keep = x >= x_min_cut
    x, f = x[keep], f[keep]
    if x.size < 8:
        return None

    # Light box smoothing (window 3); endpoints keep their raw values since
    # zero-padded convolution would fabricate a dip at both ends.
    if x.size >= 9:
        smoothed = np.convolve(f, np.full(3, 1.0 / 3.0), mode="same")
        smoothed[0], smoothed[-1] = f[0], f[-1]
        f = smoothed

    # Locate the laminar Cf minimum. If it sits at (or within a few samples
    # of) the trailing edge there is no post-minimum rise to analyze.
    i_min = int(np.argmin(f))
    if i_min >= x.size - 4:
        return None
    xp, fp = x[i_min:], f[i_min:]

    # The rise must be a genuine turbulent recovery, not TE droop noise.
    if float(np.max(fp)) < rise_factor * float(fp[0]):
        return None

    # First significant local maximum of the rise rate = the inflection of
    # the Cf recovery = the transition pickup.
    d = np.gradient(fp, xp)
    d_max = float(np.max(d))
    if d_max <= 0.0:
        return None
    for j in range(1, d.size - 1):
        if d[j] > 0.25 * d_max and d[j] >= d[j - 1] and d[j] >= d[j + 1]:
            return float(xp[j])
    # Monotone rise all the way to the last sample: take the steepest point.
    return float(xp[int(np.argmax(d))])


# =============================================================================
#  Gate evaluation
# =============================================================================
@dataclass
class GateResult:
    """One row of the gate table: verdict plus the numbers behind it.

    ``measured``/``reference``/``delta`` stay numeric (or None) so the test
    suite can assert on values; the report renderer does the formatting.
    """
    gate: str                       # short identifier, e.g. "cl_alpha_slope"
    status: str                     # "PASS" | "FAIL" | "SKIPPED"
    detail: str = ""                # human-readable context / skip reason
    measured: Optional[float] = None     # RANS-side number
    reference: Optional[float] = None    # XFOIL / NASA-side number
    delta: Optional[float] = None        # comparison metric actually gated
    threshold: str = ""             # criterion text printed in the report


def gate_slope(of_polar: Optional[Dict[str, np.ndarray]],
               xfoil_polar: Optional[Dict[str, np.ndarray]]) -> GateResult:
    """Gate (a): lift-curve slope vs XFOIL within 5 % over [-2, +6] deg."""
    thr = (f"|dslope|/slope_xfoil <= {SLOPE_TOL_REL:.0%}, fit alpha in "
           f"[{FIT_ALPHA_LO:g}, {FIT_ALPHA_HI:g}] deg")
    # Missing inputs downgrade to SKIPPED with the exact reason - the report
    # must show WHY a gate did not run, never an empty row.
    if of_polar is None:
        return GateResult("cl_alpha_slope", "SKIPPED",
                          "no OpenFOAM polar available yet", threshold=thr)
    if xfoil_polar is None:
        return GateResult("cl_alpha_slope", "SKIPPED",
                          "no matching XFOIL polar", threshold=thr)

    fit_of = fit_cl_alpha_slope(of_polar.get("alpha", np.array([])),
                                of_polar.get("cl", np.array([])))
    fit_xf = fit_cl_alpha_slope(xfoil_polar.get("alpha", np.array([])),
                                xfoil_polar.get("cl", np.array([])))
    if fit_of is None or fit_xf is None:
        side = "OpenFOAM" if fit_of is None else "XFOIL"
        return GateResult("cl_alpha_slope", "SKIPPED",
                          f"fewer than 3 {side} points in the fit window",
                          threshold=thr)

    s_of, n_of = fit_of
    s_xf, n_xf = fit_xf
    # Relative slope disagreement; a degenerate zero XFOIL slope would mean
    # a broken polar, flagged as FAIL rather than a divide-by-zero crash.
    if s_xf == 0.0:
        return GateResult("cl_alpha_slope", "FAIL",
                          "XFOIL slope is zero (broken polar?)",
                          measured=s_of, reference=s_xf, threshold=thr)
    rel = abs(s_of - s_xf) / abs(s_xf)
    status = "PASS" if rel <= SLOPE_TOL_REL else "FAIL"
    return GateResult(
        "cl_alpha_slope", status,
        f"RANS {s_of:.4f}/deg ({n_of} pts) vs XFOIL {s_xf:.4f}/deg "
        f"({n_xf} pts): {rel:.1%} apart",
        measured=s_of, reference=s_xf, delta=rel, threshold=thr)


def gate_clmax_stall(of_polar: Optional[Dict[str, np.ndarray]],
                     nasa_polar: Optional[Dict[str, np.ndarray]],
                     nasa_note: str = "") -> List[GateResult]:
    """Gate (b): Clmax within 10 % and stall AoA within 2 deg of NASA data.

    Returns two rows (clmax, stall_aoa) so each criterion carries its own
    verdict and numbers in the report.
    """
    thr_cl = f"|dClmax|/Clmax_NASA <= {CLMAX_TOL_REL:.0%}"
    thr_aa = f"|dalpha_stall| <= {STALL_TOL_DEG:g} deg"
    # NASA digitized data is the known-possibly-absent input (spec allows it
    # to lag); both rows go SKIPPED carrying the discovery note verbatim.
    if nasa_polar is None:
        reason = nasa_note or "NASA digitized polar not available"
        return [GateResult("clmax", "SKIPPED", reason, threshold=thr_cl),
                GateResult("stall_aoa", "SKIPPED", reason, threshold=thr_aa)]
    if of_polar is None:
        reason = "no OpenFOAM polar available yet"
        return [GateResult("clmax", "SKIPPED", reason, threshold=thr_cl),
                GateResult("stall_aoa", "SKIPPED", reason, threshold=thr_aa)]

    peak_of = clmax_and_stall(of_polar.get("alpha", np.array([])),
                              of_polar.get("cl", np.array([])))
    peak_na = clmax_and_stall(nasa_polar.get("alpha", np.array([])),
                              nasa_polar.get("cl", np.array([])))
    if peak_of is None or peak_na is None:
        side = "OpenFOAM" if peak_of is None else "NASA"
        reason = f"{side} polar has no finite (alpha, cl) rows"
        return [GateResult("clmax", "SKIPPED", reason, threshold=thr_cl),
                GateResult("stall_aoa", "SKIPPED", reason, threshold=thr_aa)]

    (cl_of, aa_of), (cl_na, aa_na) = peak_of, peak_na
    # Clmax relative error against the experimental value.
    rel = abs(cl_of - cl_na) / abs(cl_na)
    row_cl = GateResult(
        "clmax", "PASS" if rel <= CLMAX_TOL_REL else "FAIL",
        f"RANS Clmax {cl_of:.3f} vs NASA {cl_na:.3f}: {rel:.1%} apart",
        measured=cl_of, reference=cl_na, delta=rel, threshold=thr_cl)
    # Stall AoA absolute error in degrees.
    daa = abs(aa_of - aa_na)
    row_aa = GateResult(
        "stall_aoa", "PASS" if daa <= STALL_TOL_DEG else "FAIL",
        f"RANS stall {aa_of:.1f} deg vs NASA {aa_na:.1f} deg: "
        f"{daa:.1f} deg apart",
        measured=aa_of, reference=aa_na, delta=daa, threshold=thr_aa)
    return [row_cl, row_aa]


def gate_transition(rans_transition: Optional[Dict[str, np.ndarray]],
                    xfoil_polar: Optional[Dict[str, np.ndarray]]
                    ) -> GateResult:
    """Gate (c): RANS upper-surface xtr within 0.10c of XFOIL e^N xtr_top.

    RANS rows (alpha, xtr_top from the Cf detector) are paired with the
    nearest XFOIL alpha within ``XTR_ALPHA_MATCH`` deg; only pre-stall rows
    (alpha <= XTR_ALPHA_MAX) are gated since the e^N comparison loses meaning
    once the suction-side flow separates. The WORST matched pair drives the
    verdict so one bad station cannot hide behind a good average.
    """
    thr = (f"max |xtr_RANS - xtr_XFOIL| <= {XTR_TOL_XC:.2f} c over matched "
           f"alpha <= {XTR_ALPHA_MAX:g} deg")
    if rans_transition is None:
        return GateResult("transition_xtr", "SKIPPED",
                          "no RANS transition table (case wall-Cf sampling "
                          "not available yet)", threshold=thr)
    if xfoil_polar is None or "xtr_top" not in xfoil_polar:
        return GateResult("transition_xtr", "SKIPPED",
                          "XFOIL polar missing (or lacks xtr_top column)",
                          threshold=thr)

    a_r = np.asarray(rans_transition.get("alpha", np.array([])), dtype=float)
    x_r = np.asarray(rans_transition.get("xtr_top", np.array([])), dtype=float)
    a_x = np.asarray(xfoil_polar["alpha"], dtype=float)
    x_x = np.asarray(xfoil_polar["xtr_top"], dtype=float)
    xf_ok = np.isfinite(a_x) & np.isfinite(x_x)
    a_x, x_x = a_x[xf_ok], x_x[xf_ok]
    if a_x.size == 0:
        return GateResult("transition_xtr", "SKIPPED",
                          "XFOIL polar has no finite xtr_top rows",
                          threshold=thr)

    # Pair each pre-stall RANS row with its nearest XFOIL alpha; track the
    # worst chordwise disagreement across all pairs.
    worst: Optional[Tuple[float, float, float, float]] = None  # (dx,a,xr,xx)
    n_pairs = 0
    for ar, xr in zip(a_r, x_r):
        if not (math.isfinite(ar) and math.isfinite(xr)):
            continue
        if ar > XTR_ALPHA_MAX:
            continue
        j = int(np.argmin(np.abs(a_x - ar)))
        if abs(a_x[j] - ar) > XTR_ALPHA_MATCH:
            continue
        n_pairs += 1
        dx = abs(xr - x_x[j])
        if worst is None or dx > worst[0]:
            worst = (dx, ar, xr, float(x_x[j]))

    if worst is None:
        return GateResult("transition_xtr", "SKIPPED",
                          "no RANS/XFOIL alpha pairs matched within "
                          f"{XTR_ALPHA_MATCH:g} deg", threshold=thr)
    dx, aw, xr_w, xx_w = worst
    status = "PASS" if dx <= XTR_TOL_XC else "FAIL"
    return GateResult(
        "transition_xtr", status,
        f"{n_pairs} alpha pair(s); worst at alpha {aw:.1f} deg: RANS "
        f"xtr {xr_w:.3f} vs XFOIL {xx_w:.3f} (d = {dx:.3f} c)",
        measured=xr_w, reference=xx_w, delta=dx, threshold=thr)


def evaluate_gates(of_polar: Optional[Dict[str, np.ndarray]],
                   xfoil_polar: Optional[Dict[str, np.ndarray]],
                   nasa_polar: Optional[Dict[str, np.ndarray]],
                   rans_transition: Optional[Dict[str, np.ndarray]],
                   nasa_note: str = "") -> List[GateResult]:
    """Run every Phase-1 gate on already-loaded tables (None = absent)."""
    results = [gate_slope(of_polar, xfoil_polar)]
    results.extend(gate_clmax_stall(of_polar, nasa_polar, nasa_note))
    results.append(gate_transition(rans_transition, xfoil_polar))
    return results


# =============================================================================
#  Report rendering (validation/report.md, append-update semantics)
# =============================================================================
def _fmt_num(v: Optional[float]) -> str:
    """Compact numeric cell for the markdown tables ('-' when absent)."""
    return "-" if v is None else f"{v:.4g}"


def _md_cell(text: str) -> str:
    """Escape literal pipes so threshold/detail text (which uses |x| for
    absolute values) cannot split a markdown table row into extra cells."""
    return text.replace("|", "\\|")


def render_block(title: str, config_rows: List[Tuple[str, str]],
                 results: List[GateResult], notes: List[str],
                 generated_utc: str) -> str:
    """Render one configuration's report block as markdown text."""
    lines: List[str] = [f"## {title}", "",
                        f"*Generated: {generated_utc} (UTC) by "
                        f"validation/compare_gate.py*", ""]
    # Configuration table: every number the gates depend on, so a report
    # reader can reproduce the case without opening any script.
    lines += ["| parameter | value |", "| --- | --- |"]
    lines += [f"| {_md_cell(str(k))} | {_md_cell(str(v))} |"
              for k, v in config_rows]
    lines.append("")
    # Gate table: verdicts with the gated numbers alongside. All free-text
    # cells go through the pipe escape - thresholds spell |x| literally.
    lines += ["| gate | status | measured | reference | delta | "
              "threshold | detail |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(
            f"| {r.gate} | **{r.status}** | {_fmt_num(r.measured)} | "
            f"{_fmt_num(r.reference)} | {_fmt_num(r.delta)} | "
            f"{_md_cell(r.threshold)} | {_md_cell(r.detail)} |")
    lines.append("")
    if notes:
        lines.append("Notes:")
        lines += [f"- {n}" for n in notes]
        lines.append("")
    return "\n".join(lines)


def upsert_report(report_path: Path, key: str, block_text: str) -> None:
    """Insert or replace one keyed block inside validation/report.md.

    Blocks are delimited by HTML comments so re-running the gate for the
    same (Re, level, Ncrit) configuration UPDATES its section in place while
    other configurations' sections are preserved - the report accumulates
    one section per configuration, never duplicates.
    """
    report_path = Path(report_path)
    begin = f"<!-- M1-GATE {key} BEGIN -->"
    end = f"<!-- M1-GATE {key} END -->"
    wrapped = f"{begin}\n{block_text.rstrip()}\n{end}\n"

    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        i, j = text.find(begin), text.find(end)
        if i != -1 and j != -1 and j > i:
            # Replace the existing block for this key in place.
            text = text[:i] + wrapped + text[j + len(end):].lstrip("\n")
        else:
            # New configuration: append below whatever exists already.
            text = text.rstrip("\n") + "\n\n" + wrapped
    else:
        # Fresh report: standing preamble + the first block.
        preamble = (
            "# Phase-1 clean-section validation report\n\n"
            "Auto-generated by `validation/compare_gate.py` - do not edit "
            "blocks between the M1-GATE markers by hand; they are rewritten "
            "on every gate run. Spec reference: glasair3-vg-cfd-spec.md, "
            "Phase 1 item 4.\n\n")
        text = preamble + wrapped

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# =============================================================================
#  CLI
# =============================================================================
def re_tag(re_value: float) -> str:
    """Filename tag for a Reynolds number: 3e6 -> '3e6', 1.5e6 -> '1.5e6'."""
    return f"{re_value / 1.0e6:g}e6"


def main(argv: Optional[List[str]] = None) -> int:
    """Evaluate the Phase-1 gates and append-update validation/report.md."""
    parser = argparse.ArgumentParser(
        description="Phase-1 validation gate evaluator (M1)")
    parser.add_argument("--re", type=float, default=3.0e6,
                        help="chord Reynolds number of the configuration "
                             "under test (default 3e6)")
    parser.add_argument("--level", type=int, default=0,
                        help="mesh refinement level label (default 0)")
    parser.add_argument("--ncrit", type=int, default=9,
                        help="XFOIL e^N Ncrit to match (default 9)")
    parser.add_argument("--of-csv", type=Path, default=None,
                        help="OpenFOAM polar CSV (alpha,cl,cd,cm); default "
                             "results/validation/openfoam_polar_Re<tag>_L<lvl>.csv")
    parser.add_argument("--xfoil-dir", type=Path,
                        default=REPO / "validation" / "xfoil",
                        help="directory holding XFOIL polar_*.csv files")
    parser.add_argument("--nasa-dir", type=Path,
                        default=REPO / "validation" / "nasa" / "digitized",
                        help="directory holding digitized NASA CSVs")
    parser.add_argument("--transition-csv", type=Path, default=None,
                        help="RANS transition CSV (alpha,xtr_top); default "
                             "results/validation/transition_Re<tag>_L<lvl>.csv")
    parser.add_argument("--report", type=Path,
                        default=REPO / "validation" / "report.md",
                        help="gate report to append-update")
    parser.add_argument("--yaml", type=Path, default=REPO / "aircraft.yaml",
                        help="master parameter file (solver block, atmosphere)")
    args = parser.parse_args(argv)

    # Default artifact paths derive from the (Re, level) labels so multiple
    # configurations coexist under results/validation/ without collisions.
    tag = re_tag(args.re)
    if args.of_csv is None:
        args.of_csv = (REPO / "results" / "validation"
                       / f"openfoam_polar_Re{tag}_L{args.level}.csv")
    if args.transition_csv is None:
        args.transition_csv = (REPO / "results" / "validation"
                               / f"transition_Re{tag}_L{args.level}.csv")

    # ---- configuration context from aircraft.yaml ---------------------------
    #  The atmosphere block supplies nu and T; the solver block supplies the
    #  pinned OpenFOAM version (null until the WSL install lands -> UNPINNED
    #  warning, per the reproducibility rule in spec section 4).
    ac = load_aircraft(args.yaml)
    nu = ac.atmosphere.kinematic_viscosity
    temp = ac.atmosphere.temperature
    of_version = ac.solver.openfoam_version
    version_str = of_version if of_version else "UNPINNED (null in aircraft.yaml)"
    if not of_version:
        print("WARNING: solver.openfoam_version is null in aircraft.yaml - "
              "pin the exact installed release at M1 solver bring-up "
              "(spec section 4 reproducibility rule).")

    # 2D-case convention: unit chord, U set by the requested Re. The speed
    # of sound from the YAML temperature gives the Mach bookkeeping for the
    # quasi-incompressible check (M < 0.2 guideline).
    u_inf = args.re * nu / CHORD_2D_M
    a_sound = math.sqrt(GAMMA_AIR * R_AIR * temp)
    mach = u_inf / a_sound

    # ---- gather inputs (absence is a verdict, not an error) -----------------
    of_polar = read_csv_columns(args.of_csv) if Path(args.of_csv).is_file() else None
    xf_path, xf_note = find_xfoil_polar(args.xfoil_dir, args.re, args.ncrit)
    xfoil_polar = read_csv_columns(xf_path) if xf_path else None
    na_path, na_note = find_nasa_polar(args.nasa_dir, args.re)
    nasa_polar = read_csv_columns(na_path) if na_path else None
    rans_tr = (read_csv_columns(args.transition_csv)
               if Path(args.transition_csv).is_file() else None)

    results = evaluate_gates(of_polar, xfoil_polar, nasa_polar, rans_tr,
                             nasa_note=na_note)

    # ---- assemble the report block -------------------------------------------
    config_rows: List[Tuple[str, str]] = [
        ("aircraft", ac.meta.aircraft),
        ("airfoil", ac.wing.airfoil_file),
        ("turbulence model", ac.solver.turbulence_model),
        ("OpenFOAM version", version_str),
        ("Re (chord)", f"{args.re:.3g}"),
        ("chord convention", f"c = {CHORD_2D_M:g} m (2D unit chord)"),
        ("nu (aircraft.yaml)", f"{nu:.4e} m2/s"),
        ("U_inf = Re*nu/c", f"{u_inf:.2f} m/s"),
        ("Mach (a = {:.1f} m/s)".format(a_sound), f"{mach:.3f}"),
        ("XFOIL Ncrit", str(args.ncrit)),
        ("mesh level", str(args.level)),
        ("OpenFOAM polar", str(args.of_csv) if of_polar is not None
         else f"{args.of_csv} (absent)"),
        ("XFOIL polar", xf_path.name if xf_path else f"none ({xf_note})"),
        ("NASA polar", na_path.name if na_path else f"none ({na_note})"),
        ("RANS transition table", str(args.transition_csv)
         if rans_tr is not None else f"{args.transition_csv} (absent)"),
    ]

    notes: List[str] = []
    if mach > 0.2:
        # The Re-6M point exceeds the M < 0.2 quasi-incompressible guideline
        # at unit chord. Decision (documented here AND in the README M1
        # section): KEEP unit chord and accept M ~ 0.26 as a trend point -
        # simpleFoam and XFOIL are both run incompressible, so the RANS/XFOIL
        # comparison stays self-consistent, matching common practice for
        # incompressible airfoil validation studies; absolute values at this
        # Re carry the usual compressibility caveat.
        notes.append(
            f"Mach {mach:.3f} exceeds the 0.2 quasi-incompressible guideline "
            f"at unit chord. Kept deliberately: both simpleFoam and XFOIL "
            f"run incompressible, so the comparison is self-consistent; "
            f"treat this Re as a TREND point (see README, 'M1 validation').")
    if xf_note.startswith("WARNING"):
        notes.append(xf_note)
    if na_path and na_note.startswith("WARNING"):
        notes.append(na_note)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    title = f"Configuration: Re {tag}, mesh level {args.level}, Ncrit {args.ncrit}"
    key = f"re={tag} level={args.level} ncrit={args.ncrit}"
    upsert_report(args.report, key, render_block(title, config_rows, results,
                                                 notes, stamp))

    # ---- console summary (ASCII only) ----------------------------------------
    print("=" * 78)
    print(f"Phase-1 gate summary - Re {tag}, level {args.level}, "
          f"Ncrit {args.ncrit}")
    print("=" * 78)
    for r in results:
        print(f"  {r.gate:<16} {r.status:<8} {r.detail}")
    print("-" * 78)
    print(f"  report: {args.report}")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIPPED")
    print(f"  gates: {len(results)} total, "
          f"{sum(1 for r in results if r.status == 'PASS')} pass, "
          f"{n_fail} fail, {n_skip} skipped")
    # FAIL is the only nonzero condition: SKIPPED rows are loudly visible in
    # the report but must not block the dry-run workflow while inputs lag.
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
