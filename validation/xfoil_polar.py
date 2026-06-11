"""Batch XFOIL polar driver for the LS(1)-0413 clean-section baseline (M1).

Phase-1 validation anchor (spec section "Phase 1 - Clean-section validation"):
runs XFOIL 6.99 polar sweeps of the wing section at Re = 1.5M / 3M / 6M, with
free transition (e^N, Ncrit = 9) and tripped (forced xtr = 0.05/0.05) cases,
producing one CSV per (Re, transition) pair plus a combined comparison figure.
These polars are the reference the OpenFOAM 2D case is gated against.

Speed / Reynolds convention for ALL 2D work in this milestone:
    chord c = 1.0 m, freestream U = Re * nu / c,
with nu taken from aircraft.yaml (atmosphere.kinematic_viscosity, ISA SL).
At Re = 6e6 that nominal speed is U ~ 87.6 m/s -> Mach ~ 0.26, which exceeds
the quasi-incompressible M < 0.2 guideline. DECISION: keep Re = 6M at c = 1 m
anyway, because the Mach number here is purely nominal bookkeeping:
  * XFOIL is run strictly incompressible (MACH 0), so its polars carry no
    compressibility effect whatsoever;
  * the matching OpenFOAM Phase-1 case uses incompressible simpleFoam, where
    U only sets Re - the nondimensional solution is Reynolds-similar, so
    rescaling the chord at fixed Re would change nothing in the results;
  * Re = 6M is the cruise-end trend point required by the spec and matches
    the NASA Langley test Reynolds numbers for this section, which is the
    common practice for simpleFoam airfoil validation studies.
The same decision is recorded in the auto-generated validation/xfoil/README.md.

XFOIL handling notes (verified empirically against the 6.99 Windows binary):
  * The section is resampled to 241 cosine-clustered points with a sharp
    (closed) trailing edge via geometry.airfoil - XFOIL's BL solver prefers
    a closed TE, and the closure shear leaves the nose untouched.
  * PANE is issued after LOAD so XFOIL re-panels its own spline of those
    coordinates (curvature-based clustering, default 160 nodes). Feeding the
    raw 241-point cosine distribution straight in as panels produced bogus
    premature transition (xtr ~ 0.03c at alpha 0, Re 6M) and BL-march
    failures; with PANE the same geometry gives clean polars and transition
    locations consistent with a laminar-flow section.
  * PACC accumulates only CONVERGED points into the polar dump, so skip
    detection is exact: any requested alpha missing from the dump did not
    converge within ITER iterations and is logged as skipped.

Binary resolution order: --xfoil flag, XFOIL_BIN env var, the repo copy at
tools/xfoil/xfoil.exe (provenance: tools/xfoil/SOURCE.md), then 'wsl xfoil'.

Run:  python validation/xfoil_polar.py [--xfoil PATH] [--airfoil DAT]
          [--re 1.5e6 3e6 6e6] [--ncrit 9] [--free | --tripped]
          [--out validation/xfoil] [--iter 200]
With no transition flag BOTH free and tripped sweeps run (the full 6-polar
Phase-1 matrix). Plain-ASCII console output (Windows-console safe).
"""

from __future__ import annotations

import argparse
import math
import os
import re as regex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python validation/xfoil_polar.py", Python puts validation/ on
#  sys.path rather than the repo root, which would hide the geometry package.
#  Inserting this file's grandparent directory mirrors conftest.py and keeps
#  the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

# Headless backend before pyplot: this driver runs in consoles/CI where no
# display exists, and Agg keeps the PNG output byte-identical either way.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geometry.airfoil import load_airfoil, resample_airfoil, write_selig  # noqa: E402
from geometry.units import load_aircraft  # noqa: E402

# =============================================================================
#  Sweep constants (spec Phase 1; aircraft-independent numerics only -
#  everything aircraft-specific is read from aircraft.yaml at run time)
# =============================================================================

# Reynolds matrix from the spec: approach (1.5M), climb-ish mid point (3M),
# cruise-end trend point (6M). Overridable via --re.
DEFAULT_RE_LIST: Tuple[float, ...] = (1.5e6, 3.0e6, 6.0e6)

# Transition settings: free transition leaves the e^N method in charge
# (trip stations parked at the TE), tripped forces transition at 5% chord on
# both surfaces - the "composite skin with bugs/rain" bracketing case.
XTR_FREE: Tuple[float, float] = (1.0, 1.0)
XTR_TRIP: Tuple[float, float] = (0.05, 0.05)

# Viscous-solver iteration cap per point (spec: ITER 200) and the watchdog
# timeouts. A full 19-point polar measured ~0.5 s on the dev box, so these
# are hang guards, not tuning knobs: XFOIL 6.99 can freeze hard inside
# TRCHEK2 (NaN spin, observed at Re 1.5M tripped past stall), in which case
# the sweep process is killed, the partially-written PACC file is salvaged,
# and the remaining angles are retried in isolated single-point processes.
DEFAULT_ITER = 200
POLAR_TIMEOUT_S = 120.0      # full-sweep process watchdog
POINT_TIMEOUT_S = 60.0       # isolated single-point retry watchdog

# Section paneling handed to XFOIL: 241 cosine points, sharp TE (rationale in
# the module docstring), as required for this milestone.
N_FOIL_POINTS = 241

# Nominal chord for the 2D convention (c = 1.0 m) and ISA dry-air constants
# for the advisory Mach check. gamma/R are physical constants, not aircraft
# parameters; temperature comes from aircraft.yaml at run time.
CHORD_M = 1.0
GAMMA_AIR = 1.4
R_AIR = 287.05  # J/(kg K)

# Quasi-incompressible guideline used by the advisory Mach table; exceeding
# it triggers the documented Re-6M note, never an abort (see docstring).
MACH_GUIDELINE = 0.2

# CSV column order - the cross-file contract with the gate-report tooling.
CSV_COLUMNS: Tuple[str, ...] = ("alpha", "cl", "cd", "cdp", "cm", "xtr_top", "xtr_bot")


# =============================================================================
#  Pure helpers (unit-tested without invoking XFOIL)
# =============================================================================

def build_alpha_schedule(a_min: float = -4.0, coarse_step: float = 2.0,
                         coarse_end: float = 8.0, fine_step: float = 1.0,
                         a_max: float = 20.0) -> List[float]:
    """AoA schedule per spec: coarse steps in the linear range, fine near stall.

    Defaults give -4..8 in 2-deg steps then 9..20 in 1-deg steps (19 points).
    Values are accumulated by repeated addition of exact binary-representable
    steps and rounded to 6 decimals so downstream string formatting and the
    skip-detection tolerance never see float drift.
    """
    alphas: List[float] = []
    a = a_min
    # Coarse leg through the linear range; the epsilon guards the <= test
    # against accumulated float error without ever admitting an extra point.
    while a <= coarse_end + 1e-9:
        alphas.append(round(a, 6))
        a += coarse_step
    # Fine leg toward stall starts one fine step past the coarse end so the
    # junction point is not duplicated.
    a = coarse_end + fine_step
    while a <= a_max + 1e-9:
        alphas.append(round(a, 6))
        a += fine_step
    return alphas


def re_label(re_value: float) -> str:
    """Reynolds tag used in output filenames: 1.5e6 -> '1.5M', 3e6 -> '3M'.

    %g collapses trailing zeros so the names match the spec's file list
    (polar_Re1.5M_N9_free.csv etc.) without special-casing integral millions.
    """
    return f"{re_value / 1.0e6:g}M"


def build_xfoil_script(foil_file: str, polar_file: str, re_value: float,
                       ncrit: float, xtr: Tuple[float, float],
                       alphas: Sequence[float], mach: float = 0.0,
                       n_iter: int = DEFAULT_ITER, repanel: bool = True) -> str:
    """Assemble the piped-stdin command stream for one polar accumulation run.

    The exact grammar below was verified against the XFOIL 6.99 Windows
    binary (inline arguments accepted on N / XTR / VISC / MACH / ITER; blank
    lines exit submenus). Sequence:

      PLOP G        - disable the plot window (mandatory for batch use)
      LOAD + PANE   - read the Selig file, re-panel on XFOIL's spline
                      (see module docstring for why PANE is not optional)
      VPAR N/XTR    - amplification exponent and trip stations
      VISC Re       - enter viscous mode at the target Reynolds number
                      (VISC toggles, so it must appear exactly once)
      MACH/ITER     - incompressible setting and iteration cap
      PACC          - open polar accumulation (dump-file prompt declined)
      ALFA ...      - one line per requested angle, ascending, so each point
                      warm-starts from its converged neighbor
      PACC/QUIT     - close the polar and exit cleanly
    """
    lines = [
        "PLOP",          # plotting-options menu
        "G",             # toggle graphics-enable off (no plot window)
        "",              # leave PLOP
        f"LOAD {foil_file}",
    ]
    # Re-panel unless the caller explicitly wants raw coordinates as panels
    # (kept switchable for paneling-sensitivity checks; default is on).
    if repanel:
        lines.append("PANE")
    lines += [
        "OPER",
        "VPAR",                                  # viscous-parameter submenu
        f"N {ncrit:g}",                          # e^N critical amplification
        f"XTR {xtr[0]:.3f} {xtr[1]:.3f}",        # trip x/c, top then bottom
        "",                                      # leave VPAR
        f"VISC {re_value:.0f}",                  # viscous mode at this Re
        f"MACH {mach:g}",                        # 0 = strictly incompressible
        f"ITER {n_iter}",                        # viscous iteration cap
        "PACC",                                  # open polar accumulation
        polar_file,                              # polar save file
        "",                                      # no polar dump file
    ]
    # One ALFA per requested angle. Ascending order is deliberate: each point
    # initializes from the previous converged BL state, which is what carries
    # the sweep through the high-alpha points near stall.
    lines += [f"ALFA {a:.2f}" for a in alphas]
    lines += [
        "PACC",          # close accumulation (toggle off)
        "",              # leave OPER
        "QUIT",
    ]
    return "\n".join(lines) + "\n"


def parse_polar_dump(text: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Parse an XFOIL PACC polar save file into (metadata, data rows).

    The dump layout (XFOIL 6.99) is a fixed header - title, 'Calculated
    polar for:' name, xtrf line, 'Mach = ... Re = x.xxx e N ... Ncrit = ...'
    line - followed by a column-name row, a dashed separator, and one
    7-column numeric row per CONVERGED point. Rows are harvested only after
    the dashed line and only when all 7 whitespace-separated fields parse as
    floats, so header numerals can never masquerade as data.

    :returns: (meta, rows) where meta holds name/mach/re/ncrit/xtrf_top/
        xtrf_bot (whatever was found) and rows is a list of dicts keyed by
        CSV_COLUMNS, in file order.
    """
    meta: Dict[str, float] = {}
    rows: List[Dict[str, float]] = []
    in_table = False

    for line in text.splitlines():
        s = line.strip()
        if not in_table:
            # ---- header region: scrape the run metadata ----------------------
            if s.startswith("Calculated polar for:"):
                meta["name"] = s.split(":", 1)[1].strip()
            # 'xtrf =   1.000 (top)        1.000 (bottom)'
            m = regex.search(r"xtrf\s*=\s*([\d.]+)\s*\(top\)\s*([\d.]+)\s*\(bottom\)", s)
            if m:
                meta["xtrf_top"] = float(m.group(1))
                meta["xtrf_bot"] = float(m.group(2))
            # 'Mach =   0.000     Re =     6.000 e 6     Ncrit =   9.000'
            # (note the spaced-out exponent in XFOIL's Re formatting)
            m = regex.search(r"Mach\s*=\s*([\d.]+)", s)
            if m:
                meta["mach"] = float(m.group(1))
            m = regex.search(r"Re\s*=\s*([\d.]+)\s*e\s*(-?\d+)", s)
            if m:
                meta["re"] = float(m.group(1)) * 10.0 ** int(m.group(2))
            m = regex.search(r"Ncrit\s*=\s*([\d.]+)", s)
            if m:
                meta["ncrit"] = float(m.group(1))
            # The dashed separator marks the start of the data table.
            if s.startswith("---"):
                in_table = True
            continue

        # ---- table region: 7 floats per converged point ----------------------
        tokens = s.split()
        if len(tokens) != len(CSV_COLUMNS):
            continue
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            continue
        rows.append(dict(zip(CSV_COLUMNS, values)))

    return meta, rows


def find_skipped(requested: Sequence[float], rows: Sequence[Dict[str, float]],
                 tol: float = 0.05) -> List[float]:
    """Requested alphas absent from the polar (= unconverged, see docstring).

    XFOIL's PACC only stores converged points, so set-difference against the
    request list is an exact non-convergence detector. The tolerance absorbs
    the dump's 3-decimal alpha formatting.
    """
    got = [row["alpha"] for row in rows]
    skipped = []
    for a in requested:
        if not any(abs(a - g) <= tol for g in got):
            skipped.append(a)
    return skipped


# =============================================================================
#  Binary resolution
# =============================================================================

def xfoil_works(argv: Sequence[str], timeout: float = 30.0) -> bool:
    """True when the candidate command starts XFOIL and quits cleanly.

    Pipes a lone QUIT and checks for a zero exit plus the XFOIL banner -
    the same smoke check documented in tools/xfoil/SOURCE.md.
    """
    try:
        proc = subprocess.run(
            list(argv), input="QUIT\n", capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "XFOIL" in proc.stdout


def resolve_xfoil(cli_path: Optional[str] = None) -> List[str]:
    """Locate a working XFOIL, returning the argv prefix to invoke it.

    Resolution order (first verified candidate wins):
      1. --xfoil command-line flag
      2. XFOIL_BIN environment variable
      3. repo binary tools/xfoil/xfoil.exe (provenance: tools/xfoil/SOURCE.md)
      4. 'wsl xfoil' - an XFOIL on the WSL2 side; works because this driver
         only ever passes RELATIVE file names with cwd set to the run dir,
         which WSL maps automatically.

    An explicit flag or env var that does not start is a hard error (the user
    asked for that binary specifically); the implicit fallbacks are probed
    quietly. Raises FileNotFoundError naming everything tried.
    """
    tried: List[str] = []

    # Explicit requests are verified and then trusted-or-fatal: silently
    # falling through to a different binary than the one named would make
    # results unreproducible.
    for source, value in (("--xfoil", cli_path), ("XFOIL_BIN", os.environ.get("XFOIL_BIN"))):
        if value:
            argv = [str(value)]
            if xfoil_works(argv):
                return argv
            raise FileNotFoundError(
                f"XFOIL given via {source} ({value}) did not start/quit cleanly"
            )

    # Implicit candidates: the repo-local binary, then the WSL fallback.
    repo_bin = REPO / "tools" / "xfoil" / "xfoil.exe"
    if repo_bin.is_file():
        argv = [str(repo_bin)]
        if xfoil_works(argv):
            return argv
        tried.append(str(repo_bin))
    if shutil.which("wsl"):
        argv = ["wsl", "xfoil"]
        if xfoil_works(argv):
            return argv
        tried.append("wsl xfoil")

    raise FileNotFoundError(
        "No working XFOIL binary found (tried: %s). Download the official "
        "MIT build into tools/xfoil/ per tools/xfoil/SOURCE.md, or point "
        "--xfoil / XFOIL_BIN at an executable." % (", ".join(tried) or "nothing")
    )


# =============================================================================
#  Polar execution
# =============================================================================

@dataclass
class PolarResult:
    """One (Re, transition) polar: converged rows plus run bookkeeping.

    ``skipped`` holds requested alphas XFOIL processed but could not
    converge; ``hung`` holds alphas whose process froze (TRCHEK2 NaN spin)
    and was killed by the watchdog. Both are absent from ``rows``.
    """
    re_value: float
    transition: str                      # 'free' or 'trip'
    rows: List[Dict[str, float]] = field(default_factory=list)
    skipped: List[float] = field(default_factory=list)
    hung: List[float] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def clmax_row(self) -> Optional[Dict[str, float]]:
        """Row carrying the maximum cl (None when nothing converged)."""
        return max(self.rows, key=lambda r: r["cl"]) if self.rows else None


def _exec_xfoil(argv: Sequence[str], script: str, run_dir: Path,
                polar_name: str, timeout: float
                ) -> Tuple[Dict[str, float], List[Dict[str, float]], bool]:
    """Run one XFOIL process; return (meta, rows, timed_out).

    PACC appends each converged point to the polar save file AS IT IS
    COMPUTED, so when the watchdog kills a frozen process the points solved
    before the freeze are still on disk - this routine parses whatever made
    it into the file in both the clean-exit and killed cases.
    """
    timed_out = False
    try:
        proc = subprocess.run(
            list(argv), input=script, capture_output=True, text=True,
            cwd=str(run_dir), timeout=timeout,
        )
        stdout_tail = "\n".join(proc.stdout.splitlines()[-15:])
        exit_note = f"exit={proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        # subprocess.run has already killed the process at this point; the
        # captured-so-far output is kept for diagnostics only.
        timed_out = True
        out = exc.stdout if isinstance(exc.stdout, str) else \
            (exc.stdout or b"").decode(errors="replace")
        stdout_tail = "\n".join(out.splitlines()[-15:])
        exit_note = f"killed after {timeout:g}s"

    polar_path = run_dir / polar_name
    # A missing dump on a CLEAN exit means XFOIL died before PACC opened the
    # file - surface the output tail, which names the failing command. After
    # a kill, a missing file simply means no point converged before the
    # freeze, which the caller handles as an empty salvage.
    if not polar_path.exists():
        if timed_out:
            return {}, [], True
        raise RuntimeError(
            f"XFOIL produced no polar file ({exit_note}); last output:\n"
            f"{stdout_tail}"
        )
    meta, rows = parse_polar_dump(polar_path.read_text(encoding="utf-8",
                                                       errors="replace"))
    return meta, rows, timed_out


def run_polar(argv: Sequence[str], coords: np.ndarray, re_value: float,
              transition: str, ncrit: float, alphas: Sequence[float],
              n_iter: int = DEFAULT_ITER,
              timeout: float = POLAR_TIMEOUT_S,
              point_timeout: float = POINT_TIMEOUT_S) -> PolarResult:
    """Execute one XFOIL polar accumulation and parse its dump.

    Runs in a throwaway temp directory with RELATIVE file names: that keeps
    XFOIL's 80-ish character filename buffer safe, isolates any droppings
    (xfoil.def etc.), and makes the 'wsl xfoil' fallback path-translation
    free. The Selig file is written fresh per run from the in-memory loop.

    Freeze recovery: the primary run sweeps every alpha in ONE process so
    each point warm-starts from its converged neighbor (best convergence
    near stall). If that process freezes (XFOIL TRCHEK2 NaN spin - observed
    deep post-stall in tripped low-Re cases), the watchdog kills it, the
    partial polar is salvaged, and every alpha past the last salvaged row is
    retried in its OWN process (cold start, own watchdog) so one frozen
    angle cannot take down the angles behind it. A retry that freezes again
    is recorded in ``hung``; retries that merely fail to converge land in
    ``skipped`` like any other unconverged point.
    """
    xtr = XTR_TRIP if transition == "trip" else XTR_FREE
    hung: List[float] = []
    with tempfile.TemporaryDirectory(prefix="xfoil_") as tmp:
        tmp_path = Path(tmp)
        # The section as XFOIL sees it; title line documents the resample.
        write_selig(tmp_path / "foil.dat", coords,
                    f"LS(1)-0413 sharp-TE {coords.shape[0]} pts")

        # ---- primary full-sweep process --------------------------------------
        script = build_xfoil_script(
            "foil.dat", "polar.txt", re_value, ncrit, xtr, alphas,
            mach=0.0, n_iter=n_iter,
        )
        meta, rows, timed_out = _exec_xfoil(argv, script, tmp_path,
                                            "polar.txt", timeout)

        # ---- salvage pass after a freeze --------------------------------------
        if timed_out:
            # Points strictly beyond the last salvaged row never ran (the
            # sweep is ascending); anything missing BELOW it was processed
            # and failed normally, i.e. genuinely unconverged, not frozen.
            last = max((r["alpha"] for r in rows), default=-math.inf)
            remaining = [a for a in alphas if a > last + 0.05]
            print(f"      watchdog: sweep froze after alpha {last:g}; "
                  f"retrying {len(remaining)} point(s) in isolation")
            for i, a in enumerate(remaining):
                # Fresh polar file per retry: PACC appends to an existing
                # file, and a stale partial header would double-count rows.
                pname = f"polar_retry_{i}.txt"
                pscript = build_xfoil_script(
                    "foil.dat", pname, re_value, ncrit, xtr, [a],
                    mach=0.0, n_iter=n_iter,
                )
                _, prow, ptimeout = _exec_xfoil(argv, pscript, tmp_path,
                                                pname, point_timeout)
                if ptimeout and not prow:
                    hung.append(a)      # froze again: hard-skip this angle
                else:
                    rows.extend(prow)   # 0 rows = unconverged, 1 = recovered

    # Merge order: retries may interleave, so sort once by alpha for stable
    # CSV/plot output regardless of which path produced each row.
    rows.sort(key=lambda r: r["alpha"])
    result = PolarResult(re_value=re_value, transition=transition,
                         rows=rows, hung=hung, meta=meta)
    # Skip census excludes frozen angles - those are reported separately so
    # the log distinguishes "did not converge" from "solver froze, killed".
    result.skipped = [a for a in find_skipped(alphas, rows)
                      if not any(abs(a - h) <= 0.05 for h in hung)]
    return result


# =============================================================================
#  Outputs: CSV, combined figure, README
# =============================================================================

def csv_name(re_value: float, transition: str, ncrit: float = 9.0) -> str:
    """Output file naming contract: polar_Re{1.5|3|6}M_N{ncrit}_{free|trip}.csv.

    The N token records the e^N amplification setting the polar was run at,
    so the gate evaluator (validation/compare_gate.find_xfoil_polar) can
    match a requested Ncrit by filename alone. Tripped polars carry the
    token too: transition there is forced at the trip stations, but the run
    setting is still part of the file's provenance and one uniform pattern
    keeps the discovery glob simple. %g collapses '9.0' to 'N9'. Legacy
    files without the token (polar_Re3M_free.csv) remain discoverable
    through find_xfoil_polar's backward-compat fallback.
    """
    return f"polar_Re{re_label(re_value)}_N{ncrit:g}_{transition}.csv"


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    """Write one polar as a plain CSV with the fixed column contract.

    alpha keeps 2 decimals (the request grid is integral degrees), the
    aerodynamic coefficients keep 6 - comfortably beyond XFOIL's own dump
    precision so nothing is lost in the round-trip.
    """
    lines = [",".join(CSV_COLUMNS)]
    for row in rows:
        lines.append(
            f"{row['alpha']:.2f}," + ",".join(f"{row[c]:.6f}" for c in CSV_COLUMNS[1:])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_polars(results: Sequence[PolarResult], png_path: Path) -> None:
    """Combined two-panel comparison figure: cl-alpha and the drag polar.

    Encoding: color = Reynolds number, linestyle = transition state (free
    solid, tripped dashed), so all six polars stay distinguishable in one
    legend without a 6-row color ramp.
    """
    fig, (ax_cl, ax_dp) = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # Stable color assignment by ascending Re so reruns with a different
    # result ordering still color identically.
    re_sorted = sorted({r.re_value for r in results})
    color_of = {re_v: f"C{i}" for i, re_v in enumerate(re_sorted)}
    style_of = {"free": "-", "trip": "--"}
    marker_of = {"free": "o", "trip": "s"}

    for res in results:
        if not res.rows:
            continue  # nothing converged; the README logs it
        a = np.array([r["alpha"] for r in res.rows])
        cl = np.array([r["cl"] for r in res.rows])
        cd = np.array([r["cd"] for r in res.rows])
        label = f"Re {re_label(res.re_value)} {res.transition}"
        kw = dict(color=color_of[res.re_value],
                  linestyle=style_of[res.transition],
                  marker=marker_of[res.transition], markersize=3.5,
                  linewidth=1.4, label=label)
        ax_cl.plot(a, cl, **kw)
        ax_dp.plot(cd, cl, **kw)

    # Lift curve panel: alpha grid matches the sweep so stall reading is easy.
    ax_cl.set_xlabel("alpha (deg)")
    ax_cl.set_ylabel("cl")
    ax_cl.set_title("Lift curve")
    ax_cl.grid(True, alpha=0.3)
    ax_cl.legend(fontsize=8, loc="lower right")

    # Drag polar panel: cd on x per convention; tripped cases sit visibly
    # right of the free-transition laminar buckets.
    ax_dp.set_xlabel("cd")
    ax_dp.set_ylabel("cl")
    ax_dp.set_title("Drag polar")
    ax_dp.grid(True, alpha=0.3)

    fig.suptitle("LS(1)-0413 XFOIL 6.99 polars - Mach 0, sharp TE, PANE repanel "
                 "(free: e^N; trip: xtr 0.05/0.05)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def mach_table(re_list: Sequence[float], nu: float, temperature: float) -> List[str]:
    """Advisory U/Mach lines for the c = 1 m convention (see module docstring).

    Speed of sound from a = sqrt(gamma R T) with T from aircraft.yaml; each
    line flags whether the nominal speed respects the M < 0.2 guideline.
    The flag is informational - the documented decision is to keep Re 6M.
    """
    a_sound = math.sqrt(GAMMA_AIR * R_AIR * temperature)
    lines = []
    # Dedupe before sorting: callers may pass one entry per (Re, transition)
    # pair, and the advisory table wants each Reynolds number exactly once.
    for re_v in sorted(set(re_list)):
        u = re_v * nu / CHORD_M
        mach = u / a_sound
        verdict = "ok (M < 0.2)" if mach < MACH_GUIDELINE else \
            "EXCEEDS 0.2 guideline - kept as Re trend point (see README note)"
        lines.append(f"Re {re_label(re_v):>5}: U = {u:7.2f} m/s  M = {mach:5.3f}  {verdict}")
    return lines


def write_readme(out_dir: Path, results: Sequence[PolarResult], ncrit: float,
                 nu: float, temperature: float, airfoil_src: str,
                 alphas: Sequence[float]) -> None:
    """Auto-generate validation/xfoil/README.md (regenerated every run).

    Records the speed/Re convention, the Mach-guideline decision for the
    Re-6M point, the run matrix with Clmax / stall alpha / skip log, and the
    exact regeneration command - everything a reviewer needs to audit or
    reproduce the polars without reading the driver source.
    """
    # Show the section source repo-relative when it lives inside the repo:
    # this README is committed, and an absolute local path would churn on
    # every clone. Out-of-repo overrides (different drive etc.) stay as-is.
    try:
        airfoil_shown = Path(airfoil_src).resolve().relative_to(REPO).as_posix()
    except ValueError:
        airfoil_shown = airfoil_src
    lines = [
        "# XFOIL clean-section polars - LS(1)-0413 (Phase-1 baseline)",
        "",
        "Auto-generated by `validation/xfoil_polar.py` - do not hand-edit;",
        "rerun the driver to refresh. Binary provenance: `tools/xfoil/SOURCE.md`.",
        "",
        "## Setup",
        "",
        f"* Section: `{airfoil_shown}` resampled to {N_FOIL_POINTS} cosine-clustered",
        "  points with a sharp (closed) TE via `geometry.airfoil`, then re-paneled",
        "  inside XFOIL (`PANE`) - feeding the raw cosine points straight in as",
        "  panels produced spurious premature transition and BL-march failures.",
        f"* XFOIL 6.99, incompressible (`MACH 0`), `ITER {DEFAULT_ITER}`, PACC accumulation.",
        f"* Free transition: e^N with Ncrit = {ncrit:g}. Tripped: forced",
        f"  xtr = {XTR_TRIP[0]:g}/{XTR_TRIP[1]:g} (top/bottom).",
        f"* AoA schedule: {alphas[0]:g} to {alphas[-1]:g} deg "
        f"({len(alphas)} points; 2-deg steps to 8, 1-deg steps to 20 per spec).",
        f"* File naming: polar_Re<tag>_N{ncrit:g}_<free|trip>.csv - the N token",
        "  records the e^N setting so the gate evaluator matches Ncrit by",
        "  filename; legacy names without the token remain discoverable.",
        "",
        "## Speed / Reynolds convention (all 2D cases this milestone)",
        "",
        f"Chord c = {CHORD_M:g} m and U = Re * nu / c, with nu = {nu:g} m^2/s",
        "(`atmosphere.kinematic_viscosity`, aircraft.yaml, ISA sea level).",
        "Nominal speeds and Mach numbers (a = sqrt(gamma R T), T = "
        f"{temperature:g} K):",
        "",
        "```",
        *mach_table([r.re_value for r in results] or list(DEFAULT_RE_LIST),
                    nu, temperature),
        "```",
        "",
        "**Re = 6M decision:** the nominal U ~ 87.6 m/s implies M ~ 0.26, above",
        "the quasi-incompressible M < 0.2 guideline. Re 6M is KEPT at c = 1 m",
        "anyway because the Mach number is purely nominal here: XFOIL runs",
        "strictly incompressible, and the matching OpenFOAM case uses",
        "incompressible simpleFoam where U only sets Re (the nondimensional",
        "solution is Reynolds-similar - rescaling the chord at fixed Re would",
        "change nothing). Re 6M stays as the cruise-end trend point per the",
        "spec, matching the NASA Langley test Re for this section and common",
        "simpleFoam airfoil-study practice.",
        "",
        "## Results",
        "",
        "| polar | converged | Clmax | alpha_stall (deg) | cd_min | skipped alphas |",
        "|-------|-----------|-------|-------------------|--------|----------------|",
    ]
    for res in results:
        peak = res.clmax_row
        cd_min = min((r["cd"] for r in res.rows), default=float("nan"))
        # Skip column merges plain non-convergence with watchdog-killed
        # freezes (tagged so a reviewer can tell them apart), sorted by
        # alpha so the cell reads like the sweep did.
        entries = [(a, "") for a in res.skipped] + \
                  [(a, " (froze)") for a in res.hung]
        entries.sort(key=lambda t: t[0])
        skipped = ", ".join(f"{a:g}{tag}" for a, tag in entries) if entries else "-"
        lines.append(
            f"| {csv_name(res.re_value, res.transition, ncrit)} "
            f"| {len(res.rows)}/{len(alphas)} "
            f"| {peak['cl']:.3f} | {peak['alpha']:.1f} | {cd_min:.5f} | {skipped} |"
            if peak else
            f"| {csv_name(res.re_value, res.transition, ncrit)} | 0/{len(alphas)} "
            f"| - | - | - | all |"
        )
    lines += [
        "",
        "Skipped alphas are points XFOIL could not converge within "
        f"{DEFAULT_ITER} iterations (PACC stores converged points only); they",
        "are omitted from the CSVs rather than silently included. Points",
        "tagged '(froze)' hit a known XFOIL 6.99 hard freeze (TRCHEK2 NaN",
        "spin, seen deep post-stall in tripped low-Re cases) and were killed",
        "by the driver's watchdog after an isolated retry froze as well.",
        "",
        "Combined comparison figure: `polars.png` (lift curves + drag polars).",
        "",
        "## Regenerate",
        "",
        "```",
        "python validation/xfoil_polar.py",
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
#  CLI driver
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch XFOIL polar sweep of the LS(1)-0413 wing section "
                    "(Phase-1 clean-section validation baseline)")
    parser.add_argument("--xfoil", default=None,
                        help="path to the XFOIL executable (default: XFOIL_BIN "
                             "env, then tools/xfoil/xfoil.exe, then 'wsl xfoil')")
    parser.add_argument("--airfoil", default=None,
                        help="airfoil .dat file (default: wing.airfoil_file "
                             "from aircraft.yaml)")
    parser.add_argument("--re", type=float, nargs="+",
                        default=list(DEFAULT_RE_LIST),
                        help="Reynolds numbers (default: 1.5e6 3e6 6e6 per spec)")
    parser.add_argument("--ncrit", type=float, default=9.0,
                        help="e^N critical amplification for free transition "
                             "(default 9)")
    parser.add_argument("--tripped", action="store_true",
                        help="run only the tripped cases (forced xtr 0.05/0.05)")
    parser.add_argument("--free", action="store_true",
                        help="run only the free-transition cases")
    parser.add_argument("--iter", type=int, default=DEFAULT_ITER,
                        help="XFOIL viscous iteration cap per point (default 200)")
    parser.add_argument("--out", default=str(REPO / "validation" / "xfoil"),
                        help="output directory (default validation/xfoil)")
    args = parser.parse_args(argv)

    # Transition selection: either flag alone restricts the matrix; neither
    # (or both) runs the full free+tripped Phase-1 set.
    if args.tripped and not args.free:
        transitions = ["trip"]
    elif args.free and not args.tripped:
        transitions = ["free"]
    else:
        transitions = ["free", "trip"]

    # Aircraft-derived inputs: viscosity and temperature for the convention
    # report, and the committed wing-section file as the default geometry.
    # Nothing aircraft-specific is hard-coded in this driver.
    ac = load_aircraft(REPO / "aircraft.yaml")
    nu = ac.atmosphere.kinematic_viscosity
    temperature = ac.atmosphere.temperature
    airfoil_src = args.airfoil or str(REPO / ac.wing.airfoil_file)

    # Section preparation: one resample shared by every run in the matrix.
    coords = resample_airfoil(load_airfoil(airfoil_src),
                              n_points=N_FOIL_POINTS, te="sharp")
    alphas = build_alpha_schedule()

    # Binary resolution is fatal-with-instructions when nothing works; the
    # message points at tools/xfoil/SOURCE.md for the re-download steps.
    try:
        xfoil_argv = resolve_xfoil(args.xfoil)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("XFOIL polar matrix - LS(1)-0413 clean section (Phase-1 baseline)")
    print("=" * 78)
    print(f"binary    : {' '.join(xfoil_argv)}")
    print(f"airfoil   : {airfoil_src}")
    print(f"resample  : {N_FOIL_POINTS} pts, sharp TE, PANE repanel inside XFOIL")
    print(f"alphas    : {alphas[0]:g}..{alphas[-1]:g} deg ({len(alphas)} points)")
    print(f"matrix    : Re {', '.join(re_label(r) for r in args.re)} x "
          f"{'/'.join(transitions)}")
    print("")
    print("speed convention: c = 1.0 m, U = Re * nu / c "
          f"(nu = {nu:g} m^2/s from aircraft.yaml)")
    for line in mach_table(args.re, nu, temperature):
        print("  " + line)
    print("")

    # ---- run the matrix -------------------------------------------------------
    results: List[PolarResult] = []
    for re_value in args.re:
        for transition in transitions:
            res = run_polar(xfoil_argv, coords, re_value, transition,
                            args.ncrit, alphas, n_iter=args.iter)
            results.append(res)
            write_csv(out_dir / csv_name(re_value, transition, args.ncrit),
                      res.rows)

            # Per-polar console report: convergence census, peak, skips.
            peak = res.clmax_row
            peak_txt = (f"Clmax {peak['cl']:.3f} @ {peak['alpha']:.1f} deg"
                        if peak else "NO CONVERGED POINTS")
            print(f"  Re {re_label(re_value):>5} {transition:<4}: "
                  f"{len(res.rows):2d}/{len(alphas)} converged, {peak_txt}")
            if res.skipped:
                print(f"      skipped (unconverged): "
                      f"{', '.join(f'{a:g}' for a in res.skipped)} deg")
            if res.hung:
                print(f"      skipped (solver froze, killed): "
                      f"{', '.join(f'{a:g}' for a in res.hung)} deg")

    # ---- combined figure + README ---------------------------------------------
    plot_polars(results, out_dir / "polars.png")
    write_readme(out_dir, results, args.ncrit, nu, temperature,
                 airfoil_src, alphas)
    print("")
    print(f"wrote {len(results)} CSV polar(s), polars.png, README.md -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
