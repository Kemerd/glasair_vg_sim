# -*- coding: utf-8 -*-
"""One-command M1 driver: Phase-1 clean-section validation, end to end.

Sequence (spec milestone M1; each step degrades gracefully so the command is
runnable TODAY, before the WSL2 + OpenFOAM v2506 install is repaired):

  (a) XFOIL baseline   - ensure the polar CSVs for the requested Re exist in
                         validation/xfoil/; if absent, invoke
                         validation/xfoil_polar.py to generate them.
  (b) case generation  - build the AoA sweep case set through
                         scripts/build_validation_case.py: alpha -4..+20 deg
                         in 2-deg steps, refined to 1-deg from +8 up
                         (near-stall resolution per spec Phase-1 item 3),
                         at Re 3e6 / mesh level 0 by default.
  (c) solve            - IF a solver is reachable (native simpleFoam on PATH,
                         or via 'wsl -e bash' with the openfoam2506 bashrc),
                         run blockMesh / checkMesh / decomposePar / simpleFoam
                         per case, N cases in flight at a time (default 2),
                         with every tool's output captured to log.<tool> in
                         the case directory. Convergence is judged by
                         scripts/parse_forces.py (residuals + force-flatness
                         per spec section 4); cases that pass then face the
                         spec wall-y+ gate (parse_forces.gate_case_yplus:
                         per-case histogram printed, case FAILED when y+ > 2
                         on more than 1% of airfoil wall faces); surviving
                         cases get the
                         boundary-layer extraction (scripts/extract_bl.py),
                         the wall-Cf conversion (scripts/extract_cf.py) and
                         the upper-surface transition pickup. Post-stall
                         cases (alpha >= --pimple-alpha, default 14 deg)
                         that fail to converge steady are retried ONCE with
                         the pimpleFoam pseudo-transient variant from the
                         template's pimple_overrides/ directory. If NO solver
                         is reachable the driver stops after case generation
                         with a 'SOLVER MISSING - generated N cases ready'
                         summary and exit code 0 (the current state until the
                         WSL repair lands).
  (d) gate             - call validation/compare_gate.py with whatever
                         artifacts exist; the gate marks missing inputs
                         SKIPPED rather than failing the run.

Idempotency: each case carries an m1_status.json marker once solved; cases
whose marker records a convergence PASS are skipped on re-run (override with
--force). The polar CSV consumed by the gate is re-aggregated from the
markers on every run, so a partially-solved sweep still produces a partial
polar and a partial (SKIPPED-annotated) gate report.

Sibling-script contracts (defined here so the pieces stay decoupled):
  scripts/build_validation_case.py --alpha A --re RE --level L --out DIR
      builds one 2D case and prints the created case directory as the last
      directory-path line on stdout (fallback: this driver scans --out for a
      directory whose name carries the alpha/Re/level tags).
  scripts/parse_forces.py
      convergence gate + windowed force means. Preferred path: direct import
      of scripts.parse_forces.gate_case(case_dir) -> ConvergenceVerdict with
      .passed/.cl_mean/.cd_mean/.cm_mean. CLI fallback: exit 0 = PASS,
      1 = FAIL, 2 = load error, with 'mean Cl : ...' lines on stdout.
      Unparseable output marks the case NOT converged - never silently
      included (spec section 4). The same module's gate_case_yplus(case_dir)
      -> YPlusVerdict is the spec wall-y+ gate; solve_case calls it on every
      case that passed convergence, prints .report() (the spec-required
      per-case y+ histogram) and treats .passed == False as a case FAIL.
  scripts/extract_bl.py CASE_DIR
      wall-normal BL profile sampling -> results/bl_table_{case}.csv (the
      delta/delta* table of spec Phase-1 item 5), run on converged cases.
  scripts/extract_cf.py CASE_DIR
      the transition-gate data producer: converts the raw wall-shear surface
      dump written by the case template's wallShearStress function object
      (postProcessing/wallCf/<time>/) into the suction-surface skin-friction
      curve CASE_DIR/postProcessing/cf_upper.csv (columns x_c, cf); this
      driver runs it on converged cases and feeds the curve to the shared
      transition detector in validation/compare_gate.py.

Run:  python scripts/run_validation.py [--re 3e6] [--level 0] [--jobs 2]
          [--cores 8] [--alphas ...] [--pimple-alpha 14] [--no-solve]
          [--force] [--skip-xfoil]

ASCII-only console output; all file IO UTF-8.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re as _regex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python scripts/run_validation.py", Python puts scripts/ on
#  sys.path rather than the repo root, which would hide the geometry and
#  validation packages. Inserting this file's grandparent mirrors conftest.py
#  and keeps the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geometry.units import load_aircraft  # noqa: E402
from validation.compare_gate import (  # noqa: E402
    detect_transition_from_cf,
    main as compare_gate_main,
    read_case_application,
    read_csv_columns,
    find_xfoil_polar,
    re_tag,
)

# Fixed repo locations every step below works against. Cases for this
# milestone live under cases/validation/ (templates elsewhere in cases/ are
# untouched); aggregated polars and transition tables land in
# results/validation/ where the gate evaluator's defaults expect them.
YAML_PATH = REPO / "aircraft.yaml"
CASE_ROOT = REPO / "cases" / "validation"
RESULTS_DIR = REPO / "results" / "validation"
XFOIL_DIR = REPO / "validation" / "xfoil"
XFOIL_SCRIPT = REPO / "validation" / "xfoil_polar.py"
BUILD_SCRIPT = REPO / "scripts" / "build_validation_case.py"
PARSE_FORCES = REPO / "scripts" / "parse_forces.py"
EXTRACT_BL = REPO / "scripts" / "extract_bl.py"
EXTRACT_CF = REPO / "scripts" / "extract_cf.py"

# pimpleFoam pseudo-transient fallback variant (post-stall retries): the
# template ships the full replacement dictionaries; the builder copies them
# into every instantiated case, and apply_pimple_variant() below swaps them
# into system/ when the steady solve refuses to converge.
PIMPLE_OVERRIDES = (REPO / "cases" / "validation_2d" / "template"
                    / "pimple_overrides")
PIMPLE_VARIANT_FILES = ("controlDict", "fvSchemes", "fvSolution")

# Default AoA threshold for the automatic pimpleFoam retry: the LS(1)-0413
# clean-section stall lands in the mid-teens, so 14 deg marks "post-stall
# enough that a steady non-convergence is physics, not a setup bug".
# Overridable per run via --pimple-alpha.
PIMPLE_ALPHA_DEFAULT = 14.0

# ESI OpenFOAM v2506 environment file inside the WSL2 Ubuntu guest - the
# exact package this project pins (README bootstrap section). Sourced before
# every solver command when running through the WSL bridge.
OF_BASHRC = "/usr/lib/openfoam/openfoam2506/etc/bashrc"

# Per-case status marker (idempotency record). Holding it in the case
# directory keeps case + verdict together for the reproducibility audit.
STATUS_NAME = "m1_status.json"


# =============================================================================
#  AoA schedule (spec Phase-1 item 3: 2-deg steps, 1-deg near stall)
# =============================================================================
def alpha_schedule() -> List[float]:
    """The Phase-1 sweep: -4..+6 in 2-deg steps, then 8..20 in 1-deg steps.

    The 1-deg refinement starts at +8 deg so the approach-to-stall region of
    the LS(1)-0413 (clean-section stall expected in the mid-teens) is fully
    resolved while the linear range stays cheap.
    """
    coarse = [float(a) for a in range(-4, 8, 2)]      # -4, -2, 0, 2, 4, 6
    fine = [float(a) for a in range(8, 21)]           # 8, 9, ..., 20
    return coarse + fine


def alpha_token(alpha: float) -> str:
    """Canonical alpha tag used in case-directory names, e.g. 'aoa4'.

    Mirrors scripts/build_validation_case.case_name (the naming authority):
    %g formatting with '-' -> 'm' and '.' -> 'p', so -2.5 deg tags as
    'aoam2p5' inside 'val2d_aoam2p5_re1.5e6_lvl1'. Keeping the two scripts
    on one spelling is what makes the existing-case reuse (idempotency)
    actually find the directories the builder wrote.
    """
    return "aoa" + f"{alpha:g}".replace("-", "m").replace(".", "p")


# =============================================================================
#  Solver discovery (native PATH first, then the WSL bridge)
# =============================================================================
def _probe_wsl_simplefoam(timeout_s: float = 20.0) -> bool:
    """Hang-proof check for simpleFoam inside the default WSL distro.

    On a host with a broken WSL service (the current HNS 0x8007271d state on
    this workstation) 'wsl -e ...' can block FOREVER without printing a
    byte. Pipe-based capture would then deadlock even after a timeout kill,
    because WSL's child processes inherit the pipe handles and Python keeps
    draining them. Defense: capture to a temp FILE (reading a file never
    blocks), wait with a hard timeout, and force-kill the whole process tree
    on expiry. Every failure mode returns False - the driver then simply
    dry-runs.
    """
    probe = f"source {OF_BASHRC} 2>/dev/null || true; command -v simpleFoam"
    fd, tmp_name = tempfile.mkstemp(prefix="wsl_probe_", suffix=".txt")
    rc: Optional[int] = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            try:
                proc = subprocess.Popen(
                    ["wsl", "-e", "bash", "-lc", probe],
                    stdin=subprocess.DEVNULL, stdout=out,
                    stderr=subprocess.STDOUT)
            except OSError:
                return False                      # no wsl.exe at all
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                # Kill the full tree: wsl.exe alone dying can leave the
                # relay children running and the service wedged harder.
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    capture_output=True)
                print(f"  [probe] WSL did not answer within {timeout_s:.0f} s "
                      f"(service broken?) - treating solver as missing")
                return False
        if rc != 0:
            return False
        text = Path(tmp_name).read_text(encoding="utf-8", errors="replace")
        return bool(text.strip())
    finally:
        # Best-effort cleanup; a lingering WSL child may still hold the
        # file open, in which case the temp file is abandoned to the OS.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def probe_solver() -> Optional[str]:
    """Return 'native', 'wsl', or None depending on where simpleFoam lives.

    Native covers two situations: this script running inside WSL/Linux with
    the OpenFOAM environment already sourced (the run_validation.sh path),
    or any hypothetical Windows build on PATH. The WSL probe sources the
    openfoam2506 bashrc first, mirroring exactly what the runner will do, so
    a positive probe guarantees the run commands will also resolve.
    """
    if shutil.which("simpleFoam"):
        return "native"
    # Windows host: ask the default WSL distro. Failures of any kind (no
    # WSL, hung WSL service, no distro, no OpenFOAM) mean "not available".
    if sys.platform == "win32" and _probe_wsl_simplefoam():
        return "wsl"
    return None


def to_wsl_path(p: Path) -> str:
    """Translate a Windows path to its /mnt/<drive>/ WSL equivalent."""
    s = str(Path(p).resolve())
    drive, rest = s[0].lower(), s[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def run_case_command(mode: str, case_dir: Path, command: str,
                     log_name: str) -> int:
    """Run one solver-chain command in a case directory, log captured.

    ``mode`` is the probe_solver() result. Output (stdout+stderr) streams to
    CASE_DIR/log.<log_name> so a failed case carries its full evidence; the
    return code is the tool's own exit status.
    """
    log_path = case_dir / f"log.{log_name}"
    with open(log_path, "w", encoding="utf-8") as log:
        if mode == "wsl":
            # Bridge through WSL: source the pinned environment, cd to the
            # translated case path, then run the tool. Single -lc string so
            # the OpenFOAM aliases/functions resolve as they would in an
            # interactive Ubuntu shell.
            shell = (f"source {OF_BASHRC} 2>/dev/null || true; "
                     f"cd '{to_wsl_path(case_dir)}' && {command}")
            proc = subprocess.run(["wsl", "-e", "bash", "-lc", shell],
                                  stdout=log, stderr=subprocess.STDOUT)
        else:
            # Native (Linux/WSL-side python3, environment already sourced by
            # run_validation.sh): run in-place with the case dir as cwd.
            proc = subprocess.run(command, shell=True, cwd=str(case_dir),
                                  stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


# =============================================================================
#  Step (a): XFOIL baseline polars
# =============================================================================
def ensure_xfoil_polars(re_value: float, ncrit: int, skip: bool) -> bool:
    """Make sure an XFOIL polar exists for this Re; generate it if possible.

    Returns True when a usable polar is present afterwards. A missing
    validation/xfoil_polar.py is reported (that script is a separate M1
    deliverable) but does not abort the driver - the gate will mark the
    XFOIL-dependent rows SKIPPED.
    """
    found, note = find_xfoil_polar(XFOIL_DIR, re_value, ncrit)
    if found is not None:
        print(f"  [xfoil] polar present: {found.name}")
        return True
    if skip:
        print(f"  [xfoil] missing ({note}); --skip-xfoil set, not generating")
        return False
    if not XFOIL_SCRIPT.is_file():
        print(f"  [xfoil] missing ({note}) and {XFOIL_SCRIPT.name} not "
              f"present yet - gate will SKIP XFOIL comparisons")
        return False

    # Invoke the polar generator for the requested Re only (its --re flag
    # accepts a list; no-args would sweep the full 1.5/3/6 M spec grid).
    # Targeted generation keeps the driver responsive; the other Re points
    # are produced on their own runs or by calling xfoil_polar.py directly.
    # --ncrit is forwarded so the generated file carries the SAME e^N
    # setting (and N filename token) the gate will look for - otherwise a
    # non-default --ncrit run would generate a polar it can never match.
    print(f"  [xfoil] generating Re {re_value:g} polar (Ncrit {ncrit}) via "
          f"{XFOIL_SCRIPT.name} ...")
    proc = subprocess.run(
        [sys.executable, str(XFOIL_SCRIPT), "--re", f"{re_value:g}",
         "--ncrit", str(ncrit)],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-10:])
        print(f"  [xfoil] generator exited {proc.returncode}; last output:")
        for line in tail.splitlines():
            print(f"      {line}")
    found, note = find_xfoil_polar(XFOIL_DIR, re_value, ncrit)
    if found is not None:
        print(f"  [xfoil] polar ready: {found.name}")
        return True
    print(f"  [xfoil] still missing after generation attempt ({note})")
    return False


# =============================================================================
#  Step (b): case generation
# =============================================================================
def find_case_dir(out_root: Path, alpha: float, re_value: float,
                  level: int) -> Optional[Path]:
    """Locate an existing case directory by its alpha/Re/level name tags.

    Fallback discovery when the builder's stdout did not yield a path: scan
    two directory levels under the case root for a name containing both the
    canonical alpha token and the Re tag (level directories may nest them).
    """
    if not out_root.is_dir():
        return None
    # Trailing '_' pins the token boundary: the builder always follows the
    # alpha tag with '_re', so 'aoa2_' cannot false-match 'aoa20_...'.
    a_tok = alpha_token(alpha) + "_"
    tag = re_tag(re_value)
    lvl = f"lvl{level}"
    hits: List[Path] = []
    for d in sorted(out_root.glob("**/")):
        name = d.name
        path_str = str(d.relative_to(out_root))
        # Alpha token must be in the leaf name; Re/level tags may live in
        # the leaf or in a parent directory along the relative path.
        if a_tok in name and tag in path_str and lvl in path_str:
            hits.append(d)
    return hits[0] if hits else None


def build_cases(alphas: List[float], re_value: float, level: int
                ) -> Tuple[List[Tuple[float, Path]], List[float]]:
    """Build (or find) the case directory for every alpha in the sweep.

    Returns (built, failed): built as (alpha, case_dir) pairs, failed as the
    alphas that could not be generated. A missing build_validation_case.py
    (separate M1 deliverable) yields an empty build with a loud notice.
    """
    built: List[Tuple[float, Path]] = []
    failed: List[float] = []
    if not BUILD_SCRIPT.is_file():
        print(f"  [build] CASE BUILDER MISSING - {BUILD_SCRIPT} not present "
              f"yet; no cases generated this run")
        return built, [a for a in alphas]

    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    for alpha in alphas:
        # Idempotency at the build step: an existing case directory is
        # reused as-is (the builder owns regeneration semantics; --force
        # here only re-runs the SOLVE, never clobbers dictionaries).
        existing = find_case_dir(CASE_ROOT, alpha, re_value, level)
        if existing is not None:
            built.append((alpha, existing))
            continue

        # Builder contract: prints the created case directory as the last
        # directory-path line on stdout (see module docstring).
        cmd = [sys.executable, str(BUILD_SCRIPT),
               "--alpha", f"{alpha:g}", "--re", f"{re_value:g}",
               "--level", str(level), "--out", str(CASE_ROOT)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO))
        case_dir: Optional[Path] = None
        if proc.returncode == 0:
            # Take the LAST stdout line that names an existing directory.
            for line in reversed(proc.stdout.splitlines()):
                cand = Path(line.strip())
                if line.strip() and cand.is_dir():
                    case_dir = cand
                    break
            if case_dir is None:
                case_dir = find_case_dir(CASE_ROOT, alpha, re_value, level)
        if case_dir is None:
            failed.append(alpha)
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-5:])
            print(f"  [build] FAILED alpha {alpha:+.1f} deg "
                  f"(rc {proc.returncode}); last output:")
            for line in tail.splitlines():
                print(f"      {line}")
        else:
            built.append((alpha, case_dir))
    return built, failed


# =============================================================================
#  Step (c): solve + convergence gate + BL extraction, jobs-wide
# =============================================================================
def read_status(case_dir: Path) -> Optional[dict]:
    """Load a case's m1_status.json marker; None when absent/corrupt."""
    marker = case_dir / STATUS_NAME
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_status(case_dir: Path, payload: dict) -> None:
    """Persist the per-case verdict marker (UTF-8, LF, human-diffable)."""
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S")
    with open(case_dir / STATUS_NAME, "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def decompose_cores(case_dir: Path, fallback: int) -> int:
    """numberOfSubdomains from the case's decomposeParDict (else fallback).

    The case template (build_validation_case.py deliverable) owns the
    decomposition; mpirun's -np MUST match it, so the dictionary is the
    authority and the --cores flag only covers a template that lacks one.
    """
    dict_path = case_dir / "system" / "decomposeParDict"
    if dict_path.is_file():
        text = dict_path.read_text(encoding="utf-8", errors="replace")
        m = _regex.search(r"numberOfSubdomains\s+(\d+)\s*;", text)
        if m:
            return int(m.group(1))
    return fallback


def parse_forces_means(case_dir: Path) -> Optional[dict]:
    """Convergence verdict + windowed force means via scripts/parse_forces.

    Preferred path: import gate_case() directly (scripts/ resolves as a
    namespace package off the repo root) - the verdict object carries the
    spec section-4 convergence gate plus the window means in one call.
    Fallback path (older/replaced parser): run the CLI and scrape its
    'mean Cl : ...' report lines, taking the exit code as the verdict
    (0 = PASS, 1 = FAIL). Returns None when the parser is missing or its
    output is unusable - the caller then marks the case NOT converged
    (auto-flagged, never silently included, per spec section 4).
    """
    if not PARSE_FORCES.is_file():
        print(f"    [forces] {PARSE_FORCES.name} not present yet - case "
              f"cannot be convergence-gated")
        return None

    # ---- preferred: in-process call -------------------------------------
    try:
        from scripts.parse_forces import gate_case
    except ImportError:
        gate_case = None
    if gate_case is not None:
        try:
            verdict = gate_case(Path(case_dir))
            return {"converged": bool(verdict.passed),
                    "cl": float(verdict.cl_mean),
                    "cd": float(verdict.cd_mean),
                    "cm": float(verdict.cm_mean)}
        except (FileNotFoundError, ValueError) as exc:
            # Missing postProcessing / malformed history: environment
            # problem, reported and treated as not-gateable.
            print(f"    [forces] {exc}")
            return None

    # ---- fallback: CLI + report scrape -----------------------------------
    proc = subprocess.run(
        [sys.executable, str(PARSE_FORCES), str(case_dir)],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode == 2:        # load error by the parse_forces contract
        return None
    text = proc.stdout or ""
    found: Dict[str, float] = {}
    for key in ("cl", "cd", "cm"):
        # Accept 'mean Cl : +0.123' as well as bare 'cl = 0.123' spellings,
        # case-insensitively, so cosmetic report changes do not break us.
        m = _regex.search(rf"(?i)\b(?:mean\s+)?{key}\s*[=:]\s*([-+0-9.eE]+)",
                          text)
        if m:
            try:
                found[key] = float(m.group(1))
            except ValueError:
                pass
    if len(found) == 3:
        return {"converged": proc.returncode == 0, **found}
    return None


def case_yplus_verdict(case_dir: Path):
    """Spec section-4 wall-y+ gate for one case via scripts/parse_forces.

    Thin indirection mirroring parse_forces_means: the in-process import is
    the preferred path (the CONTRACT comment above gate_case_yplus in
    scripts/parse_forces.py names THIS caller), and module-level dispatch
    keeps the hook monkeypatch-friendly for the solve_case tests. Returns
    the YPlusVerdict, or None when the y+ machinery is missing (an old or
    replaced parse_forces) - the caller then flags the case rather than
    silently skipping a spec gate.
    """
    try:
        from scripts.parse_forces import gate_case_yplus
    except ImportError:
        return None
    return gate_case_yplus(Path(case_dir))


def extract_transition(case_dir: Path) -> Optional[float]:
    """Post-process one converged case and pick up the transition x/c.

    Three distinct artifacts meet here: extract_bl produces the delta/delta*
    BL table (spec Phase-1 item 5, results/bl_table_*.csv) from the case's
    wall-normal sample sets; extract_cf is the transition-gate DATA PRODUCER
    - it converts the wallShearStress function-object surface dump
    (postProcessing/wallCf/<time>/) into the suction-surface Cf curve
    postProcessing/cf_upper.csv (columns x_c, cf); and the TRANSITION pickup
    then reads that curve - the glob below prefers files whose name marks
    the upper/top surface. The transition definition itself lives in
    validation/compare_gate.detect_transition_from_cf (single shared
    implementation, documented there).
    """
    if EXTRACT_BL.is_file():
        proc = subprocess.run(
            [sys.executable, str(EXTRACT_BL), str(case_dir)],
            capture_output=True, text=True, cwd=str(REPO))
        if proc.returncode != 0:
            print(f"    [bl] extract_bl exited {proc.returncode} "
                  f"(BL table unavailable for this case)")

    # Cf production: preferred path is the in-process call (same pattern as
    # parse_forces above); the CLI fallback covers a replaced module. A
    # failure here only loses gate (c) for this case - reported, not fatal.
    try:
        from scripts.extract_cf import extract_case as _extract_cf_case
    except ImportError:
        _extract_cf_case = None
    if _extract_cf_case is not None:
        try:
            _extract_cf_case(Path(case_dir))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"    [cf] {exc}")
    elif EXTRACT_CF.is_file():
        proc = subprocess.run(
            [sys.executable, str(EXTRACT_CF), str(case_dir)],
            capture_output=True, text=True, cwd=str(REPO))
        if proc.returncode != 0:
            print(f"    [cf] extract_cf exited {proc.returncode} "
                  f"(transition unavailable for this case)")

    # Cf-curve discovery: explicit upper/top names first, then any cf*.csv.
    post = case_dir / "postProcessing"
    candidates: List[Path] = []
    if post.is_dir():
        for pattern in ("**/cf*upper*.csv", "**/cf*top*.csv", "**/cf*.csv"):
            candidates = sorted(post.glob(pattern))
            if candidates:
                break
    if not candidates:
        return None
    try:
        table = read_csv_columns(candidates[0])
    except (ValueError, OSError):
        return None
    if "x_c" not in table or "cf" not in table:
        return None
    return detect_transition_from_cf(table["x_c"], table["cf"])


def apply_pimple_variant(case_dir: Path) -> Path:
    """Swap a case's system/ dictionaries to the pimpleFoam variant.

    The post-instantiation hook of the documented post-stall fallback: the
    full replacement dictionaries (controlDict / fvSchemes / fvSolution,
    deltas commented inside each file) ship in the template's
    pimple_overrides/ directory, which the case builder copies into every
    instantiated case. The case-local copy is preferred; cases built before
    the variant existed fall back to the repo template so the retry never
    depends on a rebuild. Returns the directory the variant came from.
    Reverting is a rebuild (build_validation_case.py --force) - dictionaries
    are never hand-restored, per the reproducibility rule.
    """
    case_dir = Path(case_dir)
    src_dir = case_dir / "pimple_overrides"
    if not src_dir.is_dir():
        src_dir = PIMPLE_OVERRIDES
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"pimple variant dictionaries not found (neither "
            f"{case_dir / 'pimple_overrides'} nor {PIMPLE_OVERRIDES})")
    sys_dir = case_dir / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    for name in PIMPLE_VARIANT_FILES:
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"{src_dir}: variant file {name} missing")
        shutil.copyfile(src, sys_dir / name)
    return src_dir


def run_solver_chain(mode: str, case_dir: Path, n_cores: int, app: str,
                     with_mesh: bool) -> Optional[str]:
    """Mesh (optionally) + decompose + solve + reconstruct for one case.

    Returns the name of the first failed step, or None on success. The
    pimpleFoam retry reuses the already-validated mesh, so ``with_mesh``
    is False there and the chain restarts at decomposePar.
    """
    chain: List[Tuple[str, str]] = []
    if with_mesh:
        chain += [("blockMesh", "blockMesh"), ("checkMesh", "checkMesh")]
    chain += [
        ("decomposePar", "decomposePar -force"),
        (app, f"mpirun -np {n_cores} {app} -parallel"),
        ("reconstructPar", "reconstructPar -latestTime"),
    ]
    for log_name, command in chain:
        rc = run_case_command(mode, case_dir, command, log_name)
        if rc != 0:
            return log_name
        if log_name == "checkMesh":
            # checkMesh exits 0 even when checks FAIL (its status only says
            # it ran), so the real gate is the log content - same rule as
            # the case's own Allrun: a failing mesh must never reach the
            # solver (spec section 4 mesh rules).
            log_path = case_dir / "log.checkMesh"
            if log_path.is_file() and _regex.search(
                    r"Failed [0-9]+ mesh checks",
                    log_path.read_text(encoding="utf-8", errors="replace")):
                return log_name
    return None


def solve_case(mode: str, alpha: float, case_dir: Path, cores: int,
               force: bool, pimple_alpha: float = PIMPLE_ALPHA_DEFAULT
               ) -> dict:
    """Full per-case chain: mesh, decompose, solve, gates, post-processing.

    Gates, in order: the parse_forces convergence gate (residuals + force
    flatness), then - on converged solutions only, after any pimple retry -
    the spec section-4 wall-y+ gate (gate_case_yplus; histogram printed,
    y+ FAIL/INDETERMINATE flags the case and excludes it from the polar).

    Returns the status payload that also lands in m1_status.json. Designed
    to run inside a worker thread - all heavy lifting is in subprocesses, so
    'jobs'-wide threading is the right concurrency model here (the GIL only
    sees waiting).

    Post-stall fallback (spec Phase-1 item 3): when the steady simpleFoam
    attempt of a case at alpha >= ``pimple_alpha`` fails to converge -
    either the solver dies (post-stall divergence) or the parse_forces gate
    rejects it - the case is retried ONCE with the pimpleFoam localEuler
    pseudo-transient variant (apply_pimple_variant above). Mesh failures
    never trigger the retry: a broken mesh is a setup bug, not stall
    physics. The status records application + pseudo_transient so the gate
    report can flag these points.
    """
    # Idempotency: a recorded convergence PASS short-circuits the whole
    # chain unless the operator forces a re-solve.
    prior = read_status(case_dir)
    if prior and prior.get("converged") and not force:
        prior["skipped"] = True
        return prior

    status: dict = {"alpha": alpha, "case": str(case_dir),
                    "converged": False, "skipped": False}
    n_cores = decompose_cores(case_dir, cores)
    # The case's own controlDict names the solver (a case already switched
    # to the pimple variant - e.g. by a previous retry - stays switched).
    app = read_case_application(case_dir) or "simpleFoam"
    status["application"] = app

    # ---- steady attempt (mesh + solve + gate) -------------------------------
    failed = run_solver_chain(mode, case_dir, n_cores, app, with_mesh=True)
    solver_failed = failed is not None and failed == app
    if failed is not None and not solver_failed:
        # Mesh/decompose/reconstruct failure: setup problem, never retried.
        status["failed_step"] = failed
        status["log"] = str(case_dir / f"log.{failed}")
        write_status(case_dir, status)
        return status
    if not solver_failed:
        # Convergence gate: parse_forces owns the residual + force-flatness
        # criteria; anything unparseable stays NOT converged (auto-flagged).
        means = parse_forces_means(case_dir)
        if means is not None:
            status.update({k: means[k] for k in ("cl", "cd", "cm")
                           if k in means})
            status["converged"] = bool(means.get("converged", False))

    # ---- post-stall pimpleFoam pseudo-transient retry ------------------------
    #  One retry, only for post-stall alphas still configured steady; the
    #  variant swap is mechanical (full dictionaries from pimple_overrides/),
    #  never a hand edit, so the rerun stays reproducible.
    if (not status["converged"] and app == "simpleFoam"
            and alpha >= pimple_alpha):
        print(f"    [pimple] alpha {alpha:+.1f} deg did not converge steady "
              f"- retrying with the pimpleFoam localEuler variant")
        try:
            apply_pimple_variant(case_dir)
        except FileNotFoundError as exc:
            print(f"    [pimple] variant unavailable: {exc}")
        else:
            app = "pimpleFoam"
            status["application"] = app
            status["pseudo_transient"] = True
            status.pop("failed_step", None)
            status.pop("log", None)
            failed = run_solver_chain(mode, case_dir, n_cores, app,
                                      with_mesh=False)
            if failed is None:
                # Re-gate on the same force-flatness criterion; residual
                # levels under local time stepping are recorded by the case
                # but are not comparable to the steady thresholds.
                means = parse_forces_means(case_dir)
                if means is not None:
                    status.update({k: means[k] for k in ("cl", "cd", "cm")
                                   if k in means})
                    status["converged"] = bool(means.get("converged", False))

    # Record whichever step is still failing after any retry (a retry that
    # converged cleared ``failed``); absence of failed_step with converged
    # False means the chain ran but the convergence gate said no.
    if failed is not None:
        status["failed_step"] = failed
        status["log"] = str(case_dir / f"log.{failed}")

    # ---- spec section-4 wall-y+ gate (parse_forces gate_case_yplus) ----------
    #  Runs only on solutions that already passed the convergence gate, and
    #  deliberately AFTER the pimple retry: a bad y+ distribution is a MESH
    #  property, so re-solving with another time scheme can neither fix it
    #  nor should be triggered by it. The spec requires the per-case y+
    #  histogram to be printed and the case to FAIL when y+ > 2 on more than
    #  1% of wall faces; an INDETERMINATE verdict (no usable y+ evidence)
    #  also fails - auto-flagged, never silently included.
    if status["converged"]:
        yp = case_yplus_verdict(case_dir)
        if yp is None:
            # parse_forces without the y+ entry point: the gate cannot run,
            # and a spec gate that cannot run must flag, not wave through.
            print(f"    [yplus] gate unavailable (parse_forces lacks "
                  f"gate_case_yplus) - case flagged")
            status["converged"] = False
            status["failed_step"] = "yplus"
        else:
            # The spec-required histogram, indented to the driver log style.
            for line in yp.report().splitlines():
                print(f"    {line}")
            # Marker fields for the audit trail (NaN stats are omitted so
            # the JSON stays strictly portable).
            status["yplus_status"] = yp.status
            status["yplus_passed"] = bool(yp.passed)
            if math.isfinite(yp.y_max):
                status["yplus_max"] = float(yp.y_max)
            if math.isfinite(yp.fraction_above):
                status["yplus_fraction_above"] = float(yp.fraction_above)
            if not yp.passed:
                status["converged"] = False
                status["failed_step"] = "yplus"

    # BL extraction + Cf conversion + transition pickup only on converged
    # solutions - a non-converged Cf field would feed garbage into gate (c).
    if status["converged"]:
        xtr = extract_transition(case_dir)
        if xtr is not None:
            status["xtr_top"] = xtr
    write_status(case_dir, status)
    return status


def solve_all(mode: str, cases: List[Tuple[float, Path]], jobs: int,
              cores: int, force: bool,
              pimple_alpha: float = PIMPLE_ALPHA_DEFAULT) -> List[dict]:
    """Run the per-case chain across the sweep, ``jobs`` cases in flight.

    Two-wide by default: 2 cases x 8 cores leaves headroom on the 24C/32T
    host for the OS and the post-processing subprocesses, and matches the
    spec's queue-management guidance (sequential or 2-wide, configurable).
    """
    results: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = {ex.submit(solve_case, mode, a, d, cores, force,
                             pimple_alpha): (a, d)
                   for a, d in cases}
        for fut in concurrent.futures.as_completed(futures):
            alpha, case_dir = futures[fut]
            status = fut.result()
            results.append(status)
            # Live one-line progress per finished case (ASCII only); points
            # solved by the fallback are tagged so the console mirrors the
            # gate report's pseudo-transient annotation.
            verdict = ("SKIP (already converged)" if status.get("skipped")
                       else "CONVERGED" if status.get("converged")
                       else f"FAILED ({status.get('failed_step', 'gate')})")
            if status.get("pseudo_transient") and not status.get("skipped"):
                verdict += " [pimpleFoam pseudo-transient]"
            print(f"  [solve] alpha {alpha:+05.1f} deg -> {verdict}")
    results.sort(key=lambda s: s.get("alpha", 0.0))
    return results


# =============================================================================
#  Aggregation: per-case markers -> polar + transition CSVs for the gate
# =============================================================================
def aggregate_results(cases: List[Tuple[float, Path]], re_value: float,
                      level: int) -> Tuple[Optional[Path], Optional[Path]]:
    """Collect converged-case markers into the gate's input CSVs.

    Rebuilt from the markers on every run (cheap, and guarantees the polar
    always reflects the union of everything solved so far, across multiple
    partial invocations). Returns (polar_csv, transition_csv); either is
    None when no converged data exists for it yet.
    """
    polar_rows: List[Tuple[float, float, float, float]] = []
    tr_rows: List[Tuple[float, float]] = []
    for alpha, case_dir in cases:
        status = read_status(case_dir)
        if not status or not status.get("converged"):
            continue
        if all(k in status for k in ("cl", "cd", "cm")):
            polar_rows.append((alpha, status["cl"], status["cd"],
                               status["cm"]))
        if "xtr_top" in status:
            tr_rows.append((alpha, status["xtr_top"]))

    tag = re_tag(re_value)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    polar_path = trans_path = None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if polar_rows:
        # Polar CSV consumed by compare_gate (alpha,cl,cd,cm contract).
        polar_path = RESULTS_DIR / f"openfoam_polar_Re{tag}_L{level}.csv"
        with open(polar_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# OpenFOAM clean-section polar, Re {tag}, mesh level "
                     f"{level}\n# aggregated {stamp} UTC by "
                     f"scripts/run_validation.py from per-case "
                     f"{STATUS_NAME} markers\nalpha,cl,cd,cm\n")
            for a, cl, cd, cm in sorted(polar_rows):
                fh.write(f"{a:.2f},{cl:.6f},{cd:.6f},{cm:.6f}\n")
        print(f"  [aggregate] polar -> {polar_path.name} "
              f"({len(polar_rows)} converged case(s))")

    if tr_rows:
        # Transition table for gate (c) (alpha,xtr_top contract).
        trans_path = RESULTS_DIR / f"transition_Re{tag}_L{level}.csv"
        with open(trans_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# Upper-surface transition x/c (Cf post-minimum "
                     f"inflection detector), Re {tag}, level {level}\n"
                     f"# aggregated {stamp} UTC by scripts/run_validation.py"
                     f"\nalpha,xtr_top\n")
            for a, x in sorted(tr_rows):
                fh.write(f"{a:.2f},{x:.4f}\n")
        print(f"  [aggregate] transition -> {trans_path.name} "
              f"({len(tr_rows)} case(s))")
    return polar_path, trans_path


# =============================================================================
#  Driver
# =============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1 one-command Phase-1 validation driver")
    parser.add_argument("--re", type=float, default=3.0e6,
                        help="chord Reynolds number (default 3e6)")
    parser.add_argument("--level", type=int, default=0,
                        help="mesh refinement level (default 0)")
    parser.add_argument("--ncrit", type=int, default=9,
                        help="XFOIL Ncrit to gate against (default 9)")
    parser.add_argument("--jobs", type=int, default=2,
                        help="cases solved concurrently (default 2)")
    parser.add_argument("--cores", type=int, default=8,
                        help="MPI ranks per case when the case's "
                             "decomposeParDict lacks a count (default 8)")
    parser.add_argument("--alphas", type=str, default=None,
                        help="comma-separated AoA override in degrees "
                             "(default: the Phase-1 schedule)")
    parser.add_argument("--pimple-alpha", type=float,
                        default=PIMPLE_ALPHA_DEFAULT,
                        help="AoA threshold (deg) above which a non-"
                             "converged steady case is retried once with "
                             "the pimpleFoam pseudo-transient variant "
                             f"(default {PIMPLE_ALPHA_DEFAULT:g}; set very "
                             "high, e.g. 999, to disable the retry)")
    parser.add_argument("--no-solve", action="store_true",
                        help="stop after case generation even if a solver "
                             "is available (dry-run)")
    parser.add_argument("--force", action="store_true",
                        help="re-solve cases whose marker already records a "
                             "convergence PASS")
    parser.add_argument("--skip-xfoil", action="store_true",
                        help="do not attempt XFOIL polar generation")
    args = parser.parse_args(argv)

    alphas = ([float(t) for t in args.alphas.split(",")]
              if args.alphas else alpha_schedule())
    tag = re_tag(args.re)

    # Freestream context for the operator: 2D convention is unit chord with
    # U = Re * nu / c (nu from aircraft.yaml). The Mach consequence of that
    # convention (notably at Re 6e6) is documented in compare_gate, the
    # report, and the README M1 section.
    ac = load_aircraft(YAML_PATH)
    nu = ac.atmosphere.kinematic_viscosity
    u_inf = args.re * nu / 1.0
    a_sound = math.sqrt(1.4 * 287.05 * ac.atmosphere.temperature)

    print("=" * 78)
    print(f"M1 Phase-1 validation driver - Re {tag}, level {args.level}, "
          f"{len(alphas)} AoA point(s)")
    print(f"  U_inf = Re*nu/c = {u_inf:.2f} m/s at c = 1.0 m "
          f"(Mach {u_inf / a_sound:.3f})")
    print("=" * 78)

    # ---- (a) XFOIL baseline --------------------------------------------------
    print("[1/4] XFOIL baseline polars")
    ensure_xfoil_polars(args.re, args.ncrit, args.skip_xfoil)

    # ---- (b) case generation ---------------------------------------------------
    print("[2/4] case generation")
    cases, build_failed = build_cases(alphas, args.re, args.level)
    print(f"  [build] {len(cases)} case(s) ready, "
          f"{len(build_failed)} failed/missing")

    # ---- (c) solve (or dry-run notice) ----------------------------------------
    print("[3/4] solver")
    mode = None if args.no_solve else probe_solver()
    solved: List[dict] = []
    if mode is None:
        # Current expected state until the WSL repair + openfoam2506 install
        # lands: report readiness loudly, continue to the gate, exit 0.
        reason = ("--no-solve requested" if args.no_solve else
                  "simpleFoam not on PATH and not reachable via "
                  "'wsl -e bash' (openfoam2506 bashrc)")
        print(f"  SOLVER MISSING - generated {len(cases)} case(s) ready "
              f"under {CASE_ROOT}")
        print(f"  ({reason}; re-run this command once OpenFOAM v2506 is "
              f"installed in WSL2 Ubuntu)")
    elif not cases:
        print("  solver available but no cases to run (see build step above)")
    else:
        print(f"  solver mode: {mode}; running {args.jobs} case(s) wide, "
              f"{args.cores} core(s) each (decomposeParDict wins); "
              f"pimple retry at alpha >= {args.pimple_alpha:g} deg")
        solved = solve_all(mode, cases, args.jobs, args.cores, args.force,
                           args.pimple_alpha)
        n_conv = sum(1 for s in solved if s.get("converged"))
        n_skip = sum(1 for s in solved if s.get("skipped"))
        print(f"  [solve] {n_conv}/{len(solved)} converged "
              f"({n_skip} prior PASS skipped, "
              f"{len(solved) - n_conv} flagged)")

    # ---- (d) gate evaluation ---------------------------------------------------
    #  Aggregate whatever converged data exists (possibly none) and hand it
    #  to the gate; compare_gate marks every missing input SKIPPED with the
    #  reason, so this call is always safe.
    print("[4/4] gate evaluation (validation/compare_gate.py)")
    aggregate_results(cases, args.re, args.level)
    gate_rc = compare_gate_main([
        "--re", f"{args.re:g}", "--level", str(args.level),
        "--ncrit", str(args.ncrit),
    ])

    # Exit policy: gate FAIL propagates (rc 1); the solver-missing dry-run
    # and SKIPPED-only reports exit 0 so the command is green today and
    # becomes the real acceptance test the moment the solver lands.
    return gate_rc


if __name__ == "__main__":
    sys.exit(main())
