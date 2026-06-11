"""Tests for validation/compare_gate.py - the Phase-1 gate evaluator (M1).

Everything here runs on synthetic inputs (no solver, no XFOIL binary): the
gates are pure numerics over small tables, so fabricated polars with known
slopes/peaks and a fabricated wall-Cf curve with a known transition location
exercise every verdict path - PASS, FAIL, and the SKIPPED downgrades that
fire while real inputs (NASA digitized CSVs, solver results) do not exist.

Threshold expectations mirror the spec Phase-1 item-4 numbers through the
module constants (SLOPE_TOL_REL etc.) so a deliberate threshold change only
needs editing in one place.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from validation.compare_gate import (
    CLMAX_TOL_REL,
    REPO,
    SLOPE_TOL_REL,
    STALL_TOL_DEG,
    XTR_TOL_XC,
    GateResult,
    clmax_and_stall,
    detect_transition_from_cf,
    evaluate_gates,
    find_nasa_polar,
    find_xfoil_polar,
    fit_cl_alpha_slope,
    gate_clmax_stall,
    gate_slope,
    gate_transition,
    main as gate_main,
    pseudo_transient_cases,
    read_case_application,
    read_csv_columns,
    re_tag,
    upsert_report,
)
from scripts import extract_cf, run_validation

# =============================================================================
#  Synthetic-input builders
# =============================================================================
def make_polar(alphas, slope=0.110, cl0=0.40, clmax=None, stall_alpha=None):
    """Fabricate a polar dict with a linear region and an optional stall peak.

    Linear part: cl = cl0 + slope * alpha. When (clmax, stall_alpha) are
    given, the curve is capped by a parabola peaking exactly there, so both
    the slope fit (over [-2, 6] deg) and the Clmax/stall pickup see the
    intended numbers.
    """
    a = np.asarray(alphas, dtype=float)
    cl = cl0 + slope * a
    if clmax is not None and stall_alpha is not None:
        # Quadratic cap: peak value clmax at stall_alpha. Curvature is kept
        # gentle (0.004/deg^2) so the cap stays ABOVE the linear line across
        # the [-2, 6] deg fit window for every (clmax, stall) combination
        # used below - the cap then only binds near the peak and the slope
        # fit sees the pure linear region.
        cap = clmax - 0.004 * (a - stall_alpha) ** 2
        cl = np.minimum(cl, cap)
    return {"alpha": a, "cl": cl,
            "cd": np.full_like(a, 0.008), "cm": np.full_like(a, -0.09)}


def write_polar_csv(path: Path, polar: dict, extra_cols=None) -> None:
    """Write a polar dict as the CSV contract compare_gate ingests."""
    cols = ["alpha", "cl", "cd", "cm"] + list(extra_cols or [])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# synthetic test polar\n")
        fh.write(",".join(cols) + "\n")
        for i in range(len(polar["alpha"])):
            fh.write(",".join(f"{polar[c][i]:.6f}" for c in cols) + "\n")


def make_cf_curve(xtr=0.45, re_value=3.0e6, n=300, width=0.008):
    """Fabricated upper-surface Cf(x/c) with a known transition location.

    Laminar Blasius decay blended into a turbulent power-law level through a
    logistic ramp centered at ``xtr``; the logistic inflection (steepest
    rise) sits exactly at the center, which is what the detector defines as
    the transition point.
    """
    x = np.linspace(0.01, 1.0, n)
    cf_lam = 0.664 / np.sqrt(re_value * x)          # Blasius flat plate
    cf_turb = 0.0592 / (re_value * x) ** 0.2        # 1/5th-power turbulent
    w = 1.0 / (1.0 + np.exp(-(x - xtr) / width))    # logistic blend
    return x, cf_lam * (1.0 - w) + cf_turb * w


# =============================================================================
#  CSV reader: aliases, comments, ragged rows
# =============================================================================
def test_read_csv_columns_aliases_and_comments(tmp_path: Path) -> None:
    """Header aliases canonicalize and '#' comments are ignored anywhere."""
    p = tmp_path / "polar.csv"
    p.write_text(
        "# produced by some other tool\n"
        "AoA, CL, CD, CM, Top_Xtr\n"
        "0.0, 0.40, 0.008, -0.09, 0.55\n"
        "# mid-file comment\n"
        "2.0, 0.62, 0.008, -0.09, 0.50\n",
        encoding="utf-8")
    table = read_csv_columns(p)
    # Aliased headers arrive under their canonical names.
    assert set(table) >= {"alpha", "cl", "cd", "cm", "xtr_top"}
    assert table["alpha"] == pytest.approx([0.0, 2.0])
    assert table["xtr_top"] == pytest.approx([0.55, 0.50])


def test_read_csv_columns_bad_cells_become_nan(tmp_path: Path) -> None:
    """Non-numeric cells turn into NaN instead of crashing the gate."""
    p = tmp_path / "polar.csv"
    p.write_text("alpha,cl\n0.0,0.4\n2.0,-\n", encoding="utf-8")
    table = read_csv_columns(p)
    assert math.isnan(table["cl"][1])
    assert table["alpha"][1] == pytest.approx(2.0)


# =============================================================================
#  Slope fit and gate (a): pass / fail / skip
# =============================================================================
def test_fit_slope_recovers_known_slope() -> None:
    """The fit reproduces a fabricated linear slope over [-2, 6] deg only."""
    a = np.arange(-4.0, 21.0, 1.0)
    # Slope 0.110/deg in the window; the quadratic cap outside must not leak
    # into the fit because the window excludes it.
    polar = make_polar(a, slope=0.110, clmax=1.60, stall_alpha=16.0)
    fit = fit_cl_alpha_slope(polar["alpha"], polar["cl"])
    assert fit is not None
    slope, n = fit
    assert slope == pytest.approx(0.110, rel=1e-6)
    assert n == 9                                   # -2..6 inclusive, 1-deg


def test_fit_slope_insufficient_points_returns_none() -> None:
    """Fewer than 3 in-window samples -> None (caller downgrades to SKIP)."""
    assert fit_cl_alpha_slope(np.array([10.0, 12.0]),
                              np.array([1.0, 1.1])) is None


def test_gate_slope_pass() -> None:
    """Slopes 2 % apart pass the 5 % criterion with the numbers recorded."""
    a = np.arange(-4.0, 9.0, 1.0)
    of = make_polar(a, slope=0.1078)                # 2 % below reference
    xf = make_polar(a, slope=0.110)
    r = gate_slope(of, xf)
    assert r.status == "PASS"
    assert r.measured == pytest.approx(0.1078, rel=1e-6)
    assert r.reference == pytest.approx(0.110, rel=1e-6)
    assert r.delta == pytest.approx(0.02, abs=1e-6)
    assert r.delta <= SLOPE_TOL_REL


def test_gate_slope_fail() -> None:
    """Slopes 13.6 % apart fail the 5 % criterion."""
    a = np.arange(-4.0, 9.0, 1.0)
    r = gate_slope(make_polar(a, slope=0.095), make_polar(a, slope=0.110))
    assert r.status == "FAIL"
    assert r.delta is not None and r.delta > SLOPE_TOL_REL


def test_gate_slope_skipped_without_inputs() -> None:
    """Missing either polar downgrades to SKIPPED with a reason, never PASS."""
    a = np.arange(-4.0, 9.0, 1.0)
    polar = make_polar(a)
    assert gate_slope(None, polar).status == "SKIPPED"
    assert gate_slope(polar, None).status == "SKIPPED"
    assert gate_slope(None, None).detail != ""


# =============================================================================
#  Gate (b): Clmax / stall AoA vs fabricated NASA rows
# =============================================================================
def test_clmax_pickup() -> None:
    """Clmax/stall extraction lands on the fabricated peak."""
    a = np.arange(-4.0, 21.0, 1.0)
    polar = make_polar(a, clmax=1.62, stall_alpha=16.0)
    peak = clmax_and_stall(polar["alpha"], polar["cl"])
    assert peak is not None
    clmax, stall = peak
    assert clmax == pytest.approx(1.62, abs=1e-6)
    assert stall == pytest.approx(16.0, abs=1e-6)


def test_gate_clmax_stall_pass() -> None:
    """RANS 4.3 % low on Clmax and 1 deg early on stall: both rows PASS."""
    a = np.arange(-4.0, 21.0, 1.0)
    of = make_polar(a, clmax=1.55, stall_alpha=15.0)
    nasa = make_polar(a, clmax=1.62, stall_alpha=16.0)
    row_cl, row_aa = gate_clmax_stall(of, nasa)
    assert row_cl.gate == "clmax" and row_cl.status == "PASS"
    assert row_cl.delta == pytest.approx(abs(1.55 - 1.62) / 1.62, rel=1e-6)
    assert row_cl.delta <= CLMAX_TOL_REL
    assert row_aa.gate == "stall_aoa" and row_aa.status == "PASS"
    assert row_aa.delta == pytest.approx(1.0, abs=1e-9)
    assert row_aa.delta <= STALL_TOL_DEG


def test_gate_clmax_stall_fail() -> None:
    """RANS 20 % low and 4 deg early: both rows FAIL with the deltas."""
    a = np.arange(-4.0, 21.0, 1.0)
    of = make_polar(a, slope=0.105, clmax=1.30, stall_alpha=12.0)
    nasa = make_polar(a, clmax=1.62, stall_alpha=16.0)
    row_cl, row_aa = gate_clmax_stall(of, nasa)
    assert row_cl.status == "FAIL" and row_cl.delta > CLMAX_TOL_REL
    assert row_aa.status == "FAIL" and row_aa.delta > STALL_TOL_DEG


def test_gate_clmax_skipped_when_nasa_absent() -> None:
    """No NASA table -> both rows SKIPPED carrying the discovery note."""
    a = np.arange(-4.0, 21.0, 1.0)
    rows = gate_clmax_stall(make_polar(a), None,
                            nasa_note="digitized directory missing")
    assert [r.status for r in rows] == ["SKIPPED", "SKIPPED"]
    assert all("digitized directory missing" in r.detail for r in rows)


def test_find_nasa_polar_absent_directory(tmp_path: Path) -> None:
    """The NASA discovery path reports absence instead of raising."""
    path, note = find_nasa_polar(tmp_path / "does_not_exist", 3.0e6)
    assert path is None
    assert "not present" in note


def test_find_nasa_polar_nearest_re_fallback(tmp_path: Path) -> None:
    """No exact Re match -> the nearest tunnel curve anchors, loudly.

    Mirrors the real digitized set (TM X-72843 ran 2.2/4.3/6.4e6, the gate
    defaults to 3e6): the 2.2e6 curve is nearest in log-Re and must be
    selected with a WARNING note; a request further than 2x from every
    curve must be refused instead of stretching the comparison.
    """
    nasa = tmp_path / "digitized"
    nasa.mkdir()
    for tag in ("Re2.2e6", "Re4.3e6", "Re6.4e6"):
        (nasa / f"nasa_fig5_{tag}_trip.csv").write_text(
            "alpha,cl\n0.0,0.45\n", encoding="utf-8")
    path, note = find_nasa_polar(nasa, 3.0e6)
    assert path is not None and "Re2.2e6" in path.name
    assert note.startswith("WARNING") and "nearest" in note
    # Beyond the 2x bound nothing qualifies: refusal, not a stretch.
    path, note = find_nasa_polar(nasa, 0.5e6)
    assert path is None
    assert "within 2x" in note


# =============================================================================
#  Transition detector (gate c numerics)
# =============================================================================
def test_transition_detector_finds_known_location() -> None:
    """The detector lands on the logistic-ramp center of a fabricated Cf."""
    x, cf = make_cf_curve(xtr=0.45)
    found = detect_transition_from_cf(x, cf)
    assert found is not None
    assert abs(found - 0.45) <= 0.02


def test_transition_detector_other_location() -> None:
    """Same machinery, different fabricated transition point (0.25c)."""
    x, cf = make_cf_curve(xtr=0.25)
    found = detect_transition_from_cf(x, cf)
    assert found is not None
    assert abs(found - 0.25) <= 0.02


def test_transition_detector_fully_laminar_returns_none() -> None:
    """A monotonically decaying laminar Cf has no transition signature."""
    x = np.linspace(0.01, 1.0, 200)
    cf = 0.664 / np.sqrt(3.0e6 * x)
    assert detect_transition_from_cf(x, cf) is None


def test_transition_detector_ignores_le_spike() -> None:
    """A stagnation-region Cf spike below the x/c cutoff cannot win."""
    x, cf = make_cf_curve(xtr=0.45)
    # Poison the first samples (x < 0.02) with a huge spike; the cutoff
    # must discard them before any min/derivative logic runs.
    cf = cf.copy()
    cf[x < 0.02] = 0.05
    found = detect_transition_from_cf(x, cf)
    assert found is not None and abs(found - 0.45) <= 0.02


def test_transition_detector_too_few_points() -> None:
    """Tiny inputs are refused (None), not extrapolated."""
    assert detect_transition_from_cf([0.1, 0.2, 0.3], [3e-3, 2e-3, 4e-3]) is None


# =============================================================================
#  Gate (c): transition comparison vs XFOIL xtr_top
# =============================================================================
def _xfoil_with_xtr(alphas, xtrs):
    """XFOIL-style polar dict carrying the e^N xtr_top column."""
    polar = make_polar(np.asarray(alphas, dtype=float))
    polar["xtr_top"] = np.asarray(xtrs, dtype=float)
    return polar


def test_gate_transition_pass() -> None:
    """Worst pair 0.05c apart passes the 0.10c criterion."""
    xf = _xfoil_with_xtr([0.0, 2.0, 4.0, 6.0], [0.55, 0.48, 0.41, 0.34])
    rans = {"alpha": np.array([0.0, 2.0, 4.0, 6.0]),
            "xtr_top": np.array([0.50, 0.45, 0.40, 0.30])}
    r = gate_transition(rans, xf)
    assert r.status == "PASS"
    assert r.delta == pytest.approx(0.05, abs=1e-9)
    assert r.delta <= XTR_TOL_XC


def test_gate_transition_fail() -> None:
    """Worst pair 0.24c apart fails, and the worst pair drives the verdict."""
    xf = _xfoil_with_xtr([0.0, 2.0, 4.0, 6.0], [0.55, 0.48, 0.41, 0.34])
    rans = {"alpha": np.array([0.0, 2.0, 4.0, 6.0]),
            "xtr_top": np.array([0.40, 0.30, 0.20, 0.10])}
    r = gate_transition(rans, xf)
    assert r.status == "FAIL"
    assert r.delta == pytest.approx(0.24, abs=1e-9)


def test_gate_transition_skipped_paths() -> None:
    """Absent RANS table or missing xtr_top column -> SKIPPED."""
    xf = _xfoil_with_xtr([0.0, 2.0], [0.55, 0.48])
    assert gate_transition(None, xf).status == "SKIPPED"
    # XFOIL polar without the xtr_top column cannot anchor the gate.
    no_xtr = make_polar(np.array([0.0, 2.0]))
    rans = {"alpha": np.array([0.0]), "xtr_top": np.array([0.5])}
    assert gate_transition(rans, no_xtr).status == "SKIPPED"


def test_gate_transition_post_stall_rows_excluded() -> None:
    """Rows beyond the pre-stall window do not poison the comparison."""
    xf = _xfoil_with_xtr([0.0, 2.0, 14.0], [0.55, 0.48, 0.05])
    # The alpha-14 RANS row is wildly off but sits past XTR_ALPHA_MAX, so
    # only the two pre-stall pairs (both within 0.05c) are gated.
    rans = {"alpha": np.array([0.0, 2.0, 14.0]),
            "xtr_top": np.array([0.50, 0.45, 0.60])}
    r = gate_transition(rans, xf)
    assert r.status == "PASS"


# =============================================================================
#  Polar-file discovery (filename metadata)
# =============================================================================
def test_find_xfoil_polar_matches_re_and_ncrit(tmp_path: Path) -> None:
    """Re token + Ncrit preference select the right file among several."""
    a = np.arange(-2.0, 7.0, 1.0)
    for name in ("polar_Re1.5e6_N9_free.csv", "polar_Re3e6_N9_free.csv",
                 "polar_Re3e6_N9_tripped.csv", "polar_Re6e6_N9_free.csv"):
        write_polar_csv(tmp_path / name, make_polar(a))
    path, note = find_xfoil_polar(tmp_path, 3.0e6, ncrit=9)
    assert path is not None and path.name == "polar_Re3e6_N9_free.csv"
    assert "matched" in note


def test_find_xfoil_polar_absent(tmp_path: Path) -> None:
    """Empty/missing directory reports a reason instead of raising."""
    path, note = find_xfoil_polar(tmp_path / "nope", 3.0e6)
    assert path is None and "not present" in note


def test_find_xfoil_polar_legacy_names_still_resolve(tmp_path: Path) -> None:
    """Pre-Ncrit filenames (no N token) resolve with a loud assumption note.

    Backward compatibility for archives written before the naming gained
    the N token: polar_Re3M_free.csv must keep matching a Re-3M request,
    but never silently - the note carries the assumed Ncrit.
    """
    a = np.arange(-2.0, 7.0, 1.0)
    write_polar_csv(tmp_path / "polar_Re3M_free.csv", make_polar(a))
    write_polar_csv(tmp_path / "polar_Re3M_trip.csv", make_polar(a))
    path, note = find_xfoil_polar(tmp_path, 3.0e6, ncrit=9)
    assert path is not None and path.name == "polar_Re3M_free.csv"
    assert note.startswith("WARNING") and "Ncrit" in note


def test_find_xfoil_polar_prefers_ncrit_token_over_legacy(tmp_path: Path
                                                          ) -> None:
    """When both spellings coexist, the exact Ncrit-tagged file wins."""
    a = np.arange(-2.0, 7.0, 1.0)
    write_polar_csv(tmp_path / "polar_Re3M_free.csv", make_polar(a))
    write_polar_csv(tmp_path / "polar_Re3M_N9_free.csv", make_polar(a))
    path, note = find_xfoil_polar(tmp_path, 3.0e6, ncrit=9)
    assert path is not None and path.name == "polar_Re3M_N9_free.csv"
    assert note == "matched polar_Re3M_N9_free.csv"


def test_repo_polars_resolve_with_n_token() -> None:
    """The committed validation/xfoil/ polars carry the N9 token and match.

    Guards the on-disk rename: a free-transition Re-3M request against the
    real directory must land on the renamed polar_Re3M_N9_free.csv.
    """
    path, note = find_xfoil_polar(REPO / "validation" / "xfoil", 3.0e6,
                                  ncrit=9)
    assert path is not None and path.name == "polar_Re3M_N9_free.csv"
    assert note == "matched polar_Re3M_N9_free.csv"


# =============================================================================
#  evaluate_gates: full SKIPPED sweep with nothing available
# =============================================================================
def test_evaluate_gates_all_skipped_when_nothing_exists() -> None:
    """With no inputs at all, every gate is SKIPPED - nothing passes silently."""
    results = evaluate_gates(None, None, None, None,
                             nasa_note="not digitized yet")
    assert len(results) == 4
    assert all(r.status == "SKIPPED" for r in results)
    assert all(r.detail for r in results)          # every skip has a reason


# =============================================================================
#  Report writing: upsert semantics + end-to-end main()
# =============================================================================
def test_upsert_report_appends_then_updates(tmp_path: Path) -> None:
    """Same key updates in place; a new key appends a second block."""
    report = tmp_path / "report.md"
    upsert_report(report, "re=3e6 level=0 ncrit=9", "## block one v1")
    upsert_report(report, "re=3e6 level=0 ncrit=9", "## block one v2")
    upsert_report(report, "re=6e6 level=0 ncrit=9", "## block two v1")
    text = report.read_text(encoding="utf-8")
    # The v1 content of the first key was replaced, not duplicated.
    assert "block one v2" in text and "block one v1" not in text
    assert "block two v1" in text
    assert text.count("re=3e6 level=0 ncrit=9 BEGIN") == 1


def test_main_end_to_end_skipped_run(tmp_path: Path) -> None:
    """CLI run with synthetic XFOIL + OF polars, NASA absent.

    Expectations: exit 0 (SKIPPED never fails the process), report written
    with the pinned OpenFOAM version from the aircraft.yaml solver block
    (v2506 as of schema v2; the UNPINNED warning fires only when that entry
    is null), slope gate PASS on matching synthetic slopes, NASA gates
    SKIPPED, transition SKIPPED (no RANS table).
    """
    a = np.arange(-4.0, 21.0, 1.0)
    xfoil_dir = tmp_path / "xfoil"
    xfoil_dir.mkdir()
    polar = make_polar(a, slope=0.110, clmax=1.60, stall_alpha=16.0)
    polar["xtr_top"] = np.clip(0.55 - 0.03 * a, 0.02, 0.60)
    write_polar_csv(xfoil_dir / "polar_Re3e6_N9_free.csv", polar,
                    extra_cols=["xtr_top"])
    of_csv = tmp_path / "openfoam_polar.csv"
    write_polar_csv(of_csv, make_polar(a, slope=0.108, clmax=1.55,
                                       stall_alpha=15.0))
    report = tmp_path / "report.md"

    rc = gate_main([
        "--re", "3e6", "--level", "0", "--ncrit", "9",
        "--of-csv", str(of_csv),
        "--xfoil-dir", str(xfoil_dir),
        "--nasa-dir", str(tmp_path / "nasa_digitized"),   # absent on purpose
        "--transition-csv", str(tmp_path / "transition_missing.csv"),
        "--case-root", str(tmp_path / "no_cases"),        # absent on purpose
        "--report", str(report),
    ])
    assert rc == 0
    text = report.read_text(encoding="utf-8")
    # aircraft.yaml pins solver.openfoam_version (v2506): the config table must
    # carry the pinned version and the UNPINNED warning must be gone.
    assert "v2506" in text
    assert "UNPINNED" not in text
    # Verdict spot-checks: slope passed, NASA-anchored gates skipped.
    assert "cl_alpha_slope | **PASS**" in text
    assert "clmax | **SKIPPED**" in text
    assert "stall_aoa | **SKIPPED**" in text
    assert "transition_xtr | **SKIPPED**" in text


def test_main_fail_exit_code(tmp_path: Path) -> None:
    """A failing slope gate propagates exit code 1 from the CLI."""
    a = np.arange(-4.0, 9.0, 1.0)
    xfoil_dir = tmp_path / "xfoil"
    xfoil_dir.mkdir()
    write_polar_csv(xfoil_dir / "polar_Re3e6_N9_free.csv",
                    make_polar(a, slope=0.110))
    of_csv = tmp_path / "of.csv"
    write_polar_csv(of_csv, make_polar(a, slope=0.085))   # 23 % off: FAIL
    rc = gate_main([
        "--re", "3e6", "--of-csv", str(of_csv),
        "--xfoil-dir", str(xfoil_dir),
        "--nasa-dir", str(tmp_path / "absent"),
        "--transition-csv", str(tmp_path / "absent.csv"),
        "--case-root", str(tmp_path / "no_cases"),
        "--report", str(tmp_path / "report.md"),
    ])
    assert rc == 1


# =============================================================================
#  extract_cf: wallShearStress raw dump -> suction-surface Cf(x/c) producer
# =============================================================================
#  The producer is exercised against a synthetic fixture in the documented
#  raw surface-writer format ('#' header lines, then x y z tau_x tau_y tau_z
#  per face centre) since OpenFOAM is absent on this machine. Both verdict
#  directions are covered: a planted laminar-turbulent Cf signature the
#  detector must find, and a fully-laminar curve it must refuse.

# In-plane unit vector for the fabricated wall-shear direction (0.96^2 +
# 0.28^2 = 1 exactly), so |tau| equals the planted magnitude bit-for-bit.
_TAU_DIR = (0.96, 0.28, 0.0)


def _cf_profile(x, xtr, re_value=3.0e6, width=0.008):
    """Planted Cf(x): Blasius decay -> turbulent level via a logistic ramp.

    Same construction as make_cf_curve above but evaluated at CALLER x
    stations, so the fabricated wall-shear dump and the expected curve share
    one set of face-centre abscissae. xtr far beyond the TE (e.g. 10) makes
    the curve fully laminar - the detector's None direction.
    """
    x = np.asarray(x, dtype=float)
    cf_lam = 0.664 / np.sqrt(re_value * x)
    cf_turb = 0.0592 / (re_value * x) ** 0.2
    # Clip the logistic exponent: the fully-laminar plant (xtr = 10) would
    # otherwise overflow exp() harmlessly but noisily (w -> 0 either way).
    z = np.clip(-(x - xtr) / width, -700.0, 700.0)
    w = 1.0 / (1.0 + np.exp(z))
    return cf_lam * (1.0 - w) + cf_turb * w


def make_wall_shear_case(case_dir: Path, alpha_deg: float = 4.0,
                         u_inf: float = 43.8, xtr_upper=0.45, xtr_lower=None,
                         time_name: str = "3000") -> Path:
    """Fabricate a case directory in the wallCf function-object contract.

    Writes a minimal 0/U (the builder's 'internalField uniform (..)' line,
    AoA-rotated) plus postProcessing/wallCf/<time>/ holding a raw surface
    dump: 120 upper + 120 lower face centres on an airfoil-ish thickness
    envelope, with KINEMATIC wall shear tau = Cf * 0.5 * U^2 per the planted
    profiles (None = fully laminar side), and 3 blunt-base faces on the
    x = max plane that the extractor must drop. Returns the dump path.
    """
    a_rad = math.radians(alpha_deg)
    u_dir = case_dir / "0"
    u_dir.mkdir(parents=True, exist_ok=True)
    (u_dir / "U").write_text(
        "// fabricated freestream for extract_cf tests\n"
        f"internalField   uniform ({u_inf * math.cos(a_rad):.6f} "
        f"{u_inf * math.sin(a_rad):.6f} 0);\n",
        encoding="utf-8")

    # Face-centre stations + half-thickness envelope; the TE stops a half-
    # spacing short of x = 1 so the base-face drop band cannot eat it.
    x = np.linspace(0.005, 0.995, 120)
    half_t = 0.06 * np.sqrt(x) * (1.0 - 0.6 * x)
    q_kin = 0.5 * u_inf ** 2                    # kinematic dynamic pressure
    cf_up = _cf_profile(x, xtr_upper if xtr_upper is not None else 10.0)
    cf_lo = _cf_profile(x, xtr_lower if xtr_lower is not None else 10.0)

    lines = ["# sampled surface: airfoilSurface (patch airfoil)",
             "#    x    y    z    wallShearStress_x    wallShearStress_y"
             "    wallShearStress_z"]
    for xi, ti, cu, cl in zip(x, half_t, cf_up, cf_lo):
        for yi, mag in ((+ti, cu * q_kin), (-ti, cl * q_kin)):
            lines.append(f"{xi:.8f} {yi:.8f} 0.05 "
                         f"{_TAU_DIR[0] * mag:.10e} "
                         f"{_TAU_DIR[1] * mag:.10e} 0")
    # Blunt-TE base faces on the vertical x = max plane: wild values that
    # would wreck the curve if the extractor failed to drop them.
    for yb in (-0.002, 0.0, 0.002):
        lines.append(f"1.00000000 {yb:.8f} 0.05 9.9 9.9 0")

    dump_dir = case_dir / "postProcessing" / "wallCf" / time_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump = dump_dir / "wallShearStress_airfoilSurface.raw"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dump


def test_extract_cf_raw_reader_contract(tmp_path: Path) -> None:
    """The raw reader returns Nx3 points/tau and skips '#' header lines."""
    dump = make_wall_shear_case(tmp_path / "case")
    pts, tau = extract_cf.read_raw_surface(dump)
    assert pts.shape == (243, 3) and tau.shape == (243, 3)   # 2*120 + 3 base
    # Narrow files (missing tau columns) are rejected loudly, not guessed.
    bad = tmp_path / "narrow.raw"
    bad.write_text("# header\n0.1 0.2 0.3 0.4 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_cf.read_raw_surface(bad)


def test_extract_cf_rho_convention() -> None:
    """Cf = |tau_w/rho| / (0.5 U^2): kinematic over kinematic, NO density.

    A planted Cf of 3e-3 converted to kinematic stress (tau = Cf * 0.5 U^2)
    must round-trip exactly - any density factor would break the identity.
    """
    u = 40.0
    tau_kin = 3.0e-3 * 0.5 * u ** 2
    cf = extract_cf.compute_cf(np.array([[tau_kin, 0.0, 0.0]]), u)
    assert cf[0] == pytest.approx(3.0e-3, rel=1e-12)
    with pytest.raises(ValueError):
        extract_cf.compute_cf(np.array([[1.0, 0.0, 0.0]]), 0.0)


def test_extract_cf_parse_freestream(tmp_path: Path) -> None:
    """U_inf magnitude and AoA come from the case's own 0/U vector."""
    case = tmp_path / "case"
    make_wall_shear_case(case, alpha_deg=6.0, u_inf=43.8)
    u_inf, alpha = extract_cf.parse_freestream(case)
    assert u_inf == pytest.approx(43.8, rel=1e-6)
    assert math.degrees(alpha) == pytest.approx(6.0, abs=1e-6)


def test_extract_cf_end_to_end_detects_transition(tmp_path: Path) -> None:
    """Full producer chain: raw dump -> cf_upper.csv -> detector hit.

    The planted upper-surface transition at x = 0.45 must survive the whole
    pipeline (latest-time discovery, base-face drop, midline surface split,
    Cf conversion, CSV round-trip); the produced curve must also drive the
    gate to BOTH verdicts depending on the XFOIL reference.
    """
    case = tmp_path / "case"
    # Decoy earlier write: fully laminar. If the extractor read this time
    # directory instead of the latest one, no transition would be found.
    make_wall_shear_case(case, xtr_upper=None, time_name="1000")
    make_wall_shear_case(case, xtr_upper=0.45, time_name="3000")

    out = extract_cf.extract_case(case)
    assert out.name == "cf_upper.csv" and out.parent.name == "postProcessing"
    table = read_csv_columns(out)
    # Suction side only: exactly the 120 upper faces, base/lower dropped.
    assert len(table["x_c"]) == 120
    assert float(np.max(table["x_c"])) <= 1.0 + 1e-9

    # The written Cf must reproduce the planted profile (allowing for the
    # x_c renormalization by the LE/TE extent of the face centres).
    x_raw = table["x_c"] * (0.995 - 0.005) + 0.005
    expected = _cf_profile(x_raw, 0.45)
    assert np.max(np.abs(table["cf"] - expected) / expected) < 1e-4

    found = detect_transition_from_cf(table["x_c"], table["cf"])
    assert found is not None and abs(found - 0.45) <= 0.02

    # Gate verdicts off the produced curve: PASS against a close XFOIL
    # xtr_top, FAIL against a far one (worst-pair criterion).
    rans = {"alpha": np.array([4.0]), "xtr_top": np.array([found])}
    xf_close = _xfoil_with_xtr([4.0], [found + 0.05])
    xf_far = _xfoil_with_xtr([4.0], [found + 0.30])
    assert gate_transition(rans, xf_close).status == "PASS"
    assert gate_transition(rans, xf_far).status == "FAIL"


def test_extract_cf_laminar_curve_yields_no_transition(tmp_path: Path
                                                       ) -> None:
    """The other detector direction: a laminar-only dump produces a valid
    CSV whose curve carries NO transition signature (None, never a guess)."""
    case = tmp_path / "case"
    make_wall_shear_case(case, xtr_upper=None)
    table = read_csv_columns(extract_cf.extract_case(case))
    assert detect_transition_from_cf(table["x_c"], table["cf"]) is None


def test_extract_cf_negative_alpha_uses_lower_surface(tmp_path: Path) -> None:
    """At negative AoA the LOWER surface is the suction side (BL-line rule).

    Transition planted on the lower surface only; the upper is laminar. A
    wrong-side pick would find nothing.
    """
    case = tmp_path / "case"
    make_wall_shear_case(case, alpha_deg=-4.0, xtr_upper=None, xtr_lower=0.30)
    table = read_csv_columns(extract_cf.extract_case(case))
    found = detect_transition_from_cf(table["x_c"], table["cf"])
    assert found is not None and abs(found - 0.30) <= 0.02


def test_extract_cf_cli_exit_codes(tmp_path: Path) -> None:
    """CLI contract: 0 when every case produced a curve, 2 on any failure."""
    good = tmp_path / "good"
    make_wall_shear_case(good)
    assert extract_cf.main([str(good)]) == 0
    assert (good / "postProcessing" / "cf_upper.csv").is_file()
    # A case without any wall-shear dump must fail loudly, not invent data.
    empty = tmp_path / "empty"
    make_wall_shear_case(empty, time_name="100")
    import shutil as _shutil
    _shutil.rmtree(empty / "postProcessing")
    assert extract_cf.main([str(empty), str(good)]) == 2


# =============================================================================
#  Pseudo-transient annotation (controlDict 'application' census)
# =============================================================================
def _make_sweep_case(root: Path, name: str, app: str) -> Path:
    """One fake sweep case: just the system/controlDict the census reads."""
    sys_dir = root / name / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "controlDict").write_text(
        "FoamFile\n{\n    object controlDict;\n}\n"
        f"application     {app};\n"
        "startFrom       latestTime;\n",
        encoding="utf-8")
    return root / name


def test_read_case_application(tmp_path: Path) -> None:
    """The application entry round-trips; absence reads as None."""
    case = _make_sweep_case(tmp_path, "val2d_aoa4_re3e6_lvl0", "simpleFoam")
    assert read_case_application(case) == "simpleFoam"
    _make_sweep_case(tmp_path, "p", "pimpleFoam")
    assert read_case_application(tmp_path / "p") == "pimpleFoam"
    assert read_case_application(tmp_path / "does_not_exist") is None


def test_pseudo_transient_cases_scan(tmp_path: Path) -> None:
    """Only matching-config cases with a non-simpleFoam app are reported.

    Covers the alpha-token decode ('m' = minus, 'p' = decimal point) and
    the Re/level tag filters - a pimpleFoam case at the WRONG Re must not
    leak into this configuration's annotation.
    """
    root = tmp_path / "cases"
    _make_sweep_case(root, "val2d_aoa16_re3e6_lvl0", "pimpleFoam")
    _make_sweep_case(root, "val2d_aoa4_re3e6_lvl0", "simpleFoam")
    _make_sweep_case(root, "val2d_aoam2p5_re3e6_lvl0", "pimpleFoam")
    _make_sweep_case(root, "val2d_aoa18_re6e6_lvl0", "pimpleFoam")  # other Re
    _make_sweep_case(root, "val2d_aoa18_re3e6_lvl1", "pimpleFoam")  # other lvl
    hits = pseudo_transient_cases(root, 3.0e6, 0)
    assert hits == [(-2.5, "pimpleFoam"), (16.0, "pimpleFoam")]
    # Absent root: empty census, never an exception (dry-run friendliness).
    assert pseudo_transient_cases(tmp_path / "nope", 3.0e6, 0) == []


def test_main_report_flags_pseudo_transient(tmp_path: Path) -> None:
    """End-to-end: a pimpleFoam case shows up annotated in report.md."""
    root = tmp_path / "cases"
    _make_sweep_case(root, "val2d_aoa16_re3e6_lvl0", "pimpleFoam")
    _make_sweep_case(root, "val2d_aoa4_re3e6_lvl0", "simpleFoam")
    report = tmp_path / "report.md"
    rc = gate_main([
        "--re", "3e6", "--level", "0",
        "--of-csv", str(tmp_path / "absent.csv"),
        "--xfoil-dir", str(tmp_path / "no_xfoil"),
        "--nasa-dir", str(tmp_path / "no_nasa"),
        "--transition-csv", str(tmp_path / "absent_tr.csv"),
        "--case-root", str(root),
        "--report", str(report),
    ])
    assert rc == 0                       # all gates SKIPPED, none failed
    text = report.read_text(encoding="utf-8")
    # Config row + explicit note both carry the flagged point; the steady
    # alpha-4 case must NOT be listed.
    assert "PSEUDO-TRANSIENT points at alpha = +16 deg" in text
    assert "alpha +16 deg (pimpleFoam)" in text
    assert "+4 deg (pimpleFoam)" not in text


# =============================================================================
#  run_validation: pimple variant hook + post-stall auto-retry
# =============================================================================
def _make_solver_case(tmp_path: Path, name: str = "val2d_aoa16_re3e6_lvl0",
                      n_sub: int = 4) -> Path:
    """Fake case dir with the dictionaries solve_case actually reads."""
    case = _make_sweep_case(tmp_path, name, "simpleFoam")
    (case / "system" / "decomposeParDict").write_text(
        f"numberOfSubdomains {n_sub};\n", encoding="utf-8")
    return case


class _StubYPlus:
    """Minimal YPlusVerdict stand-in for the solve_case wall-y+ gate hook.

    Carries exactly the attributes solve_case reads (passed/status/y_max/
    fraction_above) plus the spec-required report() so the stub exercises
    the same code path as the real verdict without any solver output.
    """

    def __init__(self, passed: bool = True):
        self.passed = passed
        self.status = "PASS" if passed else "FAIL"
        self.y_max = 1.2 if passed else 7.5
        self.fraction_above = 0.0 if passed else 0.25
        self.n_faces = 128

    def report(self) -> str:
        return f"wall y+ gate verdict: {self.status} (stub)"


def test_apply_pimple_variant_from_template(tmp_path: Path) -> None:
    """The hook swaps system/ to the variant from the repo template.

    Cases built before pimple_overrides/ existed carry no local copy, so
    the template directory is the fallback source; afterwards the case's
    controlDict must name pimpleFoam and the schemes must be localEuler.
    """
    case = _make_solver_case(tmp_path)
    src = run_validation.apply_pimple_variant(case)
    assert src == run_validation.PIMPLE_OVERRIDES
    assert read_case_application(case) == "pimpleFoam"
    schemes = (case / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert "localEuler" in schemes
    solution = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "PIMPLE" in solution and "nOuterCorrectors" in solution


def test_apply_pimple_variant_prefers_case_local(tmp_path: Path) -> None:
    """A case-local pimple_overrides/ copy wins over the repo template."""
    case = _make_solver_case(tmp_path)
    local = case / "pimple_overrides"
    local.mkdir()
    for fname in run_validation.PIMPLE_VARIANT_FILES:
        (local / fname).write_text(
            f"application     pimpleFoam;  // case-local {fname}\n",
            encoding="utf-8")
    src = run_validation.apply_pimple_variant(case)
    assert src == local
    text = (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "case-local controlDict" in text


def test_solve_case_post_stall_retry(monkeypatch, tmp_path: Path) -> None:
    """Non-converged post-stall case retries ONCE with the pimple variant.

    The solver chain is stubbed (OpenFOAM absent); the convergence gate
    reports FAIL for the steady attempt and PASS for the retry. Expected:
    variant applied, pimpleFoam run at the decomposeParDict rank count, the
    mesh NOT rebuilt, and the status marker flagged pseudo-transient.
    """
    case = _make_solver_case(tmp_path, n_sub=4)
    commands: list = []
    monkeypatch.setattr(run_validation, "run_case_command",
                        lambda mode, d, command, log: commands.append(command)
                        or 0)
    gates = iter([{"converged": False, "cl": 0.9, "cd": 0.09, "cm": -0.05},
                  {"converged": True, "cl": 1.4, "cd": 0.12, "cm": -0.06}])
    monkeypatch.setattr(run_validation, "parse_forces_means",
                        lambda d: next(gates))
    monkeypatch.setattr(run_validation, "extract_transition", lambda d: 0.07)
    # No solver output exists, so the real y+ gate would (correctly) flag
    # the case INDETERMINATE; a passing stub isolates the retry logic.
    monkeypatch.setattr(run_validation, "case_yplus_verdict",
                        lambda d: _StubYPlus(passed=True))

    status = run_validation.solve_case("native", 16.0, case, cores=8,
                                       force=False, pimple_alpha=14.0)
    assert status["converged"] and status["pseudo_transient"]
    assert status["yplus_passed"] is True          # gate ran and recorded
    assert status["application"] == "pimpleFoam"
    assert status["cl"] == pytest.approx(1.4)        # retry means, not steady
    # Steady attempt meshed once; the retry restarts at decomposePar with
    # the SAME rank count read from decomposeParDict.
    assert commands.count("blockMesh") == 1
    assert "mpirun -np 4 simpleFoam -parallel" in commands
    assert "mpirun -np 4 pimpleFoam -parallel" in commands
    # The variant landed in the case's system/ and the marker records it.
    assert read_case_application(case) == "pimpleFoam"
    marker = run_validation.read_status(case)
    assert marker and marker["pseudo_transient"] and marker["converged"]


def test_solve_case_no_retry_below_threshold(monkeypatch, tmp_path: Path
                                             ) -> None:
    """A non-converged LINEAR-range case is flagged, never pimple-retried.

    Pre-stall non-convergence means a setup problem; masking it with the
    fallback would hide the bug, so the threshold must hold the line.
    """
    case = _make_solver_case(tmp_path, name="val2d_aoa4_re3e6_lvl0")
    commands: list = []
    monkeypatch.setattr(run_validation, "run_case_command",
                        lambda mode, d, command, log: commands.append(command)
                        or 0)
    monkeypatch.setattr(run_validation, "parse_forces_means",
                        lambda d: {"converged": False, "cl": 0.8,
                                   "cd": 0.02, "cm": -0.09})
    status = run_validation.solve_case("native", 4.0, case, cores=8,
                                       force=False, pimple_alpha=14.0)
    assert not status["converged"]
    assert status["application"] == "simpleFoam"
    assert "pseudo_transient" not in status
    assert not any("pimpleFoam" in c for c in commands)
    # The case dictionaries were left untouched (still the steady solver).
    assert read_case_application(case) == "simpleFoam"


def test_solve_case_yplus_gate_fails_case(monkeypatch, tmp_path: Path) -> None:
    """A converged solve with a failing wall-y+ verdict is a FAILED case.

    Spec section 4: y+ > 2 on more than 1% of wall faces fails the case.
    The convergence gate passes here, so only the y+ verdict can (and must)
    flip the outcome - the case is flagged, excluded from the polar
    aggregation, and never pimple-retried (a mesh problem is not stall
    physics, and alpha 4 sits below the retry threshold anyway).
    """
    case = _make_solver_case(tmp_path, name="val2d_aoa4_re3e6_lvl0")
    monkeypatch.setattr(run_validation, "run_case_command",
                        lambda mode, d, command, log: 0)
    monkeypatch.setattr(run_validation, "parse_forces_means",
                        lambda d: {"converged": True, "cl": 0.85,
                                   "cd": 0.011, "cm": -0.08})
    monkeypatch.setattr(run_validation, "case_yplus_verdict",
                        lambda d: _StubYPlus(passed=False))
    # Post-processing must never run on a y+-failed case; trip if it does.
    monkeypatch.setattr(run_validation, "extract_transition",
                        lambda d: pytest.fail("post-processing ran on a "
                                              "y+-failed case"))

    status = run_validation.solve_case("native", 4.0, case, cores=8,
                                       force=False, pimple_alpha=14.0)
    assert not status["converged"]                 # y+ verdict failed the case
    assert status["failed_step"] == "yplus"
    assert status["yplus_passed"] is False
    assert status["yplus_status"] == "FAIL"
    assert status["yplus_max"] == pytest.approx(7.5)
    # The marker on disk carries the same flagged verdict (idempotency
    # record: a y+-failed case is re-attempted on the next sweep run).
    marker = run_validation.read_status(case)
    assert marker and not marker["converged"]
    assert marker["failed_step"] == "yplus"


# =============================================================================
#  Misc helpers
# =============================================================================
def test_re_tag_formats() -> None:
    """Filename tags match the polar naming convention across the Re grid."""
    assert re_tag(1.5e6) == "1.5e6"
    assert re_tag(3.0e6) == "3e6"
    assert re_tag(6.0e6) == "6e6"


def test_gate_result_dataclass_defaults() -> None:
    """GateResult carries numeric fields as None until a gate fills them."""
    r = GateResult("demo", "SKIPPED", "reason")
    assert r.measured is None and r.reference is None and r.delta is None
