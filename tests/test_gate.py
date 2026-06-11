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
    read_csv_columns,
    re_tag,
    upsert_report,
)

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
    with the UNPINNED OpenFOAM-version warning (aircraft.yaml solver block
    is still null), slope gate PASS on matching synthetic slopes, NASA gates
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
        "--report", str(report),
    ])
    assert rc == 0
    text = report.read_text(encoding="utf-8")
    # The unpinned solver version must be loudly visible in the config table.
    assert "UNPINNED" in text
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
        "--report", str(tmp_path / "report.md"),
    ])
    assert rc == 1


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
