# -*- coding: utf-8 -*-
"""Synthetic-data tests for the M1 post-processing pair (no OpenFOAM needed).

Covers scripts/parse_forces.py and scripts/extract_bl.py end to end against
fabricated postProcessing/ trees:

  * force-coefficient parsing: ESI 'coefficient.dat' header junk (banner
    lines, '#Time' with no space, Cs/CmRoll/CmYaw and Cl(f)/Cl(r) decoy
    columns), legacy 'forceCoeffs.dat' layout, and restart concatenation
    where duplicated iteration values must keep the LAST occurrence;
  * the spec section-4 convergence gate both ways: a known-flat tail that
    must PASS and an oscillating / high-residual / short history that must
    FAIL with the right sub-gate flagged;
  * the spec section-4 wall-y+ gate: synthetic ASCII yPlus field files
    (uniform and nonuniform boundaryField values, decoy patches, stale time
    directories) driving PASS / FAIL on the >2-over-1%-of-faces rule with
    the histogram checked bin by bin, plus the stats-only fallbacks (solver
    log echo and yPlus.dat) covering the conservative PASS, the
    INDETERMINATE-treated-as-FAIL path, and the no-evidence case;
  * boundary-layer extraction: a Blasius-like sine profile with closed-form
    delta99 / delta* / theta (delta99 = (2/pi) asin(0.99) delta and so on)
    recovered within 1% from both raw .xy and csv sample formats, the
    monotonicity guard on the edge finder, the csv-over-xy / latest-time
    discovery preferences, the VG-sizing table with its aircraft-scale
    column, and the delta99 figure.

The sine profile u/Ue = sin(pi/2 * n/delta) (u = Ue beyond delta) is the
classic laminar approximation: H = 2.660, within 3% of true Blasius 2.59,
and every thickness has an exact analytic value -- ideal for asserting the
trapezoid pipeline without dragging in an ODE solve.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from geometry.units import load_aircraft
from scripts import extract_bl
from scripts import parse_forces as pf

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_YAML = REPO_ROOT / "aircraft.yaml"

# Sine-profile analytic references (per unit delta).
SINE_DELTA99 = (2.0 / math.pi) * math.asin(0.99)        # 0.909929...
SINE_DELTA_STAR = 1.0 - 2.0 / math.pi                   # 0.363380...
SINE_THETA = 2.0 / math.pi - 0.5                        # 0.136620...
SINE_H = SINE_DELTA_STAR / SINE_THETA                   # 2.659792...


# =============================================================================
#  Fabrication helpers -- force / residual histories
# =============================================================================
def _write(path: Path, text: str) -> None:
    """Write one fabricated postProcessing file, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _coeff_text_esi(times: np.ndarray, cd: np.ndarray, cl: np.ndarray,
                    cm: np.ndarray) -> str:
    """ESI-style coefficient.dat with banner junk and decoy columns.

    Deliberate traps for the parser: multiple '#' banner lines (some with
    parenthesized vectors), the column header written as '#Time' with NO
    space after the hash, and Cs/CmRoll/CmYaw plus the front/rear splits
    Cd(f)/Cd(r)/Cl(f)/Cl(r) which an exact-name matcher must NOT confuse
    with Cd/Cl.
    """
    lines = [
        "# Force coefficients",
        "# dragDir     : (1.000000e+00 0.000000e+00 0.000000e+00)",
        "# liftDir     : (0.000000e+00 1.000000e+00 0.000000e+00)",
        "# pitchAxis   : (0.000000e+00 0.000000e+00 1.000000e+00)",
        "# magUInf     : 4.382000e+01",
        "# lRef        : 1.000000e+00",
        "# Aref        : 1.000000e+00",
        "#Time         Cd            Cs            Cl            CmRoll        "
        "CmPitch       CmYaw         Cd(f)         Cd(r)         Cl(f)         Cl(r)",
    ]
    for t, d, l, m in zip(times, cd, cl, cm):
        lines.append(
            f"{t:<13g} {d:<13.6e} {0.0:<13.6e} {l:<13.6e} {0.0:<13.6e} "
            f"{m:<13.6e} {0.0:<13.6e} {0.6 * d:<13.6e} {0.4 * d:<13.6e} "
            f"{0.55 * l:<13.6e} {0.45 * l:<13.6e}"
        )
    return "\n".join(lines) + "\n"


def _coeff_text_legacy(times: np.ndarray, cd: np.ndarray, cl: np.ndarray,
                       cm: np.ndarray) -> str:
    """Legacy forceCoeffs.dat layout: plain 'Cm' column, '# Time' header."""
    lines = [
        "# forceCoeffs",
        "# Time        Cm            Cd            Cl            Cl(f)         Cl(r)",
    ]
    for t, d, l, m in zip(times, cd, cl, cm):
        lines.append(f"{t:<13g} {m:<13.6e} {d:<13.6e} {l:<13.6e} "
                     f"{0.5 * l:<13.6e} {0.5 * l:<13.6e}")
    return "\n".join(lines) + "\n"


def _solver_info_text(times: np.ndarray, res: np.ndarray) -> str:
    """ESI solverInfo.dat with STRING solver-name columns interleaved.

    The string columns (smoothSolver/GAMG) and the *_final / *_iters
    companions are exactly what the residual loader must skip while keeping
    only the *_initial columns; p carries 1.2x the base residual so the
    'worst field' bookkeeping has a deterministic answer.
    """
    lines = [
        "# Solver information",
        "# Time        U_solver     Ux_initial    Ux_final      Ux_iters  "
        "Uy_initial    Uy_final      Uy_iters  p_solver  p_initial     p_final       p_iters",
    ]
    for t, r in zip(times, res):
        lines.append(
            f"{t:<13g} smoothSolver {r:<13.6e} {r * 1e-2:<13.6e} 4         "
            f"{0.8 * r:<13.6e} {r * 1e-2:<13.6e} 4         GAMG      "
            f"{1.2 * r:<13.6e} {r * 1e-2:<13.6e} 7"
        )
    return "\n".join(lines) + "\n"


def _residuals_text(times: np.ndarray, res: np.ndarray) -> str:
    """Foundation-style residuals.dat: plain numeric per-field columns."""
    lines = ["# Residuals",
             "# Time        p             Ux            Uy            k             omega"]
    for t, r in zip(times, res):
        lines.append(f"{t:<13g} {1.2 * r:<13.6e} {r:<13.6e} {0.8 * r:<13.6e} "
                     f"{0.5 * r:<13.6e} {0.6 * r:<13.6e}")
    return "\n".join(lines) + "\n"


def _make_case(tmp_path: Path, name: str, *, cl_osc: float = 0.0008,
               res_floor: float = 3e-7, n_iter: int = 2000) -> Path:
    """Build one fabricated case: converged-looking forces plus residuals.

    The tail is flat by construction: transients decay with ~100-150
    iteration time constants, leaving only a small sinusoidal ripple whose
    amplitude ``cl_osc`` controls the flatness verdict (0.0008 -> ~0.2%
    pk-pk on Cl, comfortably PASS; 0.02 -> 5%, solid FAIL). ``res_floor``
    sets where the residual decay plateaus (3e-7 passes 1e-5; 5e-4 fails).
    """
    case = tmp_path / name
    t = np.arange(1, n_iter + 1, dtype=float)
    cl = 0.8 - 0.3 * np.exp(-t / 150.0) + cl_osc * np.sin(t / 7.0)
    cd = 0.012 + 0.05 * np.exp(-t / 100.0) + 1e-5 * np.sin(t / 5.0)
    cm = -0.08 + 0.02 * np.exp(-t / 120.0) + 5e-5 * np.sin(t / 9.0)
    res = np.maximum(1e-1 * np.exp(-t / 120.0), res_floor)
    _write(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
           _coeff_text_esi(t, cd, cl, cm))
    _write(case / "postProcessing" / "solverInfo" / "0" / "solverInfo.dat",
           _solver_info_text(t, res))
    return case


# =============================================================================
#  Force-coefficient parsing and restart handling
# =============================================================================
def test_force_restart_keeps_last(tmp_path: Path) -> None:
    """Overlapping restart segments: duplicated times keep the LAST row.

    Segment A (start dir 0) runs iterations 1..1200 at a stale Cl = 0.5;
    segment B (restart dir 800) re-runs 801..2000 at Cl = 0.8. The merged
    history must be strictly increasing, 2000 samples long, and carry the
    RESTART values throughout the overlap -- the pre-restart tail belongs
    to the abandoned trajectory. Uses the legacy file name/layout so both
    accepted formats get coverage.
    """
    case = tmp_path / "restart_case"
    t_a = np.arange(1, 1201, dtype=float)
    t_b = np.arange(801, 2001, dtype=float)
    mk = lambda t, val: (np.full_like(t, 0.012), np.full_like(t, val),
                         np.full_like(t, -0.08))
    cd_a, cl_a, cm_a = mk(t_a, 0.5)
    cd_b, cl_b, cm_b = mk(t_b, 0.8)
    _write(case / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat",
           _coeff_text_legacy(t_a, cd_a, cl_a, cm_a))
    _write(case / "postProcessing" / "forceCoeffs" / "800" / "forceCoeffs.dat",
           _coeff_text_legacy(t_b, cd_b, cl_b, cm_b))

    hist = pf.load_force_coeffs(case)
    assert len(hist) == 2000
    assert np.all(np.diff(hist.time) > 0.0)          # strictly increasing
    assert hist.n_duplicates_dropped == 400          # the 801..1200 overlap
    # Overlap region must show segment-B physics, pre-overlap segment-A.
    assert hist.cl[np.searchsorted(hist.time, 900.0)] == pytest.approx(0.8)
    assert hist.cl[np.searchsorted(hist.time, 500.0)] == pytest.approx(0.5)
    assert hist.cm[np.searchsorted(hist.time, 900.0)] == pytest.approx(-0.08)


def test_esi_header_and_decoy_columns(tmp_path: Path) -> None:
    """'#Time' headers and Cl(f)/Cs decoys parse to the right physics."""
    case = _make_case(tmp_path, "hdr_case", n_iter=600)
    hist = pf.load_force_coeffs(case)
    # Cl must come from the 'Cl' column (tail ~0.8), never Cl(f) (~0.44)
    # or Cs (0.0); Cm must come from CmPitch (~-0.08), never CmRoll (0.0).
    assert hist.cl[-1] == pytest.approx(0.8, abs=0.01)
    assert hist.cd[-1] == pytest.approx(0.012, abs=0.001)
    assert hist.cm[-1] == pytest.approx(-0.08, abs=0.005)


# =============================================================================
#  Convergence gate -- both verdicts plus the failure modes in between
# =============================================================================
def test_gate_pass(tmp_path: Path) -> None:
    """Flat tail + decayed residuals = PASS with correct window means."""
    case = _make_case(tmp_path, "aoa_p04")
    verdict = pf.gate_case(case)
    assert verdict.passed
    assert verdict.residual_pass and verdict.flatness_pass
    assert verdict.window_used == 500
    # Window means recover the constructed asymptotes.
    assert verdict.cl_mean == pytest.approx(0.8, abs=5e-4)
    assert verdict.cd_mean == pytest.approx(0.012, abs=5e-5)
    assert verdict.cm_mean == pytest.approx(-0.08, abs=5e-4)
    # Flatness evidence: ripple amplitude 0.0008 -> ~0.2% pk-pk on Cl.
    assert verdict.oscillation["Cl"] < pf.FLAT_LIMIT_DEFAULT
    # Residual evidence: worst field is p (1.2x base), at the 3e-7 floor.
    assert verdict.worst_residual_field == "p"
    assert verdict.worst_residual == pytest.approx(1.2 * 3e-7, rel=1e-6)
    # Report renders and carries the verdict word (orchestrator greps it).
    assert "PASS" in verdict.report()


def test_gate_fail_oscillating(tmp_path: Path) -> None:
    """2% Cl ripple busts the 0.5% flatness gate; residual sub-gate clean."""
    case = _make_case(tmp_path, "aoa_p12", cl_osc=0.02)
    verdict = pf.gate_case(case)
    assert not verdict.passed
    assert not verdict.flatness_pass
    assert verdict.residual_pass                    # residuals still fine
    # pk-pk/|mean| = 2*0.02/0.8 = 5%, far over the limit.
    assert verdict.oscillation["Cl"] > pf.FLAT_LIMIT_DEFAULT
    assert "FAIL" in verdict.report()


def test_gate_fail_residuals(tmp_path: Path) -> None:
    """Residual plateau at 5e-4 fails the 1e-5 gate despite flat forces."""
    case = _make_case(tmp_path, "aoa_p06", res_floor=5e-4)
    verdict = pf.gate_case(case)
    assert not verdict.passed
    assert verdict.flatness_pass                    # forces are flat
    assert not verdict.residual_pass
    assert verdict.worst_residual == pytest.approx(1.2 * 5e-4, rel=1e-6)


def test_gate_short_history_flagged(tmp_path: Path) -> None:
    """A history shorter than the window cannot certify flatness -> FAIL."""
    case = _make_case(tmp_path, "aoa_p02", n_iter=200)
    verdict = pf.gate_case(case)
    assert not verdict.passed
    assert not verdict.flatness_pass
    assert verdict.window_used == 200
    assert any("cannot certify" in note for note in verdict.notes)


def test_gate_missing_residuals_flagged(tmp_path: Path) -> None:
    """No residual file: gate must auto-flag FAIL, never crash or pass."""
    case = tmp_path / "no_res"
    t = np.arange(1, 1001, dtype=float)
    flat = np.full_like(t, 0.7)
    _write(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
           _coeff_text_esi(t, flat * 0.02, flat, flat * -0.1))
    verdict = pf.gate_case(case)
    assert not verdict.passed
    assert not verdict.residual_pass
    assert any("no residual history" in note for note in verdict.notes)


def test_residuals_plain_format(tmp_path: Path) -> None:
    """Foundation residuals.dat (plain numeric columns) loads field-per-column."""
    case = tmp_path / "plain_res"
    # 800 iterations so the exp(-t/40) decay genuinely reaches the 2e-8
    # floor (1e-1 * exp(-20) ~ 2e-10) and the final-value assertions test
    # the plateau, not a point still on the slope.
    t = np.arange(1, 801, dtype=float)
    res = np.maximum(1e-1 * np.exp(-t / 40.0), 2e-8)
    _write(case / "postProcessing" / "residuals" / "0" / "residuals.dat",
           _residuals_text(t, res))
    hist = pf.load_residuals(case)
    assert set(hist.fields) == {"p", "Ux", "Uy", "k", "omega"}
    assert hist.fields["p"][-1] == pytest.approx(1.2 * 2e-8, rel=1e-6)
    assert hist.fields["Ux"][-1] == pytest.approx(2e-8, rel=1e-6)


# =============================================================================
#  Fabrication helpers -- wall y+ evidence (field files, logs, dat tables)
# =============================================================================
def _yplus_field_text(airfoil_value: str) -> str:
    """OpenFOAM ASCII volScalarField shell around one airfoil 'value' entry.

    Faithful to what the v2506 yPlus FO writes: FoamFile banner comments,
    a uniform internalField decoy, and decoy patches BEFORE and AFTER the
    airfoil entry (farfield uniform 0 would zero the stats if patch
    selection ever grabbed the wrong block; the empty frontAndBack patch
    has no value entry at all and must not crash the parser).
    """
    return (
        "/*--------------------------------*- C++ -*------------------------*\\\n"
        "| =========                 |                                       |\n"
        "\\*------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    arch        \"LSB;label=32;scalar=64\";\n"
        "    class       volScalarField;\n"
        "    location    \"2000\";\n"
        "    object      yPlus;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "dimensions      [0 0 0 0 0 0 0];\n"
        "\n"
        "internalField   uniform 0;\n"
        "\n"
        "boundaryField\n"
        "{\n"
        "    farfield\n"
        "    {\n"
        "        type            calculated;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    airfoil\n"
        "    {\n"
        "        type            calculated;\n"
        f"        value           {airfoil_value};\n"
        "    }\n"
        "    frontAndBack\n"
        "    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
        "\n"
        "// ******************************************************************* //\n"
    )


def _yplus_nonuniform(values: np.ndarray) -> str:
    """Render a per-face scalar list exactly as OpenFOAM serializes it."""
    body = "\n".join(f"{v:.6g}" for v in values)
    return f"nonuniform List<scalar> \n{len(values)}\n(\n{body}\n)\n"


def _yplus_log_text(entries: list) -> str:
    """Fabricate solver-log excerpts with v2506 yPlus FO write blocks.

    ``entries`` is a list of (time, min, max, avg) tuples; each becomes one
    FO echo block. A decoy patch line per block guards the patch filter.
    """
    chunks = []
    for t, mn, mx, av in entries:
        chunks.append(
            f"Time = {t}\n"
            "\n"
            "yPlus yPlus1 write:\n"
            "    writing object yPlus\n"
            f"    patch wingTip y+ : min = 99, max = 999, average = 500\n"
            f"    patch airfoil y+ : min = {mn}, max = {mx}, "
            f"average = {av}\n"
        )
    return "\n".join(chunks) + "\n"


# =============================================================================
#  Wall-y+ gate -- per-face field evidence
# =============================================================================
def test_yplus_field_pass_with_histogram(tmp_path: Path) -> None:
    """All faces below 2: PASS, with the fixed-bin histogram counted exactly.

    The fabricated distribution drops a known face count into each bin
    (edges 0/0.5/1/2/5/10/inf) so the assertion pins the binning itself,
    not just the verdict. Driven through gate_case_yplus to lock the driver
    entry point's signature and defaults.
    """
    values = np.concatenate([
        np.full(100, 0.25),     # [0.0, 0.5)
        np.full(250, 0.75),     # [0.5, 1.0)
        np.full(50, 1.50),      # [1.0, 2.0)
    ])
    case = tmp_path / "aoa_p04"
    _write(case / "2000" / "yPlus", _yplus_field_text(_yplus_nonuniform(values)))

    v = pf.gate_case_yplus(case)
    assert v.passed and v.status == "PASS"
    assert v.source == "field"
    assert v.n_faces == 400
    assert v.fraction_above == 0.0
    assert v.bin_counts == [100, 250, 50, 0, 0, 0]
    assert v.y_max == pytest.approx(1.5)
    assert v.y_mean == pytest.approx((100 * 0.25 + 250 * 0.75 + 50 * 1.5) / 400)
    # The spec-required histogram must actually render in the report.
    rep = v.report()
    assert "PASS" in rep
    assert "y+ histogram (400 faces)" in rep
    assert "#" in rep                              # at least one ASCII bar


def test_yplus_field_fail_over_one_percent(tmp_path: Path) -> None:
    """2% of faces beyond y+ = 2 busts the 1% allowance: hard FAIL."""
    values = np.full(400, 0.8)
    values[:8] = 3.0                                # 8/400 = 2% offenders
    case = tmp_path / "aoa_p12"
    _write(case / "1500" / "yPlus", _yplus_field_text(_yplus_nonuniform(values)))

    v = pf.yplus_verdict(case)
    assert not v.passed and v.status == "FAIL"
    assert v.source == "field"
    assert v.fraction_above == pytest.approx(0.02)
    assert v.bin_counts[3] == 8                     # offenders in [2, 5)
    assert "FAIL" in v.report()


def test_yplus_field_exactly_one_percent_passes(tmp_path: Path) -> None:
    """Spec wording is 'MORE than 1%': exactly 1% of faces over 2 still PASSes."""
    values = np.full(400, 0.9)
    values[:4] = 2.5                                # 4/400 = exactly 1%
    case = tmp_path / "aoa_p10"
    _write(case / "1000" / "yPlus", _yplus_field_text(_yplus_nonuniform(values)))

    v = pf.yplus_verdict(case)
    assert v.passed and v.status == "PASS"
    assert v.fraction_above == pytest.approx(0.01)


def test_yplus_field_uniform_value(tmp_path: Path) -> None:
    """A 'uniform' airfoil value still gates exactly (fraction is 0 or 1)."""
    ok = tmp_path / "uniform_ok"
    _write(ok / "800" / "yPlus", _yplus_field_text("uniform 0.7"))
    v = pf.yplus_verdict(ok)
    assert v.passed and v.source == "field"
    assert v.y_max == pytest.approx(0.7)
    assert any("uniform" in note for note in v.notes)

    bad = tmp_path / "uniform_bad"
    _write(bad / "800" / "yPlus", _yplus_field_text("uniform 2.6"))
    v = pf.yplus_verdict(bad)
    assert not v.passed and v.status == "FAIL"
    assert v.fraction_above == pytest.approx(1.0)


def test_yplus_latest_time_wins(tmp_path: Path) -> None:
    """Only the LATEST time directory's field counts; stale fields are decoys.

    The 1000/ field is catastrophically bad (everything at y+ = 4) while
    2000/ is clean -- picking the wrong directory flips the verdict, so
    this also proves numeric (not lexical) time ordering. A 0/ directory
    without a yPlus file must simply be skipped.
    """
    case = tmp_path / "aoa_p06"
    (case / "0").mkdir(parents=True)                 # no yPlus inside
    _write(case / "1000" / "yPlus",
           _yplus_field_text(_yplus_nonuniform(np.full(200, 4.0))))
    _write(case / "2000" / "yPlus",
           _yplus_field_text(_yplus_nonuniform(np.full(200, 0.6))))

    v = pf.yplus_verdict(case)
    assert v.passed
    assert v.evidence is not None and v.evidence.parent.name == "2000"


# =============================================================================
#  Wall-y+ gate -- stats-only fallbacks and the no-evidence path
# =============================================================================
def test_yplus_log_conservative_pass(tmp_path: Path) -> None:
    """Log-only evidence with max <= 2: conservative PASS, LAST echo wins.

    The early write block carries max = 2.41 (a transient before the BL
    settles); only the final block (max = 1.42) describes the converged
    field, so keep-last is load-bearing for the verdict here.
    """
    case = tmp_path / "log_only"
    case.mkdir()
    (case / "log.simpleFoam").write_text(
        _yplus_log_text([(1900, 0.31, 2.41, 0.94), (2000, 0.28, 1.42, 0.87)]),
        encoding="utf-8")

    v = pf.yplus_verdict(case)
    assert v.passed and v.status == "PASS"
    assert v.source == "stats"
    assert v.n_faces == 0 and v.bin_counts == []
    assert v.y_max == pytest.approx(1.42)
    assert math.isnan(v.fraction_above)
    assert any("conservative" in note for note in v.notes)


def test_yplus_log_indeterminate_treated_as_fail(tmp_path: Path) -> None:
    """Log-only with max > 2: INDETERMINATE, passed False, fix-it guidance."""
    case = tmp_path / "log_bad"
    case.mkdir()
    (case / "log.simpleFoam").write_text(
        _yplus_log_text([(2000, 0.30, 2.40, 0.95)]), encoding="utf-8")

    v = pf.yplus_verdict(case)
    assert not v.passed and v.status == "INDETERMINATE"
    assert v.y_max == pytest.approx(2.40)
    # The message must tell the user how to recover per-face evidence.
    assert any("field write" in note for note in v.notes)
    assert "INDETERMINATE" in v.report()


def test_yplus_dat_table_fallback(tmp_path: Path) -> None:
    """yPlus.dat stats (with a decoy patch row) gate when no field exists."""
    case = tmp_path / "dat_only"
    _write(case / "postProcessing" / "yPlus1" / "0" / "yPlus.dat",
           "# Y+ ()\n"
           "# Time patch min max average\n"
           "1000 wingTip 5.0 9.0 7.0\n"
           "1000 airfoil 0.40 1.90 0.90\n"
           "2000 wingTip 5.0 9.0 7.0\n"
           "2000 airfoil 0.35 1.80 0.85\n")

    v = pf.yplus_verdict(case)
    assert v.passed and v.source == "stats"
    assert v.y_max == pytest.approx(1.80)            # last airfoil row wins
    assert v.evidence is not None and v.evidence.name == "yPlus.dat"


def test_yplus_dat_restart_segments_numeric_order(tmp_path: Path) -> None:
    """Restart segments pick stats by NUMERIC start time, not lexical order.

    Segment 500/ holds stale pre-restart stats (max 2.5 -> would go
    INDETERMINATE); segment 1000/ holds the converged stats (max 1.5 ->
    PASS). Lexically '500' > '1000', so a plain string sort would let the
    stale segment shadow the newer one -- this pins the float ordering.
    """
    case = tmp_path / "restart_stats"
    fo = case / "postProcessing" / "yPlus1"
    _write(fo / "500" / "yPlus.dat",
           "# Time patch min max average\n"
           "600 airfoil 0.50 2.50 1.10\n")
    _write(fo / "1000" / "yPlus.dat",
           "# Time patch min max average\n"
           "2000 airfoil 0.35 1.50 0.85\n")

    v = pf.yplus_verdict(case)
    assert v.passed and v.status == "PASS"
    assert v.y_max == pytest.approx(1.50)            # newer segment wins
    assert v.evidence is not None and v.evidence.parent.name == "1000"


def test_yplus_no_evidence(tmp_path: Path) -> None:
    """A case with no y+ output at all: INDETERMINATE -> auto-flagged FAIL."""
    case = tmp_path / "empty_case"
    case.mkdir()
    v = pf.yplus_verdict(case)
    assert not v.passed and v.status == "INDETERMINATE"
    assert v.source == "none"
    assert any("no y+ evidence" in note for note in v.notes)


# =============================================================================
#  Fabrication helpers -- boundary-layer profiles
# =============================================================================
def _sine_profile(delta: float, u_e: float, n_max: float,
                  n_pts: int) -> tuple:
    """Blasius-like sine profile on a cosine-clustered wall-normal grid.

    u/Ue = sin(pi/2 * n/delta) inside the layer, exactly Ue beyond it; the
    clustering packs points near the wall the way the real prism layers do,
    so the trapezoid accuracy being tested matches production conditions.
    """
    s = np.linspace(0.0, 1.0, n_pts)
    n = n_max * (1.0 - np.cos(0.5 * np.pi * s))
    u = np.where(n < delta, u_e * np.sin(0.5 * np.pi * n / delta), u_e)
    return n, u


def _raw_xy_text(n: np.ndarray, u: np.ndarray) -> str:
    """Raw setFormat: distance + U vector, the vector split across a known
    direction so |U| reconstruction (not just Ux pass-through) is tested."""
    ca, sa = math.cos(0.12), math.sin(0.12)
    lines = [f"{ni:.8e}\t{ui * ca:.8e}\t{ui * sa:.8e}\t{0.0:.8e}"
             for ni, ui in zip(n, u)]
    return "\n".join(lines) + "\n"


def _csv_text(n: np.ndarray, u: np.ndarray) -> str:
    """csv setFormat: xyz points marching along a wall-normal direction.

    Wall distance is NOT a column here -- the reader must reconstruct it as
    |r - r0| from the coordinates, which this fabrication makes exact by
    placing points at base + n * unit_direction.
    """
    base = np.array([0.20, 0.05, 0.0])
    direction = np.array([0.3, 0.95, 0.0])
    direction /= np.linalg.norm(direction)
    ca, sa = math.cos(0.3), math.sin(0.3)
    lines = ["x,y,z,U_x,U_y,U_z"]
    for ni, ui in zip(n, u):
        pt = base + ni * direction
        lines.append(f"{pt[0]:.8e},{pt[1]:.8e},{pt[2]:.8e},"
                     f"{ui * ca:.8e},{ui * sa:.8e},{0.0:.8e}")
    return "\n".join(lines) + "\n"


def _assert_sine_metrics(metrics: extract_bl.BLMetrics, delta: float,
                         u_e: float, tol: float = 0.01) -> None:
    """Assert all four BL quantities within ``tol`` of the analytic values."""
    assert metrics.u_edge == pytest.approx(u_e, rel=tol)
    assert metrics.delta99 == pytest.approx(SINE_DELTA99 * delta, rel=tol)
    assert metrics.delta_star == pytest.approx(SINE_DELTA_STAR * delta, rel=tol)
    assert metrics.theta == pytest.approx(SINE_THETA * delta, rel=tol)
    assert metrics.H == pytest.approx(SINE_H, rel=tol)


# =============================================================================
#  Boundary-layer extraction
# =============================================================================
def test_station_token_parsing() -> None:
    """Every documented station encoding decodes to the same chord fraction."""
    cases = {
        "bl_x007_U": 0.07,            # percent digits
        "bl_x0p07_U": 0.07,           # 'p' decimal mark
        "bl_x0.10_U": 0.10,           # literal decimal
        "BL_x100_U": 1.00,            # TE, percent form
        "blProfiles_x0p35_U": 0.35,   # prefixed set name
        "bl_x35_U": 0.35,             # bare percent digits
    }
    for name, expected in cases.items():
        assert extract_bl.parse_station_token(name) == pytest.approx(expected)
    # Non-stations must be rejected, not guessed at.
    assert extract_bl.parse_station_token("wallShearStress") is None
    assert extract_bl.parse_station_token("bl_x250_U") is None  # 2.5c: bogus


def test_bl_sine_profile_raw_xy(tmp_path: Path) -> None:
    """Raw .xy profile recovers the analytic sine-profile BL within 1%."""
    delta, u_e = 0.008, 43.8
    n, u = _sine_profile(delta, u_e, n_max=3 * delta, n_pts=400)
    case = tmp_path / "aoa_p04"
    _write(case / "postProcessing" / "blProfiles" / "2000" / "bl_x0p07_U.xy",
           _raw_xy_text(n, u))

    stations = extract_bl.extract_case(case)
    assert len(stations) == 1
    assert stations[0].x_over_c == pytest.approx(0.07)
    _assert_sine_metrics(stations[0].metrics, delta, u_e)


def test_bl_sine_profile_csv(tmp_path: Path) -> None:
    """csv (xyz header) profile: distance reconstruction + 1% recovery."""
    delta, u_e = 0.012, 21.9
    n, u = _sine_profile(delta, u_e, n_max=3 * delta, n_pts=400)
    case = tmp_path / "aoa_p08"
    _write(case / "postProcessing" / "blProfiles" / "1500" / "bl_x020_U.csv",
           _csv_text(n, u))

    stations = extract_bl.extract_case(case)
    assert len(stations) == 1
    assert stations[0].x_over_c == pytest.approx(0.20)
    _assert_sine_metrics(stations[0].metrics, delta, u_e)


def test_bl_wall_anchor_prepended(tmp_path: Path) -> None:
    """A profile whose first sample sits off the wall still lands within 1%.

    Set writers often start at the first cell center rather than the wall
    face; the extractor's synthetic (0, 0) no-slip anchor must recover the
    lost near-wall area of the integrals.
    """
    delta, u_e = 0.008, 43.8
    n, u = _sine_profile(delta, u_e, n_max=3 * delta, n_pts=400)
    keep = n >= 2e-4                     # drop everything below 0.2 mm
    case = tmp_path / "aoa_p06"
    _write(case / "postProcessing" / "blProfiles" / "900" / "bl_x010_U.xy",
           _raw_xy_text(n[keep], u[keep]))
    stations = extract_bl.extract_case(case)
    _assert_sine_metrics(stations[0].metrics, delta, u_e)


def test_bl_edge_monotonicity_guard() -> None:
    """Outer-flow decay plus a far-field overshoot must not move the edge.

    Suction-side reality: |U| peaks at the BL edge, decays through the
    outer inviscid gradient, and can exceed the peak again far from the
    wall (domain effects). The guard truncates at the first 2% drop below
    the running max, so the edge stays at the physical peak (50), not the
    far-field overshoot (52).
    """
    delta, u_peak = 0.01, 50.0
    n = np.linspace(0.0, 3.0 * delta, 600)
    u = np.where(n < delta, u_peak * np.sin(0.5 * np.pi * n / delta), 0.0)
    outer = (n >= delta) & (n < 2.0 * delta)        # linear decay 50 -> 45
    u[outer] = u_peak - 5.0 * (n[outer] - delta) / delta
    far = n >= 2.0 * delta                          # spurious rise 45 -> 52
    u[far] = 45.0 + 7.0 * (n[far] - 2.0 * delta) / delta

    metrics = extract_bl.compute_bl_metrics(n, u)
    assert metrics.u_edge == pytest.approx(u_peak, rel=0.002)
    assert metrics.delta99 == pytest.approx(SINE_DELTA99 * delta, rel=0.02)


def test_bl_discovery_prefers_csv_and_latest_time(tmp_path: Path) -> None:
    """Same station in both formats and several times: csv + latest win.

    The stale time directory (500) and the .xy twin carry a 2x-thick decoy
    layer; only the csv in the latest directory (2000) holds the true
    profile, so any wrong pick shows up as a 2x delta99 error.
    """
    delta, u_e = 0.008, 43.8
    n, u = _sine_profile(delta, u_e, n_max=3 * delta, n_pts=400)
    n2, u2 = _sine_profile(2 * delta, u_e, n_max=6 * delta, n_pts=400)
    case = tmp_path / "aoa_p10"
    fo = case / "postProcessing" / "blProfiles"
    _write(fo / "500" / "bl_x007_U.csv", _csv_text(n2, u2))    # stale time
    _write(fo / "2000" / "bl_x007_U.csv", _csv_text(n, u))     # truth
    _write(fo / "2000" / "bl_x007_U.xy", _raw_xy_text(n2, u2))  # format decoy
    # A pressure-only sample must be ignored entirely, not crash the walk.
    _write(fo / "2000" / "bl_x007_p.xy",
           "\n".join(f"{ni:.6e}\t{101325.0:.6e}" for ni in n) + "\n")

    stations = extract_bl.extract_case(case)
    assert len(stations) == 1
    assert stations[0].source.suffix == ".csv"
    assert stations[0].source.parent.name == "2000"
    _assert_sine_metrics(stations[0].metrics, delta, u_e)


def test_bl_table_and_aircraft_scale(tmp_path: Path) -> None:
    """The VG-sizing CSV: exact header, mm units, YAML-driven scale column."""
    delta, u_e = 0.008, 43.8
    n, u = _sine_profile(delta, u_e, n_max=3 * delta, n_pts=400)
    case = tmp_path / "aoa_p04"
    fo = case / "postProcessing" / "blProfiles" / "2000"
    _write(fo / "bl_x007_U.xy", _raw_xy_text(n, u))
    _write(fo / "bl_x030_U.xy", _raw_xy_text(n * 1.5, u))   # thicker aft station

    stations = extract_bl.extract_case(case)
    out = extract_bl.write_bl_table(stations, case.name, tmp_path / "results",
                                    AIRCRAFT_YAML, case_chord_m=1.0)
    assert out.name == "bl_table_aoa_p04.csv"

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == ("x_over_c,delta99_mm,delta_star_mm,theta_mm,H,"
                        "delta99_aircraft_mm")
    assert len(lines) == 3                                   # header + 2 stations

    # The aircraft column must equal delta99 * (aileron-mid chord / 1 m),
    # with the chord pulled through the same YAML choke point production uses
    # (DXF-measured 35.520 in -> 0.90221 m, never hard-coded here).
    ac = load_aircraft(AIRCRAFT_YAML)
    ratio = ac.wing.aileron.chord_at_mid_station / 1.0
    for line in lines[1:]:
        vals = [float(v) for v in line.split(",")]
        assert vals[5] == pytest.approx(vals[1] * ratio, abs=2e-3)
    # Stations must be sorted by x/c and physically ordered (BL grows aft).
    assert [float(l.split(",")[0]) for l in lines[1:]] == [0.07, 0.30]
    d99 = [float(l.split(",")[1]) for l in lines[1:]]
    assert d99[1] > d99[0]


def test_bl_plot_written(tmp_path: Path) -> None:
    """The delta99-vs-x/c figure lands on disk, one curve per AoA case."""
    delta, u_e = 0.008, 43.8
    per_case = {}
    for name, fac in (("aoa_m04", 0.8), ("aoa_p08", 1.4)):
        n, u = _sine_profile(fac * delta, u_e, n_max=3 * fac * delta, n_pts=300)
        case = tmp_path / name
        fo = case / "postProcessing" / "blProfiles" / "1000"
        _write(fo / "bl_x007_U.xy", _raw_xy_text(n, u))
        _write(fo / "bl_x050_U.xy", _raw_xy_text(n * 1.8, u))
        per_case[name] = extract_bl.extract_case(case)

    fig_path = extract_bl.plot_delta99(per_case, tmp_path / "results")
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_aoa_label_parsing() -> None:
    """Case-name AoA tokens map to readable labels; minus encodings work."""
    assert extract_bl.parse_aoa_label("aoa_p04") == "AoA +4.0 deg"
    assert extract_bl.parse_aoa_label("aoa_m02") == "AoA -2.0 deg"
    assert extract_bl.parse_aoa_label("alpha6p5") == "AoA +6.5 deg"
    assert extract_bl.parse_aoa_label("re3M_aoa_n10") == "AoA -10.0 deg"
    # Unparseable names fall back to the raw case name (still plottable).
    assert extract_bl.parse_aoa_label("baseline") == "baseline"
