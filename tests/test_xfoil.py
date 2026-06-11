"""Unit tests for validation/xfoil_polar.py (M1 XFOIL batch driver).

Everything here runs WITHOUT invoking XFOIL: the stdin-script builder and
the PACC polar-dump parser are exercised against canned fixtures captured
verbatim from real XFOIL 6.99 output on this machine. The one true
integration test at the bottom is gated on the binary actually being
present (tools/xfoil/SOURCE.md documents how to fetch it) and is skipped
otherwise, so the suite stays green on a fresh clone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from validation.xfoil_polar import (
    DEFAULT_RE_LIST,
    N_FOIL_POINTS,
    REPO,
    XTR_FREE,
    XTR_TRIP,
    build_alpha_schedule,
    build_xfoil_script,
    csv_name,
    find_skipped,
    parse_polar_dump,
    re_label,
    resolve_xfoil,
    run_polar,
)

# =============================================================================
#  Canned fixtures - copied byte-for-byte from a real XFOIL 6.99 PACC dump
#  (Re 1.5M tripped run on the committed LS(1)-0413 resample), including the
#  spaced-out exponent in the Re field and the '1 1 ... fixed' type line that
#  a naive numeric-line parser would swallow as data.
# =============================================================================

POLAR_DUMP_FIXTURE = """\

       XFOIL         Version 6.99

 Calculated polar for: LS(1)-0413 sharp-TE 241 pts

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   0.050 (top)        0.050 (bottom)
 Mach =   0.000     Re =     1.500 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
  -4.000  -0.0294   0.01056   0.00251  -0.0930   0.0500   0.0351
  -2.000   0.1968   0.01032   0.00210  -0.0941   0.0500   0.0500
   0.000   0.4212   0.01055   0.00222  -0.0947   0.0500   0.0500
   8.000   1.2603   0.01443   0.00670  -0.0872   0.0500   0.0500
  14.000   1.6615   0.02741   0.02059  -0.0530   0.0209   0.0500
"""

# Header-only variant: what PACC leaves behind when the very first point of
# a sweep fails (or the process is killed before anything converges).
POLAR_DUMP_EMPTY = """\

       XFOIL         Version 6.99

 Calculated polar for: LS(1)-0413 sharp-TE 241 pts

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     6.000 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
"""


# =============================================================================
#  AoA schedule (spec: 2-deg steps to 8, then 1-deg to 20 - fine near stall)
# =============================================================================

class TestAlphaSchedule:
    def test_default_schedule_matches_spec(self):
        alphas = build_alpha_schedule()
        expected = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                    16.0, 17.0, 18.0, 19.0, 20.0]
        assert alphas == expected

    def test_schedule_is_strictly_ascending(self):
        # Ascending order is load-bearing: the sweep warm-starts each point
        # from its lower neighbor, and the freeze-salvage logic assumes
        # everything past the last converged row never ran.
        alphas = build_alpha_schedule()
        assert all(b > a for a, b in zip(alphas, alphas[1:]))

    def test_fine_spacing_near_stall(self):
        # Every step at and above 8 deg must be the 1-deg stall resolution.
        alphas = build_alpha_schedule()
        near_stall = [a for a in alphas if a >= 8.0]
        steps = [b - a for a, b in zip(near_stall, near_stall[1:])]
        assert steps and all(abs(s - 1.0) < 1e-9 for s in steps)


# =============================================================================
#  Filename tags
# =============================================================================

class TestNaming:
    def test_re_labels_for_spec_matrix(self):
        # The file list is polar_Re{1.5|3|6}M_N9_{free|trip}.csv - labels
        # must collapse integral millions but keep the 1.5 fraction.
        assert [re_label(r) for r in DEFAULT_RE_LIST] == ["1.5M", "3M", "6M"]

    def test_csv_names_carry_ncrit_token(self):
        # Naming contract polar_Re<tag>_N<ncrit>_<transition>.csv: the gate
        # evaluator matches the requested Ncrit by filename, so the token is
        # load-bearing - on tripped files too (uniform discovery pattern).
        assert csv_name(1.5e6, "free", 9.0) == "polar_Re1.5M_N9_free.csv"
        assert csv_name(6.0e6, "trip", 9.0) == "polar_Re6M_N9_trip.csv"

    def test_csv_name_ncrit_formats(self):
        # %g formatting: integral Ncrit collapses (9.0 -> N9), fractional
        # values keep their decimals (7.5 -> N7.5), default is the spec N9.
        assert csv_name(3.0e6, "free") == "polar_Re3M_N9_free.csv"
        assert csv_name(3.0e6, "free", 7.5) == "polar_Re3M_N7.5_free.csv"
        assert csv_name(3.0e6, "free", 4) == "polar_Re3M_N4_free.csv"


# =============================================================================
#  Stdin-script builder
# =============================================================================

class TestScriptBuilder:
    def _build(self, **kw):
        defaults = dict(foil_file="foil.dat", polar_file="polar.txt",
                        re_value=6.0e6, ncrit=9.0, xtr=XTR_FREE,
                        alphas=[-4.0, 0.0, 20.0])
        defaults.update(kw)
        return build_xfoil_script(**defaults)

    def test_graphics_disabled_first(self):
        # The PLOP/G toggle must precede everything else or the Windows
        # build opens a plot window and batch use breaks.
        lines = self._build().splitlines()
        assert lines[0] == "PLOP"
        assert lines[1] == "G"
        assert lines[2] == ""

    def test_load_then_repanel(self):
        lines = self._build().splitlines()
        i_load = lines.index("LOAD foil.dat")
        # PANE directly after LOAD: re-paneling on XFOIL's spline is what
        # keeps transition predictions clean (see driver module docstring).
        assert lines[i_load + 1] == "PANE"

    def test_repanel_can_be_disabled(self):
        assert "PANE" not in self._build(repanel=False).splitlines()

    def test_viscous_setup_lines(self):
        script = self._build()
        assert "VISC 6000000" in script.splitlines()
        assert "MACH 0" in script.splitlines()
        assert "ITER 200" in script.splitlines()
        assert "N 9" in script.splitlines()

    def test_free_transition_trip_stations(self):
        assert "XTR 1.000 1.000" in self._build(xtr=XTR_FREE).splitlines()

    def test_tripped_trip_stations(self):
        # --tripped mode contract: forced transition at 5% on both surfaces.
        assert "XTR 0.050 0.050" in self._build(xtr=XTR_TRIP).splitlines()

    def test_pacc_opens_with_polar_file_and_closes(self):
        lines = self._build().splitlines()
        pacc_idx = [i for i, l in enumerate(lines) if l == "PACC"]
        # Exactly two PACC commands: accumulation on, accumulation off.
        assert len(pacc_idx) == 2
        # The save-file name follows the opening PACC, then the declined
        # dump-file prompt (blank line).
        assert lines[pacc_idx[0] + 1] == "polar.txt"
        assert lines[pacc_idx[0] + 2] == ""

    def test_alfa_lines_in_request_order(self):
        lines = self._build().splitlines()
        alfa = [l for l in lines if l.startswith("ALFA")]
        assert alfa == ["ALFA -4.00", "ALFA 0.00", "ALFA 20.00"]

    def test_quits_cleanly(self):
        # Stream must end with QUIT or XFOIL hits EOF at a prompt and the
        # Fortran runtime spins forever re-reading closed stdin.
        lines = self._build().splitlines()
        assert lines[-1] == "QUIT"

    def test_iteration_cap_configurable(self):
        assert "ITER 350" in self._build(n_iter=350).splitlines()


# =============================================================================
#  Polar-dump parser
# =============================================================================

class TestPolarParser:
    def test_row_count_and_column_mapping(self):
        meta, rows = parse_polar_dump(POLAR_DUMP_FIXTURE)
        assert len(rows) == 5
        first = rows[0]
        # Column contract: alpha,cl,cd,cdp,cm,xtr_top,xtr_bot in file order.
        assert first["alpha"] == pytest.approx(-4.000)
        assert first["cl"] == pytest.approx(-0.0294)
        assert first["cd"] == pytest.approx(0.01056)
        assert first["cdp"] == pytest.approx(0.00251)
        assert first["cm"] == pytest.approx(-0.0930)
        assert first["xtr_top"] == pytest.approx(0.0500)
        assert first["xtr_bot"] == pytest.approx(0.0351)
        assert rows[-1]["alpha"] == pytest.approx(14.0)

    def test_metadata_scraped_from_header(self):
        meta, _ = parse_polar_dump(POLAR_DUMP_FIXTURE)
        # The Re field is formatted '1.500 e 6' (spaced exponent) by XFOIL;
        # the parser must reassemble it into a plain float.
        assert meta["re"] == pytest.approx(1.5e6)
        assert meta["mach"] == pytest.approx(0.0)
        assert meta["ncrit"] == pytest.approx(9.0)
        assert meta["xtrf_top"] == pytest.approx(0.05)
        assert meta["xtrf_bot"] == pytest.approx(0.05)
        assert meta["name"].startswith("LS(1)-0413")

    def test_header_numerals_never_become_rows(self):
        # The '1 1 Reynolds number fixed' line and the Mach/Re/Ncrit line
        # both contain numbers; only 7-float lines AFTER the dashed rule
        # may be parsed as data.
        _, rows = parse_polar_dump(POLAR_DUMP_FIXTURE)
        assert all(-4.0 <= r["alpha"] <= 14.0 for r in rows)

    def test_empty_table_yields_no_rows(self):
        meta, rows = parse_polar_dump(POLAR_DUMP_EMPTY)
        assert rows == []
        # Header metadata is still recoverable from a row-less dump.
        assert meta["re"] == pytest.approx(6.0e6)
        assert meta["xtrf_top"] == pytest.approx(1.0)


# =============================================================================
#  Unconverged-point (skip) detection
# =============================================================================

class TestSkipDetection:
    def test_missing_alphas_reported(self):
        _, rows = parse_polar_dump(POLAR_DUMP_FIXTURE)
        requested = [-4.0, -2.0, 0.0, 2.0, 8.0, 14.0, 15.0]
        # 2 and 15 are absent from the fixture -> unconverged -> skipped.
        assert find_skipped(requested, rows) == [2.0, 15.0]

    def test_all_present_means_no_skips(self):
        _, rows = parse_polar_dump(POLAR_DUMP_FIXTURE)
        assert find_skipped([-4.0, 0.0, 14.0], rows) == []

    def test_tolerance_absorbs_dump_formatting(self):
        # The dump prints alpha to 3 decimals; a 0.05-deg tolerance must
        # match those back to the requested grid without false skips.
        rows = [{"alpha": 9.001}, {"alpha": 10.0}]
        assert find_skipped([9.0, 10.0, 11.0], rows) == [11.0]

    def test_everything_skipped_for_empty_polar(self):
        assert find_skipped([0.0, 4.0], []) == [0.0, 4.0]


# =============================================================================
#  Integration (slow): only when an XFOIL binary is actually available
# =============================================================================

# Cheap presence check (no subprocess at collection time): the repo-local
# binary from tools/xfoil/SOURCE.md or an explicit XFOIL_BIN override.
_HAVE_XFOIL = (REPO / "tools" / "xfoil" / "xfoil.exe").is_file() \
    or bool(os.environ.get("XFOIL_BIN"))


@pytest.mark.skipif(not _HAVE_XFOIL,
                    reason="xfoil binary absent - fetch per tools/xfoil/SOURCE.md")
def test_integration_two_point_polar():
    """Slow end-to-end check: real XFOIL, tiny 2-point free polar at Re 3M.

    Asserts the full driver chain (resample -> temp Selig file -> stdin
    script -> subprocess -> dump parse) produces physically sane numbers:
    positive lift slope, drag in the laminar-bucket band, and the metadata
    echoing the requested run settings.
    """
    from geometry.airfoil import load_airfoil, resample_airfoil

    argv = resolve_xfoil(None)
    coords = resample_airfoil(
        load_airfoil(REPO / "geometry" / "ls413.dat"),
        n_points=N_FOIL_POINTS, te="sharp",
    )
    res = run_polar(argv, coords, 3.0e6, "free", 9.0, [0.0, 4.0])

    # Both benign low-alpha points must converge on this section.
    assert len(res.rows) == 2 and not res.skipped and not res.hung
    cl0, cl4 = res.rows[0]["cl"], res.rows[1]["cl"]
    # LS(1)-0413 carries ~0.45 of camber lift at alpha 0 and a healthy
    # positive slope; bands are deliberately loose (paneling-insensitive).
    assert 0.3 < cl0 < 0.7
    assert cl4 > cl0 + 0.3
    # Free-transition drag at low lift sits in the laminar bucket.
    assert all(0.003 < r["cd"] < 0.012 for r in res.rows)
    # Run settings round-trip through XFOIL's own dump header.
    assert res.meta["re"] == pytest.approx(3.0e6)
    assert res.meta["ncrit"] == pytest.approx(9.0)
