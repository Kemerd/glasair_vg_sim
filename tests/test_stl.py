"""Tests for geometry/stl_gen.py - extruded-section STL generation.

Exercises the watertightness contract (these solids are snappyHexMesh viscous
walls), the hinge-split construction with its open-gap requirement, the gap
audit (as-built clearance and OML opening measurement), the include_gap
split-vs-single contract, and the end-to-end YAML-driven generators against
the committed schema-v2 aircraft.yaml (DXF-measured values).

Section data used here:
  * LS(1)-0413 loop committed at geometry/ls413.dat (NASA/Langley GA(W)-2)
  * NACA 0010 analytic section (the spec's placeholder for the tail surfaces)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import trimesh

from geometry.airfoil import load_airfoil, naca4_coords, resample_airfoil
from geometry.stl_gen import (
    SectionCaseResult,
    _mid_thickness_y,
    check_watertight,
    extrude_section,
    extrude_two_element,
    gen_fin_section_stl,
    gen_stab_section_stl,
    gen_wing_section_stl,
    measure_gap_metrics,
    mesh_report,
    section_polygon,
    split_at_hinge,
)

# Repo root anchors all data-file paths so the tests run from any cwd.
REPO = Path(__file__).resolve().parent.parent

# Hinge-split scenario shared by several tests: NACA 0010 elevator-style cut
# at 70% chord, factory 1/16 in gap (1.5875 mm exactly), 0.7 m chord (about
# the stab root chord).
HINGE_FRAC = 0.70
GAP_M = 1.5875e-3
CHORD_M = 0.7

# DXF-measured values from the schema-v2 aircraft.yaml that the generators
# must consume (SI conversions of the committed inch values).
WING_CHORD_AT_AILERON_M = 35.520 * 0.0254       # wing.aileron.chord_at_mid_station
WING_AILERON_HINGE_FRAC = 0.8013                # wing.aileron.hinge_chord_fraction
STAB_ELEVATOR_HINGE_FRAC = 1.0 - 0.34           # 1 - elevator.chord_fraction
FIN_CHORD_M = 44.778 * 0.0254                   # vertical_tail.chord_root_incl_rudder
FIN_RUDDER_HINGE_FRAC = 1.0 - 0.3712            # 1 - rudder.chord_fraction
FACTORY_GAP_M = 0.0625 * 0.0254                 # 1/16 in = 1.5875e-3 m exactly


# -----------------------------------------------------------------------------
#  Fixtures - one resampled LS(1)-0413 loop and one analytic NACA 0010 loop
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ls413_coords():
    """Blunt-TE cosine-resampled LS(1)-0413, the Study-1 wing section."""
    raw = load_airfoil(REPO / "geometry" / "ls413.dat")
    return resample_airfoil(raw, n_points=241, te="blunt")


@pytest.fixture(scope="module")
def naca0010_coords():
    """Sharp-TE NACA 0010, the spec's tail-surface placeholder section."""
    return naca4_coords("0010", n_points=241, te="sharp")


# -----------------------------------------------------------------------------
#  Single-element extrusion: watertight solid on the periodic z convention
# -----------------------------------------------------------------------------

class TestExtrudeSection:
    def test_ls413_extrusion_is_watertight_solid(self, ls413_coords):
        """1.2 m chord x 0.5 m span wing slab must be a closed solid."""
        mesh = extrude_section(ls413_coords, chord_m=1.2, span_m=0.5)

        # Watertightness is the hard requirement for snappy wall geometry;
        # a positive volume confirms outward-facing normals as well.
        assert check_watertight(mesh), mesh_report(mesh)
        assert mesh.volume > 0.0

        # Chordwise bbox extent must match the requested physical chord
        # (chord-normalized x in [0,1] scaled by 1.2 m), within 1%.
        dx = mesh.bounds[1][0] - mesh.bounds[0][0]
        assert dx == pytest.approx(1.2, rel=0.01)

        # Periodic-domain convention: the solid is centered on midspan, so
        # the z bounds must be symmetric about zero to numerical precision.
        zmin, zmax = mesh.bounds[0][2], mesh.bounds[1][2]
        assert abs(zmin + zmax) < 1e-6
        assert (zmax - zmin) == pytest.approx(0.5, rel=1e-6)


# -----------------------------------------------------------------------------
#  Hinge split: open gap, rigid rotation, no element interference
# -----------------------------------------------------------------------------

class TestSplitAtHinge:
    def test_deflected_split_pieces_valid_and_clear(self, naca0010_coords):
        """TE-up 25 deg elevator cut: both pieces valid, gap stays open."""
        # Negative deflection = TE up per the module sign convention; this is
        # the nose-up-command direction the stab study sweeps.
        main, ctrl = split_at_hinge(
            naca0010_coords, HINGE_FRAC, math.radians(-25.0), GAP_M, CHORD_M
        )

        # Both elements must come back as healthy polygons with real area.
        assert main.is_valid and not main.is_empty and main.area > 0.0
        assert ctrl.is_valid and not ctrl.is_empty and ctrl.area > 0.0

        # No interference: a deflected control fouling the fixed element
        # would produce overlapping wall geometry that snappy cannot mesh.
        assert main.intersection(ctrl).area < 1e-12

        # The open gap must survive the deflection: the cove construction
        # guarantees clearance >= gap, asserted here with 20% margin to
        # absorb the polygonized cove arc and FP noise.
        assert main.distance(ctrl) >= 0.8 * GAP_M

    def test_zero_deflection_split_keeps_gap_open(self, naca0010_coords):
        """The 0 deg baseline carries the SAME open gap as deflected cases.

        Spec Phase 3 runs the baseline WITH the gap - the leak path exists
        on the real aircraft at every deflection - so the split at zero
        deflection must produce a clear two-element pair, not a fused solid.
        """
        main, ctrl = split_at_hinge(naca0010_coords, HINGE_FRAC, 0.0, GAP_M, CHORD_M)

        # Healthy pair with zero overlap and an open channel at least one
        # nominal gap wide (the cove makes it wider; see TestGapMetrics).
        assert main.is_valid and ctrl.is_valid
        assert main.intersection(ctrl).area < 1e-12
        assert main.distance(ctrl) >= GAP_M

    def test_control_area_preserved_under_rotation(self, naca0010_coords):
        """Rigid rotation must not change the control surface area."""
        # Same cut at zero and at -25 deg: the pre-rotation polygons are
        # identical, so any area drift would expose a non-rigid transform.
        _, ctrl_0 = split_at_hinge(naca0010_coords, HINGE_FRAC, 0.0, GAP_M, CHORD_M)
        _, ctrl_25 = split_at_hinge(
            naca0010_coords, HINGE_FRAC, math.radians(-25.0), GAP_M, CHORD_M
        )
        assert ctrl_25.area == pytest.approx(ctrl_0.area, rel=0.01)

    def test_zero_deflection_control_chord_extent(self, naca0010_coords):
        """Undeflected control spans hinge to TE: (1 - 0.70) * 0.7 m chord."""
        _, ctrl = split_at_hinge(naca0010_coords, HINGE_FRAC, 0.0, GAP_M, CHORD_M)

        # Chordwise extent of the control element straight off its bounds;
        # the sharp-TE NACA section ends exactly at x = chord.
        extent = ctrl.bounds[2] - ctrl.bounds[0]
        assert extent == pytest.approx((1.0 - HINGE_FRAC) * CHORD_M, rel=0.02)


# -----------------------------------------------------------------------------
#  Gap audit: the as-built channel is measured, never assumed
# -----------------------------------------------------------------------------

class TestGapMetrics:
    @staticmethod
    def _audit(coords, deflection_deg):
        """Split + audit helper: returns (main, ctrl, GapMetrics)."""
        main, ctrl = split_at_hinge(
            coords, HINGE_FRAC, math.radians(deflection_deg), GAP_M, CHORD_M
        )
        # Same waterline construction the case builder uses: mid-thickness
        # at the hinge station on the unsplit physical-scale section.
        hinge_y = _mid_thickness_y(
            section_polygon(coords, CHORD_M), HINGE_FRAC * CHORD_M
        )
        return main, ctrl, measure_gap_metrics(main, ctrl, hinge_y, GAP_M)

    def test_cove_guarantee_min_clearance_at_deflection(self, naca0010_coords):
        """Deflected case: min clearance >= nominal gap, never less."""
        main, ctrl, m = self._audit(naca0010_coords, -20.0)

        # The cove construction promises clearance >= gap everywhere; the
        # audit must agree with a direct shapely distance computation.
        assert m.min_clearance_m >= m.nominal_gap_m
        assert m.min_clearance_m == pytest.approx(main.distance(ctrl), abs=1e-12)
        assert m.nominal_gap_m == GAP_M

        # OML openings are point-to-polygon distances at the cove lips, so
        # neither can undercut the global polygon-to-polygon minimum.
        assert m.opening_upper_m >= m.min_clearance_m - 1e-12
        assert m.opening_lower_m >= m.min_clearance_m - 1e-12

    def test_opening_is_deflection_dependent_not_nominal(self, naca0010_coords):
        """The achieved opening exceeds the nominal gap and tracks deflection.

        This is the documented cove side effect (2.3-3.3 mm tight-side
        opening across 10-25 deg vs the 1.5875 mm nominal): the audit exists
        precisely so this is observable instead of silently assumed away.
        """
        # Tight-side opening across the sweep: strictly wider than nominal.
        for deg in (-10.0, -15.0, -20.0, -25.0):
            _, _, m = self._audit(naca0010_coords, deg)
            tight = min(m.opening_upper_m, m.opening_lower_m)
            assert tight > m.nominal_gap_m
            # Review-measured envelope for this section family: the tight
            # side stays in the low-millimeter band, far from pathological.
            assert 2.0e-3 < tight < 4.0e-3

        # TE-up rotation swings the control nose down: the LOWER lip opens
        # wide while the upper side stays tight. Sign sanity of the audit.
        _, _, m20 = self._audit(naca0010_coords, -20.0)
        assert m20.opening_lower_m > m20.opening_upper_m

    def test_zero_deflection_audit_symmetric_and_open(self, naca0010_coords):
        """Gapped baseline: open channel, symmetric lips on a symmetric foil."""
        main, ctrl, m = self._audit(naca0010_coords, 0.0)

        # Open everywhere and never tighter than the nominal gap.
        assert main.intersection(ctrl).area < 1e-12
        assert m.min_clearance_m >= m.nominal_gap_m

        # NACA 0010 is symmetric about the chord line, so the upper and
        # lower OML openings must match to numerical tolerance.
        assert m.opening_upper_m == pytest.approx(m.opening_lower_m, rel=1e-3)


# -----------------------------------------------------------------------------
#  Two-element extrusion: the pair must form one meshable case geometry
# -----------------------------------------------------------------------------

class TestExtrudeTwoElement:
    def test_both_elements_watertight(self, naca0010_coords):
        """Fixed + deflected control extrusions must both close up."""
        main_poly, ctrl_poly = split_at_hinge(
            naca0010_coords, HINGE_FRAC, math.radians(-25.0), GAP_M, CHORD_M
        )
        main_mesh, ctrl_mesh = extrude_two_element(main_poly, ctrl_poly, span_m=0.06)

        # Each element is an independent snappy wall region; both must be
        # closed solids or the surface snapping stage will leak.
        assert check_watertight(main_mesh), mesh_report(main_mesh)
        assert check_watertight(ctrl_mesh), mesh_report(ctrl_mesh)

        # Shared z convention: both solids centered on the midspan plane.
        for mesh in (main_mesh, ctrl_mesh):
            assert abs(mesh.bounds[0][2] + mesh.bounds[1][2]) < 1e-6


# -----------------------------------------------------------------------------
#  End-to-end: aircraft.yaml -> stab STL pair on disk -> reload watertight
# -----------------------------------------------------------------------------

class TestGenStabSectionStl:
    def test_te_up_20deg_writes_two_watertight_stls(self, tmp_path):
        """Full Study-2 path: -20 deg elevator emits main + control solids."""
        result = gen_stab_section_stl(
            REPO / "aircraft.yaml",
            tmp_path,
            elevator_deflection_rad=math.radians(-20.0),
        )

        # Any gapped case must produce exactly the two-element pair, with
        # the case summary carrying the YAML-derived parameters.
        assert isinstance(result, SectionCaseResult)
        assert len(result.paths) == 2
        for p in result.paths:
            assert p.exists() and p.suffix == ".stl"
        assert result.include_gap is True
        assert result.hinge_frac == pytest.approx(STAB_ELEVATOR_HINGE_FRAC)

        # The gap audit must ride back on the result: nominal straight from
        # the yaml (1/16 in exactly) and the cove guarantee respected.
        assert result.gap is not None
        assert result.gap.nominal_gap_m == pytest.approx(FACTORY_GAP_M, rel=1e-9)
        assert result.gap.min_clearance_m >= result.gap.nominal_gap_m

        # Round-trip through the ASCII STL on disk: what snappy will read
        # must still be a closed solid after parsing, not just in memory.
        for p in result.paths:
            mesh = trimesh.load(str(p), force="mesh")
            assert check_watertight(mesh), f"{p.name}: {mesh_report(mesh)}"
            assert mesh.volume > 0.0

    def test_zero_deflection_gapped_baseline(self, tmp_path):
        """0 deg with include_gap=True (default): TWO solids, gap open.

        This is the Phase-3 baseline contract: the leak path exists at zero
        deflection on the real aircraft, so the baseline shares the split
        mesh family of the deflected cases instead of fusing into one solid.
        """
        result = gen_stab_section_stl(
            REPO / "aircraft.yaml", tmp_path, elevator_deflection_rad=0.0
        )

        # Two named regions, main + control, both closed solids on disk.
        assert len(result.paths) == 2
        names = sorted(p.name for p in result.paths)
        assert any("_main" in n for n in names)
        assert any("_control" in n for n in names)
        for p in result.paths:
            mesh = trimesh.load(str(p), force="mesh")
            assert check_watertight(mesh), f"{p.name}: {mesh_report(mesh)}"

        # The audit proves the gap is genuinely open at zero deflection.
        assert result.gap is not None
        assert result.gap.min_clearance_m >= result.gap.nominal_gap_m

    def test_include_gap_false_single_clean_solid(self, tmp_path):
        """include_gap=False at 0 deg: the deliberate Phase-1 clean solid."""
        result = gen_stab_section_stl(
            REPO / "aircraft.yaml", tmp_path,
            elevator_deflection_rad=0.0, include_gap=False,
        )

        # One gapless solid, no audit (there is no channel to measure).
        assert len(result.paths) == 1
        assert result.include_gap is False
        assert result.gap is None
        mesh = trimesh.load(str(result.paths[0]), force="mesh")
        assert check_watertight(mesh), mesh_report(mesh)

    def test_include_gap_false_rejects_deflection(self, tmp_path):
        """A deflected control has no gapless form: must raise, not drop."""
        with pytest.raises(ValueError, match="include_gap=False"):
            gen_stab_section_stl(
                REPO / "aircraft.yaml", tmp_path,
                elevator_deflection_rad=math.radians(-20.0), include_gap=False,
            )


# -----------------------------------------------------------------------------
#  End-to-end: wing generator consumes the DXF-measured aileron station data
# -----------------------------------------------------------------------------

class TestGenWingSectionStl:
    def test_clean_solid_uses_measured_chord_and_yaml_hinge(self, tmp_path):
        """Phase-1 clean wing solid: measured chord, DXF hinge fraction."""
        result = gen_wing_section_stl(
            REPO / "aircraft.yaml", tmp_path,
            aileron_deflection_rad=0.0, include_gap=False,
        )

        # The section chord must be the DXF-measured chord_at_mid_station
        # (0.9022 m), NOT the trapezoid interpolation it cross-checks against.
        assert result.chord_m == pytest.approx(WING_CHORD_AT_AILERON_M, rel=1e-9)
        # Hinge fraction comes straight from wing.aileron.hinge_chord_fraction.
        assert result.hinge_frac == pytest.approx(WING_AILERON_HINGE_FRAC)

        # Single clean solid whose chordwise extent matches the chord.
        assert len(result.paths) == 1
        mesh = trimesh.load(str(result.paths[0]), force="mesh")
        assert check_watertight(mesh), mesh_report(mesh)
        dx = mesh.bounds[1][0] - mesh.bounds[0][0]
        assert dx == pytest.approx(WING_CHORD_AT_AILERON_M, rel=0.01)

    def test_gapped_baseline_uses_aileron_gap(self, tmp_path):
        """Default gapped 0 deg wing case: pair with the wing's own gap."""
        result = gen_wing_section_stl(
            REPO / "aircraft.yaml", tmp_path, aileron_deflection_rad=0.0
        )

        # Two solids, audit present, and the gap is wing.aileron.hinge_gap
        # (1/16 in) - no longer borrowed from the elevator entry.
        assert len(result.paths) == 2
        assert result.gap is not None
        assert result.gap.nominal_gap_m == pytest.approx(FACTORY_GAP_M, rel=1e-9)
        assert result.gap.min_clearance_m >= result.gap.nominal_gap_m
        for p in result.paths:
            mesh = trimesh.load(str(p), force="mesh")
            assert check_watertight(mesh), f"{p.name}: {mesh_report(mesh)}"

    def test_planform_inconsistency_raises_runtime_error(self, tmp_path):
        """Corrupting the measured chord must trip the 1% consistency gate."""
        # Clone aircraft.yaml with chord_at_mid_station knocked from the
        # measured 35.520 in down to 30.000 in (an 18% disagreement with the
        # chord_root/chord_tip trapezoid at the same station).
        src = (REPO / "aircraft.yaml").read_text(encoding="utf-8")
        assert "35.520" in src, "yaml fixture drifted; update this test"
        tampered = tmp_path / "aircraft.yaml"
        tampered.write_text(src.replace("35.520", "30.000"), encoding="utf-8")

        # The generator must refuse loudly, naming both numbers, before any
        # geometry is produced from an inconsistent wing block.
        with pytest.raises(RuntimeError, match="chord_at_mid_station"):
            gen_wing_section_stl(tampered, tmp_path, aileron_deflection_rad=0.0)


# -----------------------------------------------------------------------------
#  End-to-end: fin generator now carries its own measured chord (no stand-in)
# -----------------------------------------------------------------------------

class TestGenFinSectionStl:
    def test_fin_chord_and_hinge_from_yaml(self, tmp_path):
        """Fin case at +15 deg: measured root chord, DXF rudder hinge."""
        result = gen_fin_section_stl(
            REPO / "aircraft.yaml", tmp_path,
            rudder_deflection_rad=math.radians(15.0),
        )

        # The fin chord is the DXF-measured chord_root_incl_rudder (1.1374 m);
        # the old stab-root-chord stand-in (0.7005 m) must be gone.
        assert result.chord_m == pytest.approx(FIN_CHORD_M, rel=1e-9)
        # Hinge from the measured rudder chord fraction: 1 - 0.3712.
        assert result.hinge_frac == pytest.approx(FIN_RUDDER_HINGE_FRAC)

        # Standard gapped-pair contract: two closed solids, open channel.
        assert len(result.paths) == 2
        assert result.gap is not None
        assert result.gap.nominal_gap_m == pytest.approx(FACTORY_GAP_M, rel=1e-9)
        assert result.gap.min_clearance_m >= result.gap.nominal_gap_m
        for p in result.paths:
            mesh = trimesh.load(str(p), force="mesh")
            assert check_watertight(mesh), f"{p.name}: {mesh_report(mesh)}"
