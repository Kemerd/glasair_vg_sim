"""Tests for geometry/units.py — unit-tag conversion and aircraft.yaml loading.

Spot-check values are the exact legal conversion definitions (NIST SP 811 /
BIPM SI brochure); aircraft.yaml expectations trace to the provenance tags in
that file ([3VIEW] factory drawing, [DXF] measured 3-view DXFs, [VT2005]
Virginia Tech slides, [IMP74] Strausak Impulse #74 article). If a YAML number
is deliberately revised, the corresponding expectation here must be revised
in the same commit (schema v2 expectations below).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from geometry.units import (
    UNIT_TO_SI,
    AircraftParams,
    dynamic_pressure,
    load_aircraft,
    parse_quantity,
    reynolds,
)

# Repo root resolves from this file's location so the suite passes no matter
# which directory pytest is launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_YAML = REPO_ROOT / "aircraft.yaml"


def _q(value: float, unit: str) -> dict:
    """Build a {value, unit} quantity mapping (test shorthand)."""
    return {"value": value, "unit": unit}


# =============================================================================
#  Conversion-factor spot checks — every supported tag, exact definitions
# =============================================================================
@pytest.mark.parametrize(
    "unit, expected_si",
    [
        # length
        ("m", 1.0),
        ("mm", 1.0e-3),
        ("in", 0.0254),                 # international inch, exact
        ("ft", 0.3048),                 # international foot, exact
        # area
        ("m2", 1.0),
        ("ft2", 0.09290304),            # 0.3048^2, exact
        # speed
        ("mps", 1.0),
        ("mph", 0.44704),               # statute mph, exact
        ("kt", 1852.0 / 3600.0),        # international knot, exact
        # angle
        ("deg", math.pi / 180.0),
        ("rad", 1.0),
        # temperature (absolute only)
        ("K", 1.0),
        # mass
        ("kg", 1.0),
        ("lb", 0.45359237),             # avoirdupois pound, exact
        ("slug", 14.59390294),
        # passthrough physical-property tags
        ("kgpm3", 1.0),
        ("m2ps", 1.0),
    ],
)
def test_unit_factor(unit: str, expected_si: float) -> None:
    """A unit quantity of 1 <tag> converts to exactly its SI factor."""
    assert parse_quantity(_q(1.0, unit)) == pytest.approx(expected_si, rel=1e-12)


def test_all_table_tags_covered() -> None:
    """The parametrized factor list above covers the whole UNIT_TO_SI table.

    Guards against someone adding a tag to the table without adding a factor
    spot check — the parametrize list is the authority on expected values.
    """
    tested = {
        "m", "mm", "in", "ft", "m2", "ft2", "mps", "mph", "kt",
        "deg", "rad", "K", "kg", "lb", "slug", "kgpm3", "m2ps",
    }
    assert set(UNIT_TO_SI) == tested


def test_scaling_is_linear() -> None:
    """Factors apply multiplicatively (50 mm -> 0.050 m, 90 deg -> pi/2)."""
    assert parse_quantity(_q(50.0, "mm")) == pytest.approx(0.050, rel=1e-12)
    assert parse_quantity(_q(90.0, "deg")) == pytest.approx(math.pi / 2.0, rel=1e-12)
    # Negative values (e.g. control-surface deflection sign conventions)
    # must scale identically — no abs() lurking anywhere.
    assert parse_quantity(_q(-12.5, "deg")) == pytest.approx(-12.5 * math.pi / 180.0)


# =============================================================================
#  Rejection paths — malformed quantities must fail loudly
# =============================================================================
def test_unknown_unit_rejected() -> None:
    """An unrecognized tag raises ValueError naming the tag."""
    with pytest.raises(ValueError, match="furlong"):
        parse_quantity(_q(1.0, "furlong"))


def test_unknown_unit_names_key_path() -> None:
    """The key-path context threads into the error for YAML debugging."""
    with pytest.raises(ValueError, match=r"wing\.span"):
        parse_quantity(_q(1.0, "cubit"), key_path="wing.span")


def test_unknown_unit_rejected_inside_yaml(tmp_path: Path) -> None:
    """A bad tag deep in a YAML file is reported with its full key path."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "wing:\n  span: {value: 23.0, unit: parsecs}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"parsecs.*wing\.span"):
        load_aircraft(bad)


def test_malformed_quantity_rejected() -> None:
    """Missing keys or non-numeric values are ValueErrors, not garbage."""
    with pytest.raises(ValueError):
        parse_quantity({"value": 1.0})              # no unit key
    with pytest.raises(ValueError):
        parse_quantity(_q("fast", "mps"))           # non-numeric value
    with pytest.raises(ValueError):
        parse_quantity(_q(True, "m"))               # bool is not a length


# =============================================================================
#  load_aircraft on the real master file — values trace to provenance tags
# =============================================================================
@pytest.fixture(scope="module")
def ac() -> AircraftParams:
    """Load the committed aircraft.yaml once for all checks below."""
    return load_aircraft(AIRCRAFT_YAML)


def test_wing_span_si(ac: AircraftParams) -> None:
    """[3VIEW] 23.276 ft -> 7.0945248 m exactly (ft factor is exact)."""
    assert ac.wing.span == pytest.approx(7.0945248, abs=1e-6)


def test_wing_mac_si(ac: AircraftParams) -> None:
    """[VT2005] MAC 3.76 ft -> 1.146048 m."""
    assert ac.wing.mac == pytest.approx(1.146048, abs=1e-6)


def test_dimensionless_passthrough(ac: AircraftParams) -> None:
    """Plain scalars (taper ratio, AR) arrive untouched and untyped-converted.

    Taper is the [DXF] centerline-to-tip value from the schema-v2 rewrite
    (the drawing supersedes VT's 0.75); AR stays the [VT2005] number.
    """
    assert ac.wing.taper_ratio == 0.6031
    assert ac.wing.aspect_ratio == 6.20


def test_string_passthrough(ac: AircraftParams) -> None:
    """Strings such as file references survive verbatim."""
    assert ac.wing.airfoil_file == "geometry/ls413.dat"


def test_vg_spacing_outboard_si(ac: AircraftParams) -> None:
    """[IMP74] 50 mm outboard VG spacing -> 0.050 m."""
    assert ac.vg_defaults.wing.spacing_outboard == pytest.approx(0.050, abs=1e-12)


def test_vane_incidence_radians(ac: AircraftParams) -> None:
    """[IMP74] 15 deg vane incidence arrives in radians."""
    assert ac.vg_defaults.vane_incidence == pytest.approx(0.2617994, abs=1e-6)


def test_togw_si(ac: AircraftParams) -> None:
    """[VT2005] TOGW 2400 lb -> 1088.621688 kg."""
    assert ac.weights.togw == pytest.approx(1088.62, abs=0.01)


def test_none_passthrough(ac: AircraftParams) -> None:
    """Unresolved YAML nulls stay None so downstream code can refuse them.

    vertical_tail.area is deliberately null in schema v2 (curved fin
    planform, deferred until Study 3 digitizes the outline). It is the last
    live null in the master file: solver.openfoam_version was pinned to
    "v2506" at M1, so its None-passthrough coverage moved to the synthetic
    fixture below.
    """
    assert ac.vertical_tail.area is None


def test_none_passthrough_synthetic(tmp_path: Path) -> None:
    """A null in any nested block loads as None (synthetic file).

    Exercised on a throwaway YAML because the master file no longer carries
    a solver-block null; the loader behavior must survive regardless of
    which keys happen to be resolved in aircraft.yaml at any given time.
    """
    mini = tmp_path / "mini.yaml"
    mini.write_text(
        "solver:\n"
        "  openfoam_version: null\n"
        "  turbulence_model: kOmegaSSTLM\n",
        encoding="utf-8",
    )
    ac2 = load_aircraft(mini)
    assert ac2.solver.openfoam_version is None
    # Sibling strings in the same block still pass through verbatim.
    assert ac2.solver.turbulence_model == "kOmegaSSTLM"


def test_solver_version_pinned(ac: AircraftParams) -> None:
    """solver.openfoam_version is the M1-pinned "v2506" string, verbatim.

    The pin is part of the section-4 reproducibility rule (every case must
    be rebuildable from aircraft.yaml + the pinned OpenFOAM version); a
    silent edit back to null or another release must trip the suite.
    """
    assert ac.solver.openfoam_version == "v2506"


def test_cruise_points_list_recursion(ac: AircraftParams) -> None:
    """conditions.cruise_points: list of mappings -> list of param objects
    with quantities converted and dimensionless members passed through."""
    pts = ac.conditions.cruise_points
    assert isinstance(pts, list) and len(pts) == 3
    # Each element is itself an attribute-access node.
    for pt in pts:
        assert isinstance(pt, AircraftParams)
    # First point: 8000 ft -> 2438.4 m, 258 mph -> 115.33632 m/s, CL verbatim.
    assert pts[0].altitude == pytest.approx(2438.4, abs=1e-6)
    assert pts[0].speed == pytest.approx(115.33632, abs=1e-6)
    assert pts[0].cl == 0.222


def test_raw_preserved(ac: AircraftParams) -> None:
    """.raw keeps the original un-converted mapping at every nesting level."""
    # Root raw holds the source quantity mapping for wing.span verbatim.
    assert ac.raw["wing"]["span"] == {"value": 23.276, "unit": "ft"}
    # Nested nodes carry their own slice of the raw tree.
    assert ac.wing.raw["mac"] == {"value": 3.76, "unit": "ft"}


def test_angles_converted_throughout(ac: AircraftParams) -> None:
    """Deeply nested angle quantities are radians too (hinge/deflection math
    downstream assumes it)."""
    assert ac.wing.dihedral == pytest.approx(3.0 * math.pi / 180.0, rel=1e-12)
    assert ac.wing.aileron.deflection_max_up == pytest.approx(
        20.0 * math.pi / 180.0, rel=1e-12
    )


def test_hinge_gap_si(ac: AircraftParams) -> None:
    """[SPEC] 1/16-inch hinge gap stored as {0.0625 in} in schema v2.

    The international inch is exact (0.0254 m), so the SI value is exactly
    0.0625 * 0.0254 = 1.5875e-3 m — no rounded 1.59 mm intermediate. The
    aileron and rudder carry the same factory gap convention; checked here
    too so a single-surface YAML edit cannot drift silently.
    """
    assert ac.horizontal_tail.elevator.hinge_gap == pytest.approx(1.5875e-3, abs=1e-12)
    assert ac.wing.aileron.hinge_gap == pytest.approx(1.5875e-3, abs=1e-12)
    assert ac.vertical_tail.rudder.hinge_gap == pytest.approx(1.5875e-3, abs=1e-12)


# =============================================================================
#  Flow helpers
# =============================================================================
def test_reynolds_formula() -> None:
    """Re = V c / nu, straight SI arithmetic."""
    assert reynolds(10.0, 2.0, 1.0e-5) == pytest.approx(2.0e6, rel=1e-12)


def test_reynolds_on_aircraft_numbers(ac: AircraftParams) -> None:
    """Approach-condition Re on the MAC lands in the expected mid-3M band
    (the value Phase-1 XFOIL/OpenFOAM grids are built around)."""
    re_app = reynolds(
        ac.conditions.approach_speed, ac.wing.mac, ac.atmosphere.kinematic_viscosity
    )
    assert 3.0e6 < re_app < 4.5e6


def test_dynamic_pressure_formula() -> None:
    """q = 0.5 rho V^2 at ISA sea level, V = 10 m/s -> 61.25 Pa."""
    assert dynamic_pressure(1.225, 10.0) == pytest.approx(61.25, rel=1e-12)
