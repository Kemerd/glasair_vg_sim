"""Tests for geometry/dxf_reader.py — DXF planform extraction + YAML fallback.

Synthetic DXFs are generated with ezdxf into pytest tmp_path, dimensioned to
match the aircraft.yaml (schema v2) [DXF]-measured wing chords (root
53.244 in, tip 32.114 in, half-span 11.638 ft = half of the 23.276 ft
[3VIEW] span) but written in METERS in modelspace, matching the reader's
documented unit convention. They are CLEAN single-panel outlines — the shape
read_dxf_planform is specified for. The committed full-sheet 3-views in
geometry/dxf/ are deliberately NOT parsed here; they belong to
scripts/measure_dxf.py, and the reference-gate tests below prove the reader
refuses geometry that deviates from aircraft.yaml the way a full sheet does.

The YAML-fallback test reads the real committed aircraft.yaml through the
geometry.units loader, so it doubles as an integration check of the two
modules' contract (SI floats, radians for angles).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import ezdxf
import pytest

from geometry.dxf_reader import (
    DxfParseError,
    Planform,
    _main,
    list_layers,
    planform_from_yaml,
    read_dxf_planform,
)
from geometry.units import load_aircraft

# =============================================================================
#  Reference geometry (aircraft.yaml v2 wing block, converted by hand here so
#  the tests do not depend on the loader they are cross-checking)
# =============================================================================
FT_TO_M: float = 0.3048                  # international foot, exact
IN_TO_M: float = 0.0254                  # international inch, exact
CHORD_ROOT_M: float = 53.244 * IN_TO_M   # 1.3523976 m  [DXF] centerline chord
CHORD_TIP_M: float = 32.114 * IN_TO_M    # 0.8156956 m  [DXF]
HALF_SPAN_M: float = 11.638 * FT_TO_M    # 3.547262 m  = (23.276 ft)/2 [3VIEW]
SPAN_M: float = 2.0 * HALF_SPAN_M        # 7.094525 m

# Repo root resolved relative to this file so pytest may run from anywhere.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
AIRCRAFT_YAML: Path = REPO_ROOT / "aircraft.yaml"


# =============================================================================
#  DXF builders
# =============================================================================
def _trapezoid_outline(scale_factor: float = 1.0) -> List[Tuple[float, float]]:
    """Closed half-wing outline, x = chordwise / y = spanwise, meters.

    Zero LE sweep (LE on x = 0 at both rows) so the recovered le_sweep_rad
    has a known expected value of exactly zero. scale_factor != 1 fabricates
    a wrong-size drawing for the reference-gate tests (geometry stays
    similar, every length multiplies).
    """
    s = scale_factor
    return [
        (0.0, 0.0),                               # root LE
        (CHORD_ROOT_M * s, 0.0),                  # root TE
        (CHORD_TIP_M * s, HALF_SPAN_M * s),       # tip TE
        (0.0, HALF_SPAN_M * s),                   # tip LE
    ]


def _build_trapezoid_dxf(
    path: Path,
    use_polyline: bool = True,
    add_decoy: bool = True,
    scale_factor: float = 1.0,
    insunits: int = 6,
) -> None:
    """Write a synthetic trapezoid-wing DXF for the round-trip tests.

    The outline lives on layer 'WING'. When add_decoy is set, an unrelated
    border line is placed OUTSIDE the panel (negative y) on layer 'BORDER':
    if the reader's layer filter leaked, that line would shift the spanwise
    minimum and corrupt the root chord — so the round-trip assertions also
    prove the filter works. insunits sets the $INSUNITS header (6 = meters
    by default, 0 = unitless for the assumed-meters CLI print test).
    """
    doc = ezdxf.new("R2010")
    # Declare modelspace units explicitly. Code 6 (meters) is the default
    # contract; code 0 (unitless) exercises the documented assumed-meters
    # path — both resolve to scale 1.0 so the same outline serves both.
    doc.header["$INSUNITS"] = insunits
    doc.layers.add("WING")
    msp = doc.modelspace()

    outline = _trapezoid_outline(scale_factor)
    if use_polyline:
        # Single closed LWPOLYLINE — the common shape of traced CAD outlines.
        msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": "WING"})
    else:
        # Same outline as four discrete LINE entities, exercising the LINE
        # vertex-collection path with shared (exactly coincident) endpoints.
        ring = outline + [outline[0]]
        for start, end in zip(ring[:-1], ring[1:]):
            msp.add_line(start, end, dxfattribs={"layer": "WING"})

    if add_decoy:
        # Off-panel linework that must be excluded by the layer filter.
        doc.layers.add("BORDER")
        msp.add_line((-0.5, -0.5), (5.0, -0.5), dxfattribs={"layer": "BORDER"})

    doc.saveas(path)


# =============================================================================
#  DXF round-trip
# =============================================================================
def test_trapezoid_roundtrip_lwpolyline(tmp_path: Path) -> None:
    """Chords within 1 mm and area within 0.5% through a full write/read."""
    dxf_path = tmp_path / "wing_trapezoid.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=True)

    pf = read_dxf_planform(dxf_path, layer="WING")

    # Chord recovery: 1 mm tolerance per the milestone contract. Exact
    # coordinates went in, so this really tests bookkeeping, not numerics.
    assert abs(pf.chord_at(0.0) - CHORD_ROOT_M) < 1.0e-3
    assert abs(pf.chord_at(1.0) - CHORD_TIP_M) < 1.0e-3

    # Span: the reader doubles the drawn half-span panel.
    assert abs(pf.span_m - SPAN_M) < 2.0e-3

    # Trapezoid area of the full wing, within 0.5% relative.
    expected_area = SPAN_M * (CHORD_ROOT_M + CHORD_TIP_M) / 2.0
    assert abs(pf.area() - expected_area) / expected_area < 0.005

    # The synthetic outline has its LE on x = 0 at both rows -> zero sweep.
    assert pf.le_sweep_rad is not None
    assert abs(pf.le_sweep_rad) < 1.0e-9


def test_trapezoid_roundtrip_line_entities(tmp_path: Path) -> None:
    """Same outline drawn as four LINEs, read without a layer filter."""
    dxf_path = tmp_path / "wing_lines.dxf"
    # No decoy here: with layer=None every entity participates, and the test
    # targets the LINE extraction path plus exact-coincident-endpoint rows.
    _build_trapezoid_dxf(dxf_path, use_polyline=False, add_decoy=False)

    pf = read_dxf_planform(dxf_path, layer=None)
    assert abs(pf.chord_at(0.0) - CHORD_ROOT_M) < 1.0e-3
    assert abs(pf.chord_at(1.0) - CHORD_TIP_M) < 1.0e-3


def test_layer_filter_excludes_decoy(tmp_path: Path) -> None:
    """The BORDER decoy must not contaminate the WING-layer extraction.

    The decoy sits at y = -0.5 m; were it included, the spanwise minimum
    row would move off the root rib and the 'root chord' would become the
    5.5 m decoy extent. Comparing against the true root chord catches that.
    """
    dxf_path = tmp_path / "wing_with_border.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=True)

    pf = read_dxf_planform(dxf_path, layer="WING")
    assert abs(pf.chord_root_m - CHORD_ROOT_M) < 1.0e-3

    # Case-insensitive layer matching (DXF table names are caseless).
    pf_lower = read_dxf_planform(dxf_path, layer="wing")
    assert abs(pf_lower.chord_root_m - CHORD_ROOT_M) < 1.0e-3


def test_list_layers(tmp_path: Path) -> None:
    """Layer inventory exposes the names needed to drive the filter."""
    dxf_path = tmp_path / "wing_layers.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=True)
    layers = list_layers(dxf_path)
    # '0' always exists; WING and BORDER were added by the builder.
    assert "WING" in layers
    assert "BORDER" in layers
    assert "0" in layers


# =============================================================================
#  Failure modes
# =============================================================================
def test_empty_dxf_raises(tmp_path: Path) -> None:
    """A structurally valid DXF with no linework must raise DxfParseError."""
    dxf_path = tmp_path / "empty.dxf"
    ezdxf.new("R2010").saveas(dxf_path)

    with pytest.raises(DxfParseError) as excinfo:
        read_dxf_planform(dxf_path)
    # The message must be actionable: it names what was (not) found.
    assert "no usable" in str(excinfo.value)


def test_wrong_layer_filter_names_available_layers(tmp_path: Path) -> None:
    """Filtering to a nonexistent layer fails AND lists what exists."""
    dxf_path = tmp_path / "wing_misfiltered.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=True)

    with pytest.raises(DxfParseError) as excinfo:
        read_dxf_planform(dxf_path, layer="NO_SUCH_LAYER")
    # Diagnostics should point the user straight at the real layer name.
    assert "WING" in str(excinfo.value)


def test_missing_file_raises(tmp_path: Path) -> None:
    """A nonexistent path surfaces as DxfParseError, not a bare IOError."""
    with pytest.raises(DxfParseError):
        read_dxf_planform(tmp_path / "does_not_exist.dxf")


# =============================================================================
#  Reference sanity gate (single-panel assumption enforcement)
# =============================================================================
def test_reference_gate_trips_on_wrong_scale_sheet(tmp_path: Path) -> None:
    """A panel drawn at 2x scale must trip the gate, not parse 'fine'.

    This stands in for every silent failure of the single-panel assumption:
    the committed full-sheet topview reads as ~2x the true span the same way
    (14.2 m vs 7.09 m), so a 2x synthetic sheet exercises exactly the
    deviation magnitude the gate exists to catch. The reference comes from
    the real aircraft.yaml so this is also an integration test of the
    planform_from_yaml -> read_dxf_planform handoff.
    """
    dxf_path = tmp_path / "wing_2x_sheet.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=False,
                         scale_factor=2.0)
    reference = planform_from_yaml(load_aircraft(AIRCRAFT_YAML))

    with pytest.raises(DxfParseError) as excinfo:
        read_dxf_planform(dxf_path, layer="WING", reference=reference)

    # The message must carry the actual numbers (measured vs reference) and
    # route the user to the supported fallback by name.
    msg = str(excinfo.value)
    assert "sanity gate" in msg
    assert f"{reference.span_m:.4f}" in msg
    assert f"{reference.chord_root_m:.4f}" in msg
    assert "planform_from_yaml" in msg


def test_reference_gate_passes_clean_panel(tmp_path: Path) -> None:
    """A clean 1:1 panel matching aircraft.yaml sails through the gate."""
    dxf_path = tmp_path / "wing_clean.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=True)
    reference = planform_from_yaml(load_aircraft(AIRCRAFT_YAML))

    # No exception expected; the extracted planform must still be the real
    # geometry, untouched by the gate (it only checks, never adjusts).
    pf = read_dxf_planform(dxf_path, layer="WING", reference=reference)
    assert abs(pf.span_m - reference.span_m) < 2.0e-3
    assert abs(pf.chord_root_m - reference.chord_root_m) < 1.0e-3


# =============================================================================
#  CLI behavior (units print + gate wiring)
# =============================================================================
def test_cli_prints_resolved_insunits(tmp_path: Path, capsys) -> None:
    """The CLI summary must state the resolved $INSUNITS scale explicitly."""
    dxf_path = tmp_path / "wing_cli_meters.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=False,
                         insunits=6)

    rc = _main([str(dxf_path)])
    out = capsys.readouterr().out
    # Successful extraction, with the meters resolution named in clear text.
    assert rc == 0
    assert "$INSUNITS 6 (meters)" in out
    assert "scale 1 m per drawing unit" in out


def test_cli_flags_unitless_assumed_meters(tmp_path: Path, capsys) -> None:
    """A unitless drawing must be called out as ASSUMED meters, loudly."""
    dxf_path = tmp_path / "wing_cli_unitless.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=False,
                         insunits=0)

    rc = _main([str(dxf_path)])
    out = capsys.readouterr().out
    # Coordinates are meters either way, so extraction succeeds — but the
    # summary must flag that meters was an ASSUMPTION, not declared units.
    assert rc == 0
    assert "ASSUMED meters" in out


def test_cli_yaml_arms_reference_gate(tmp_path: Path, capsys) -> None:
    """--yaml builds the reference and a wrong-size sheet exits nonzero."""
    dxf_path = tmp_path / "wing_cli_2x.dxf"
    _build_trapezoid_dxf(dxf_path, use_polyline=True, add_decoy=False,
                         scale_factor=2.0)

    rc = _main([str(dxf_path), "--yaml", str(AIRCRAFT_YAML)])
    out = capsys.readouterr().out
    # Gate trips -> error path: nonzero exit, gate named, fallback hinted.
    assert rc == 1
    assert "sanity gate" in out
    assert "planform_from_yaml" in out


# =============================================================================
#  Planform math and validation
# =============================================================================
def test_chord_at_validates_eta() -> None:
    """eta outside [0, 1] (and NaN) must raise ValueError, not extrapolate."""
    pf = Planform(span_m=SPAN_M, chord_root_m=CHORD_ROOT_M, chord_tip_m=CHORD_TIP_M)
    with pytest.raises(ValueError):
        pf.chord_at(-0.01)
    with pytest.raises(ValueError):
        pf.chord_at(1.01)
    with pytest.raises(ValueError):
        pf.chord_at(float("nan"))
    # Boundary stations remain legal and exact.
    assert pf.chord_at(0.0) == pytest.approx(CHORD_ROOT_M)
    assert pf.chord_at(1.0) == pytest.approx(CHORD_TIP_M)
    # Mid-span is the arithmetic mean for a linear taper.
    assert pf.chord_at(0.5) == pytest.approx((CHORD_ROOT_M + CHORD_TIP_M) / 2.0)


def test_rectangle_area_and_mac() -> None:
    """Degenerate taper = 1 sanity: S = b*c and MAC = c exactly."""
    pf = Planform(span_m=10.0, chord_root_m=2.0, chord_tip_m=2.0)
    assert pf.area() == pytest.approx(20.0)
    assert pf.mac() == pytest.approx(2.0)
    assert pf.taper_ratio == pytest.approx(1.0)


def test_planform_rejects_degenerate_dimensions() -> None:
    """Constructor validation: non-positive span/root chord must fail."""
    with pytest.raises(ValueError):
        Planform(span_m=0.0, chord_root_m=1.0, chord_tip_m=0.5)
    with pytest.raises(ValueError):
        Planform(span_m=7.0, chord_root_m=0.0, chord_tip_m=0.5)
    with pytest.raises(ValueError):
        Planform(span_m=7.0, chord_root_m=1.0, chord_tip_m=-0.1)


# =============================================================================
#  YAML fallback against the real aircraft.yaml (schema v2, [DXF] chords)
# =============================================================================
def test_planform_from_yaml_real_file() -> None:
    """Fallback path reproduces the [DXF]-measured Glasair III wing numbers.

    Expected values are the aircraft.yaml v2 entries converted to SI:
      chord_root 53.244 in -> 1.3523976 m, chord_tip 32.114 in -> 0.8156956 m,
      span 23.276 ft -> 7.0945248 m, sweep_le 1.943 deg aft.

    MAC: the trapezoid formula on these chords gives ~1.1062 m, which sits
    ~3.5% BELOW the published [VT2005] 3.76 ft = 1.146 m kept in aircraft.yaml
    as the Re/moment reference. That offset is definitional, not an error —
    the published MAC includes fuselage carryover while the [DXF] trapezoid
    runs straight centerline -> tip; both values and the explanation are
    recorded in the aircraft.yaml comments. The assertion therefore targets
    the TRAPEZOID value and separately pins the published-value offset band.
    """
    ac = load_aircraft(AIRCRAFT_YAML)
    pf = planform_from_yaml(ac)

    # Chords and span to sub-millimeter (pure unit conversion, no modeling).
    assert pf.chord_root_m == pytest.approx(1.3523976, abs=5.0e-4)
    assert pf.chord_tip_m == pytest.approx(0.8156956, abs=5.0e-4)
    assert pf.span_m == pytest.approx(7.0945248, abs=5.0e-4)

    # Taper must reproduce the [DXF] centerline->tip value in the YAML.
    assert pf.taper_ratio == pytest.approx(0.6031, abs=5.0e-4)

    # Trapezoid MAC for the measured chords: hand value 1.1062 m (43.55 in,
    # also quoted in the aircraft.yaml mac comment). Tight 1 mm band — this
    # is closed-form arithmetic on exact unit conversions.
    assert pf.mac() == pytest.approx(1.1062, abs=1.0e-3)

    # Definitional offset vs the published 3.76 ft MAC: ~3.5% low, bracketed
    # so a regression in either direction (formula or YAML edit) shows up.
    published_mac_m = 3.76 * FT_TO_M
    mac_offset = (published_mac_m - pf.mac()) / published_mac_m
    assert 0.02 < mac_offset < 0.05

    # Trapezoid area should land on the ~82.8 ft2 straight-line integration
    # documented in the aircraft.yaml area_ref comment (NOT the 87.6 ft2
    # force-coefficient S_ref, which keeps VT's different reference
    # convention on purpose).
    dxf_trapezoid_area_m2 = 82.8 * FT_TO_M ** 2
    assert abs(pf.area() - dxf_trapezoid_area_m2) / dxf_trapezoid_area_m2 < 0.01

    # LE sweep is 1.943 deg aft per [DXF] and must arrive in radians.
    assert pf.le_sweep_rad == pytest.approx(math.radians(1.943), abs=1.0e-9)

    # chord_at must interpolate between exactly those converted chords.
    assert pf.chord_at(0.0) == pytest.approx(pf.chord_root_m)
    assert pf.chord_at(1.0) == pytest.approx(pf.chord_tip_m)
