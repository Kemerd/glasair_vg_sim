"""Unit tests for the Phase-1 2D validation case builder (M1).

Covers the contracts the milestone gates on:

  * C-grid blockMeshDict generator: vertex/block counts of the 5-block
    topology, first wall-normal cell within 2% of the y+ correlation chain
    (scripts/first_cell_height.py -- the generator must REUSE it, so the
    test recomputes the chain independently and compares), every cell-to-
    cell growth ratio <= 1.2, far field >= 25 chords, >= 30 cells inside
    the boundary-layer estimate, sqrt(2) refinement-level scaling.
  * Template instantiation: every required case file is produced and no
    @TOKEN@ placeholder survives into the instantiated tree; Allrun stays
    LF-only (it runs under WSL bash).
  * AoA frame rotation: liftDir/dragDir orthonormal and correct at 0 and
    10 degrees, with the freestream vector along dragDir.
  * Inlet turbulence chain: Mack inversion of Ncrit = 9 lands at the
    documented Tu ~ 0.07% and the Langtry-Menter correlation at that Tu
    lands at the hand-computed Re_theta_t.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from geometry.airfoil import load_airfoil, resample_airfoil
from geometry.units import load_aircraft
from scripts.build_validation_case import (
    BL_STATIONS,
    CHORD,
    FARFIELD_RADIUS,
    GROWTH_CAP,
    MIN_BL_LAYERS,
    N_AIRFOIL_POINTS,
    TOKEN_RE,
    WAKE_LENGTH,
    aoa_directions,
    build_case,
    case_name,
    generate_blockmeshdict,
    inlet_turbulence,
    langtry_rethetat,
    mack_tu_from_ncrit,
    plan_cgrid,
    sampling_set_entries,
)
from scripts.first_cell_height import first_cell_height

# Resolve repo inputs relative to this file so the suite passes regardless
# of pytest's working directory (same convention as the M0 test modules).
REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "aircraft.yaml"

# Reference build point for most tests: the same condition as the committed
# example case (AoA 4, Re 3e6, level 0).
RE_REF = 3.0e6


# -----------------------------------------------------------------------------
#  Shared fixtures (module scope: planning and text generation are pure)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def atmo():
    """SI atmosphere numbers straight from aircraft.yaml."""
    ac = load_aircraft(YAML_PATH)
    return {
        "nu": ac.atmosphere.kinematic_viscosity,
        "rho": ac.atmosphere.density,
        "T": ac.atmosphere.temperature,
    }


@pytest.fixture(scope="module")
def coords():
    """The same blunt-TE resample the builder meshes."""
    ac = load_aircraft(YAML_PATH)
    raw = load_airfoil(REPO_ROOT / ac.wing.airfoil_file)
    return resample_airfoil(raw, n_points=N_AIRFOIL_POINTS, te="blunt")


@pytest.fixture(scope="module")
def plan0(atmo):
    """Level-0 mesh plan at the reference Re."""
    return plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], level=0)


@pytest.fixture(scope="module")
def bmd_text(coords, plan0):
    """Generated blockMeshDict text for the reference plan."""
    return generate_blockmeshdict(coords, plan0)


def _parse_vertices(text: str) -> np.ndarray:
    """Extract the vertex coordinate triples from a blockMeshDict body."""
    # Isolate the vertices(...) section so vectors inside edges/blocks do
    # not pollute the count.
    m = re.search(r"vertices\s*\((.*?)\n\);", text, flags=re.S)
    assert m, "vertices section not found"
    triples = re.findall(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        m.group(1),
    )
    return np.array(triples, dtype=float)


# -----------------------------------------------------------------------------
#  blockMeshDict generator: topology and spec-rule audits
# -----------------------------------------------------------------------------

class TestCGridGenerator:
    def test_vertex_count_is_two_z_planes_of_ten(self, bmd_text):
        # 5-block C-topology: 10 planar points extruded to 2 z-planes.
        verts = _parse_vertices(bmd_text)
        assert verts.shape == (20, 3)

    def test_block_count_is_five(self, bmd_text):
        assert len(re.findall(r"\bhex\s*\(", bmd_text)) == 5

    def test_far_field_at_least_25_chords(self, bmd_text):
        verts = _parse_vertices(bmd_text)
        # Upstream: the C-arc is centered on the LE, so min x = -25 c.
        assert verts[:, 0].min() <= -FARFIELD_RADIUS * CHORD + 1e-9
        # Above/below: straight far boundaries at +/- 25 c.
        assert verts[:, 1].max() >= FARFIELD_RADIUS * CHORD - 1e-9
        assert verts[:, 1].min() <= -FARFIELD_RADIUS * CHORD + 1e-9
        # Downstream: wake block ends 25 c behind the TE (x = 26 c).
        assert verts[:, 0].max() >= (CHORD + WAKE_LENGTH) - 1e-9

    def test_first_cell_matches_correlation_within_2pct(self, plan0, atmo):
        # Independent recomputation of the correlation chain at the series
        # convention U = Re * nu / c, then back out the as-meshed first
        # cell from the numbers the dictionary is rendered with (count +
        # exact ratio over the 25 c wall-normal run).
        u = RE_REF * atmo["nu"] / CHORD
        y1_ref = first_cell_height(u, CHORD, atmo["rho"], atmo["nu"],
                                   y_plus=1.0)["y1"]
        n, r = plan0.normal.n, plan0.normal.ratio
        h1_meshed = FARFIELD_RADIUS * (r - 1.0) / (r ** n - 1.0)
        assert h1_meshed == pytest.approx(y1_ref, rel=0.02)
        # The grading value written into the dictionary must agree with the
        # ratio the audit above was run on.
        assert plan0.normal.grading == pytest.approx(r ** (n - 1), rel=1e-9)

    def test_all_growth_ratios_capped_at_1p2(self, plan0):
        eps = 1e-9
        assert plan0.normal.ratio <= GROWTH_CAP + eps
        assert plan0.wake.ratio <= GROWTH_CAP + eps
        for zone in plan0.zones:
            # Zone ratios are stored as magnitudes >= 1 for exactly this audit.
            assert zone.ratio <= GROWTH_CAP + eps

    def test_at_least_30_layers_inside_boundary_layer(self, plan0):
        assert plan0.layers_in_d99 >= MIN_BL_LAYERS

    def test_wake_first_cell_continues_te_spacing(self, plan0):
        # No size jump across the surface/wake block interface.
        assert plan0.wake.first == pytest.approx(plan0.d_te, rel=1e-9)

    def test_refinement_levels_scale_by_sqrt2(self, atmo, plan0):
        plan1 = plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], 1)
        plan2 = plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], 2)
        s = math.sqrt(2.0)
        # Surface counts step by sqrt(2) (integer rounding allowed)...
        assert plan1.n_surf == pytest.approx(plan0.n_surf * s, rel=0.02)
        assert plan2.n_surf == pytest.approx(plan0.n_surf * 2.0, rel=0.02)
        # ...and the first cell shrinks by the same factor: true uniform
        # refinement, the premise of the GCI study.
        assert plan1.h1 == pytest.approx(plan0.h1 / s, rel=1e-9)
        assert plan2.h1 == pytest.approx(plan0.h1 / 2.0, rel=1e-9)
        # Refinement must never relax the spec rules.
        assert plan2.normal.ratio <= GROWTH_CAP + 1e-9
        assert plan2.layers_in_d99 >= MIN_BL_LAYERS
        # Cell budget grows roughly 2x per level (2D refinement).
        assert plan1.n_cells_total > 1.6 * plan0.n_cells_total
        assert plan2.n_cells_total > 1.6 * plan1.n_cells_total

    def test_boundary_patches_present(self, bmd_text):
        for patch in ("airfoil", "farfield", "outlet", "frontAndBack"):
            assert patch in bmd_text
        # 2D convention: the z faces must be 'empty'.
        assert "type empty;" in bmd_text

    def test_surface_uses_spline_edges(self, bmd_text):
        # 2 surfaces x 2 z-planes of spline edges, plus 4 far polyLines.
        assert len(re.findall(r"^\s*spline\s", bmd_text, flags=re.M)) == 4
        assert len(re.findall(r"^\s*polyLine\s", bmd_text, flags=re.M)) == 4


# -----------------------------------------------------------------------------
#  AoA frame rotation
# -----------------------------------------------------------------------------

class TestAoaRotation:
    def test_zero_aoa_axes(self):
        d = aoa_directions(0.0)
        assert d["drag"] == pytest.approx((1.0, 0.0, 0.0))
        assert d["lift"] == pytest.approx((0.0, 1.0, 0.0))

    def test_ten_degrees(self):
        a = math.radians(10.0)
        d = aoa_directions(a)
        assert d["drag"] == pytest.approx((math.cos(a), math.sin(a), 0.0))
        assert d["lift"] == pytest.approx((-math.sin(a), math.cos(a), 0.0))

    @pytest.mark.parametrize("deg", [-4.0, 0.0, 4.0, 10.0, 20.0])
    def test_orthonormal_right_handed(self, deg):
        d = aoa_directions(math.radians(deg))
        drag = np.array(d["drag"])
        lift = np.array(d["lift"])
        # Unit vectors, mutually orthogonal...
        assert np.linalg.norm(drag) == pytest.approx(1.0, abs=1e-12)
        assert np.linalg.norm(lift) == pytest.approx(1.0, abs=1e-12)
        assert float(drag @ lift) == pytest.approx(0.0, abs=1e-12)
        # ...and right-handed: drag x lift = +z (lift is 90 deg CCW).
        assert np.cross(drag, lift)[2] == pytest.approx(1.0, abs=1e-12)


# -----------------------------------------------------------------------------
#  Inlet turbulence chain (Mack / Langtry-Menter)
# -----------------------------------------------------------------------------

class TestInletTurbulence:
    def test_mack_inversion_at_ncrit_9(self):
        # Hand-derived anchor: Tu = exp(-(8.43 + 9)/2.4) = 7.0e-4 = 0.070%.
        tu = mack_tu_from_ncrit(9.0)
        assert tu == pytest.approx(7.0e-4, rel=0.01)
        # Round trip through Mack's relation recovers Ncrit.
        assert -8.43 - 2.4 * math.log(tu) == pytest.approx(9.0, abs=1e-9)

    def test_langtry_correlation_at_mack_tu(self):
        # Hand-computed: 1173.51 - 589.428*0.070 + 0.2196/0.070^2 ~ 1177.
        tu_pct = 100.0 * mack_tu_from_ncrit(9.0)
        assert langtry_rethetat(tu_pct) == pytest.approx(1177.0, abs=8.0)

    def test_langtry_floor(self):
        # The published correlation floors at 20 for extreme Tu.
        assert langtry_rethetat(200.0) == 20.0

    def test_chain_consistency(self, atmo):
        u = RE_REF * atmo["nu"] / CHORD
        t = inlet_turbulence(u, atmo["nu"])
        # k from the isotropic definition at the Mack Tu...
        assert t["k"] == pytest.approx(1.5 * (t["tu"] * u) ** 2, rel=1e-12)
        # ...and omega from the documented nut/nu = 10 inlet ratio.
        assert t["k"] / (t["omega"] * atmo["nu"]) == pytest.approx(10.0,
                                                                   rel=1e-9)


# -----------------------------------------------------------------------------
#  Sampling lines (suction-surface BL stations)
# -----------------------------------------------------------------------------

class TestSamplingLines:
    def test_one_entry_per_station_upper_surface(self, coords):
        text = sampling_set_entries(coords, math.radians(4.0))
        # Set names follow the extract_bl contract: 'bl_x<percent digits>'
        # ('bl_x007' = 7% chord), one entry per Phase-1 station.
        names = re.findall(r"\bbl_x\d{3}\b", text)
        assert len(names) == len(BL_STATIONS)
        # Positive AoA: suction side is the UPPER surface; the wall-normal
        # lines must point upward (end y above the surface anchor y).
        assert "upper (suction) surface" in text

    def test_set_names_round_trip_through_extract_bl(self, coords):
        # Cross-module contract lock: every set name the builder emits must
        # decode back to its station through the BL-extraction parser (the
        # consumer of these sample files), or the whole VG-sizing table
        # would silently come out empty/mis-stationed.
        from scripts.extract_bl import parse_station_token
        text = sampling_set_entries(coords, math.radians(4.0))
        names = re.findall(r"\bbl_x\d{3}\b", text)
        decoded = sorted(parse_station_token(f"{n}_U") for n in names)
        assert decoded == [pytest.approx(s, abs=1e-9) for s in BL_STATIONS]

    def test_negative_aoa_uses_lower_surface(self, coords):
        text = sampling_set_entries(coords, math.radians(-4.0))
        assert "lower (suction) surface" in text


# -----------------------------------------------------------------------------
#  Template instantiation end to end (dry-run: no OpenFOAM involved)
# -----------------------------------------------------------------------------

# Every file a runnable case must carry; missing any is a build failure.
REQUIRED_FILES = [
    "0/U", "0/p", "0/k", "0/omega", "0/nut", "0/gammaInt", "0/ReThetat",
    "constant/transportProperties", "constant/turbulenceProperties",
    "system/blockMeshDict", "system/controlDict", "system/decomposeParDict",
    "system/forceCoeffs", "system/fvSchemes", "system/fvSolution",
    "system/residuals", "system/sampling", "system/yPlusObj",
    "README.md", "Allrun",
]


class TestCaseInstantiation:
    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory):
        """One full case build into a throwaway root, shared by the class."""
        out_root = tmp_path_factory.mktemp("runs")
        case_dir, plan = build_case(4.0, RE_REF, 0, YAML_PATH, out_root)
        return case_dir, plan

    def test_case_directory_name(self, built):
        case_dir, _ = built
        assert case_dir.name == "val2d_aoa4_re3e6_lvl0"
        assert case_name(-2.5, 1.5e6, 1) == "val2d_aoam2p5_re1.5e6_lvl1"

    def test_every_required_file_exists(self, built):
        case_dir, _ = built
        for rel in REQUIRED_FILES:
            assert (case_dir / rel).is_file(), f"missing {rel}"

    def test_no_unresolved_tokens_anywhere(self, built):
        case_dir, _ = built
        for path in sorted(case_dir.rglob("*")):
            if not path.is_file():
                continue
            leftovers = TOKEN_RE.findall(path.read_text(encoding="utf-8"))
            assert not leftovers, f"{path.name}: unresolved {leftovers}"

    def test_blockmesh_stub_was_replaced(self, built):
        case_dir, _ = built
        text = (case_dir / "system" / "blockMeshDict").read_text(
            encoding="utf-8")
        # The template stub carries a poison token and no blocks; the
        # generated dictionary must have real content and no poison.
        assert "FatalError" not in text
        assert "blocks" in text and "spline" in text

    def test_velocity_vector_matches_aoa(self, built):
        case_dir, plan = built
        a = math.radians(4.0)
        text = (case_dir / "0" / "U").read_text(encoding="utf-8")
        m = re.search(r"internalField\s+uniform\s+\(([-+0-9.eE]+)\s+"
                      r"([-+0-9.eE]+)\s+0\)", text)
        assert m, "internalField vector not found in 0/U"
        ux, uy = float(m.group(1)), float(m.group(2))
        # Vector along dragDir with magnitude U = Re * nu / c.
        assert ux == pytest.approx(plan.u_inf * math.cos(a), rel=1e-6)
        assert uy == pytest.approx(plan.u_inf * math.sin(a), rel=1e-6)
        assert math.hypot(ux, uy) == pytest.approx(plan.u_inf, rel=1e-6)

    def test_force_coeffs_directions_filled(self, built):
        case_dir, _ = built
        a = math.radians(4.0)
        text = (case_dir / "system" / "forceCoeffs").read_text(
            encoding="utf-8")
        # liftDir/dragDir rotated into the wind frame for this AoA.
        assert f"({math.cos(a):.8g} {math.sin(a):.8g} 0)" in text
        assert f"({-math.sin(a):.8g} {math.cos(a):.8g} 0)" in text

    def test_allrun_is_lf_only(self, built):
        case_dir, _ = built
        raw = (case_dir / "Allrun").read_bytes()
        # bash under WSL rejects CRLF scripts; the builder pins LF.
        assert b"\r" not in raw

    def test_rebuild_requires_force(self, built):
        case_dir, _ = built
        with pytest.raises(FileExistsError):
            build_case(4.0, RE_REF, 0, YAML_PATH, case_dir.parent)
        # And --force succeeds over the existing tree.
        rebuilt, _ = build_case(4.0, RE_REF, 0, YAML_PATH, case_dir.parent,
                                force=True)
        assert (rebuilt / "system" / "controlDict").is_file()
