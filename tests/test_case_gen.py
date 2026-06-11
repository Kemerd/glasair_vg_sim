"""Unit tests for the Phase-1 2D validation case builder (M1).

Covers the contracts the milestone gates on:

  * C-grid blockMeshDict generator: vertex/block counts of the 5-block
    topology, first wall-normal cell within 2% of the y+ correlation chain
    (scripts/first_cell_height.py -- the generator must REUSE it, so the
    test recomputes the chain independently and compares), every cell-to-
    cell growth ratio <= 1.2, far field >= 25 chords, >= 30 cells inside
    the boundary-layer estimate, sqrt(2) refinement-level scaling.
  * Blunt-base slab grading: the W_m transverse direction must meet the
    W_u/W_l wake-cut corners at the y+ = 1 first cell (edge cell within 5%
    of h1, ratio <= 1.2) at every (Re, level) combination -- the regression
    for the 57x..218x corner size jump the uniform slab used to carry.
  * Template instantiation: every required case file is produced and no
    @TOKEN@ placeholder survives into the instantiated tree; Allrun stays
    LF-only (it runs under WSL bash); decomposition ranks scale with the
    refinement level; gradSchemes limits grad(U) only.
  * AoA frame rotation: liftDir/dragDir orthonormal and correct at 0 and
    10 degrees, with the freestream vector along dragDir.
  * Inlet turbulence chain: Mack inversion of Ncrit = 9 lands at the
    documented Tu ~ 0.07% and the Langtry-Menter correlation at that Tu
    lands at the hand-computed Re_theta_t; the boundary k is pre-boosted by
    the analytic freestream-decay factor so the LE-incident Tu round-trips
    to the Mack value exactly.
  * BL sampling: two lines per station (dense inner sublayer line + the
    0.15c outer line), names round-trip through the extract_bl parser, and
    a synthetic inner/outer pair MERGES into one profile whose integrals
    recover the analytic sine-profile values.
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
    BETA_STAR,
    BL_INNER_D99_FACTOR,
    BL_INNER_SPACING,
    BL_STATIONS,
    CHORD,
    FARFIELD_RADIUS,
    GROWTH_CAP,
    MIN_BL_LAYERS,
    N_AIRFOIL_POINTS,
    TOKEN_RE,
    WAKE_LENGTH,
    _split_loop,
    aoa_directions,
    build_case,
    case_name,
    generate_blockmeshdict,
    inlet_turbulence,
    langtry_rethetat,
    mack_tu_from_ncrit,
    plan_cgrid,
    sampling_set_entries,
    subdomains_for_level,
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
def base_h(coords):
    """Blunt-TE base opening measured from the resampled loop (~0.0055c),
    exactly as build_case measures it before planning the grid."""
    upper, lower = _split_loop(coords)
    return float(upper[-1, 1] - lower[-1, 1])


@pytest.fixture(scope="module")
def plan0(atmo, base_h):
    """Level-0 mesh plan at the reference Re."""
    return plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], level=0,
                      base_height=base_h)


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

    def test_refinement_levels_scale_by_sqrt2(self, atmo, base_h, plan0):
        plan1 = plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], 1,
                           base_height=base_h)
        plan2 = plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], 2,
                           base_height=base_h)
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

    def test_wm_base_multigrading_rendered(self, bmd_text, plan0):
        # The W_m transverse direction must carry the symmetric two-zone
        # multigrading verbatim: half the base / half the cells growing from
        # the lower corner, mirrored (inverse expansion) into the upper one.
        g = f"{plan0.base.grading:.10g}"
        g_inv = f"{1.0 / plan0.base.grading:.10g}"
        expected = (f"( (0.5 {plan0.base.n} {g}) "
                    f"(0.5 {plan0.base.n} {g_inv}) )")
        assert expected in bmd_text
        # The old uniform-slab grading must be gone: no '( g_wake 1 1 )'.
        assert f"( {plan0.wake.grading:.10g} 1 1 )" not in bmd_text


# -----------------------------------------------------------------------------
#  Blunt-base slab grading (W_m transverse direction)
# -----------------------------------------------------------------------------
#  Regression for the wake-slab discontinuity: a uniform run across the
#  0.0055c base met the W_u/W_l y+ = 1 first cell with a 57x (Re 1.5M) to
#  218x (Re 6M) wall-normal size jump on the shared wake-cut faces. The
#  symmetric two-zone grading must land the edge cell ON h1 (within 5%)
#  with the cell-to-cell ratio capped at 1.2 -- at EVERY (Re, level) combo
#  of the sweep, since both h1 and the level scaling move the solve.

class TestBaseSlabGrading:
    @pytest.mark.parametrize("re_t", [1.5e6, 3.0e6, 6.0e6])
    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_edge_cell_matches_h1_within_5pct_and_cap(self, atmo, base_h,
                                                      re_t, level):
        plan = plan_cgrid(re_t, atmo["nu"], atmo["rho"], atmo["T"], level,
                          base_height=base_h)
        b = plan.base
        # Spec growth-ratio cap holds on the base run too.
        assert b.ratio <= GROWTH_CAP + 1e-9
        # Edge cell RECOMPUTED from the rendered numbers (count + ratio over
        # the half base) -- the audit must not trust b.first, it re-derives
        # the size the dictionary actually produces at the corner.
        if b.ratio > 1.0 + 1e-12:
            h_edge = b.length * (b.ratio - 1.0) / (b.ratio ** b.n - 1.0)
        else:
            h_edge = b.length / b.n
        assert h_edge == pytest.approx(plan.h1, rel=0.05)
        # Symmetric halves: the rendered total is exactly two half-runs.
        assert plan.n_base == 2 * b.n
        assert b.length == pytest.approx(0.5 * base_h, rel=1e-12)

    def test_reference_cell_budget(self, atmo, base_h):
        # The documented budget point: ~46 cells across the base at the
        # reference condition (Re 3M, level 0) -- a ~+7% whole-mesh cost.
        plan = plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], 0,
                          base_height=base_h)
        assert 40 <= plan.n_base <= 52

    def test_mismatched_base_height_is_rejected(self, coords, atmo, base_h):
        # generate_blockmeshdict must refuse a plan solved for a different
        # base opening than the loop being meshed (silent corner mismatch).
        plan = plan_cgrid(RE_REF, atmo["nu"], atmo["rho"], atmo["T"], 0,
                          base_height=2.0 * base_h)
        with pytest.raises(ValueError, match="base height"):
            generate_blockmeshdict(coords, plan)


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
        # LE-incident TARGET pair: isotropic k at the Mack Tu, and omega
        # from the documented nut/nu = 10 ratio at that target state.
        assert t["k_target"] == pytest.approx(1.5 * (t["tu"] * u) ** 2,
                                              rel=1e-12)
        assert t["k_target"] / (t["omega_target"] * atmo["nu"]) == \
            pytest.approx(10.0, rel=1e-9)
        # Analytic freestream-decay factor over the 25c approach, evaluated
        # at the target-state omega (recomputed here independently).
        decay = math.exp(-BETA_STAR * t["omega_target"]
                         * FARFIELD_RADIUS * CHORD / u)
        assert t["decay"] == pytest.approx(decay, rel=1e-12)
        assert 0.0 < t["decay"] < 1.0
        # BOUNDARY pair: k pre-boosted by the inverse decay factor, with the
        # nut/nu = 10 ratio re-held on the boosted value.
        assert t["k"] == pytest.approx(t["k_target"] / decay, rel=1e-12)
        assert t["k"] / (t["omega"] * atmo["nu"]) == pytest.approx(10.0,
                                                                   rel=1e-9)
        # Sanity on the magnitude at Re 3M: the boost is a ~1.6x factor
        # (decay ~0.61), i.e. material -- skipping the compensation would
        # have the LE see ~0.78x the target intensity.
        assert 1.0 / t["decay"] == pytest.approx(1.64, abs=0.05)

    @pytest.mark.parametrize("re_t", [1.5e6, 3.0e6, 6.0e6])
    def test_decayed_le_tu_round_trips_to_mack(self, atmo, re_t):
        # The point of the compensation: marching the boundary k through the
        # analytic decay lands the LE-incident Tu on the Mack Ncrit-9 value
        # EXACTLY (within the constant-omega decay model), at every sweep Re.
        u = re_t * atmo["nu"] / CHORD
        t = inlet_turbulence(u, atmo["nu"])
        k_le = t["k"] * t["decay"]
        tu_le = math.sqrt(2.0 * k_le / 3.0) / u
        assert tu_le == pytest.approx(mack_tu_from_ncrit(9.0), rel=1e-9)


# -----------------------------------------------------------------------------
#  Sampling lines (suction-surface BL stations)
# -----------------------------------------------------------------------------

class TestSamplingLines:
    def test_two_entries_per_station_upper_surface(self, coords, plan0):
        text = sampling_set_entries(coords, math.radians(4.0), plan0.d99)
        # Set names follow the extract_bl contract: 'bl_x<percent digits>'
        # ('bl_x007' = 7% chord) for the outer line plus a '_inner' sibling
        # per station. (\b does not fire between digit and underscore, so
        # the outer pattern cannot accidentally count the inner names.)
        outer = re.findall(r"\bbl_x\d{3}\b", text)
        inner = re.findall(r"\bbl_x\d{3}_inner\b", text)
        assert len(outer) == len(BL_STATIONS)
        assert len(inner) == len(BL_STATIONS)
        # Positive AoA: suction side is the UPPER surface; the wall-normal
        # lines must point upward (end y above the surface anchor y).
        assert "upper (suction) surface" in text

    def test_inner_line_length_and_pitch(self, coords, plan0):
        # The inner line exists to sample the resolved sublayer: ~2x the
        # d99 estimate long at ~1e-5 m pitch (vs 2.5e-4 m on the outer line,
        # which steps clean over the ~8e-6 m first cell).
        text = sampling_set_entries(coords, math.radians(4.0), plan0.d99)
        m = re.search(
            r"bl_x010_inner\s*\{.*?start\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
            r"[-+0-9.eE]+\);.*?end\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
            r"[-+0-9.eE]+\);.*?nPoints\s+(\d+);",
            text, flags=re.S)
        assert m, "bl_x010_inner entry not found"
        sx, sy, ex, ey = (float(m.group(i)) for i in range(1, 5))
        npts = int(m.group(5))
        length = math.hypot(ex - sx, ey - sy)
        # Length: the requested multiple of the d99 estimate (the 1e-6 m
        # wall offset on the start point is negligible at this scale).
        assert length == pytest.approx(BL_INNER_D99_FACTOR * plan0.d99,
                                       rel=0.001)
        # Pitch: the uniform spacing the point count realizes.
        assert length / (npts - 1) == pytest.approx(BL_INNER_SPACING,
                                                    rel=0.02)

    def test_set_names_round_trip_through_extract_bl(self, coords, plan0):
        # Cross-module contract lock: every set name the builder emits --
        # inner AND outer -- must decode back to its station through the
        # BL-extraction parser (the consumer of these sample files), or the
        # whole VG-sizing table would silently come out empty/mis-stationed.
        from scripts.extract_bl import parse_station_token
        text = sampling_set_entries(coords, math.radians(4.0), plan0.d99)
        outer = re.findall(r"\bbl_x\d{3}\b", text)
        inner = re.findall(r"\bbl_x\d{3}_inner\b", text)
        for names in (outer, inner):
            decoded = sorted(parse_station_token(f"{n}_U") for n in names)
            assert decoded == [pytest.approx(s, abs=1e-9)
                               for s in BL_STATIONS]

    def test_negative_aoa_uses_lower_surface(self, coords, plan0):
        text = sampling_set_entries(coords, math.radians(-4.0), plan0.d99)
        assert "lower (suction) surface" in text

    def test_inner_outer_profiles_merge_in_extract_bl(self, tmp_path):
        """Round trip of the two-line scheme through the extract_bl reader.

        Fabricates the sine profile (closed-form delta99/delta*/theta) the
        way the case writes it: a COARSE outer line (600 points over 0.15c,
        2.5e-4 m pitch -- the resolved sublayer falls entirely between its
        first two samples) plus a DENSE inner line (1e-5 m pitch to 2x
        delta). The reader must merge both into one profile and recover the
        analytic integrals; either line alone caps n_points well below the
        merged count, so the count also proves the merge actually happened.
        """
        from scripts.extract_bl import extract_case
        delta, u_e = 0.008, 43.8
        sine = lambda n: np.where(n < delta,
                                  u_e * np.sin(0.5 * np.pi * n / delta), u_e)

        def raw_xy(n):
            u = sine(n)
            return "\n".join(f"{ni:.8e}\t{ui:.8e}\t0.0\t0.0"
                             for ni, ui in zip(n, u)) + "\n"

        n_outer = np.linspace(1.0e-6, 0.15, 600)          # case outer line
        n_inner = np.arange(1.0e-6, 2.0 * delta, 1.0e-5)  # case inner line
        fo = tmp_path / "aoa_p04" / "postProcessing" / "blProfiles" / "2000"
        fo.mkdir(parents=True)
        (fo / "bl_x010_U.xy").write_text(raw_xy(n_outer), encoding="utf-8")
        (fo / "bl_x010_inner_U.xy").write_text(raw_xy(n_inner),
                                               encoding="utf-8")

        stations = extract_case(tmp_path / "aoa_p04")
        assert len(stations) == 1
        st = stations[0]
        assert st.x_over_c == pytest.approx(0.10)
        # Both files merged: count exceeds what either line alone provides
        # (inner ~1600 + outer 600 + wall anchor), and the provenance lists
        # the pair with the outer line as the primary source.
        assert st.metrics.n_points > 2000
        assert len(st.sources) == 2
        assert st.source.name == "bl_x010_U.xy"
        # The merged profile recovers the analytic sine-profile integrals
        # (same 1% bar as the extract_bl unit tests).
        assert st.metrics.delta99 == pytest.approx(
            (2.0 / math.pi) * math.asin(0.99) * delta, rel=0.01)
        assert st.metrics.delta_star == pytest.approx(
            (1.0 - 2.0 / math.pi) * delta, rel=0.01)
        assert st.metrics.theta == pytest.approx(
            (2.0 / math.pi - 0.5) * delta, rel=0.01)


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

    def test_decompose_ranks_scale_with_level(self, built):
        # The dictionary value is the single authority Allrun reads back
        # with foamDictionary; a level-0 build must carry 2 ranks.
        case_dir, _ = built
        text = (case_dir / "system" / "decomposeParDict").read_text(
            encoding="utf-8")
        assert "numberOfSubdomains 2;" in text
        assert "scotch;" in text
        # The mapping itself: 2/4/8 at levels 0/1/2, nothing else accepted.
        assert [subdomains_for_level(lv) for lv in (0, 1, 2)] == [2, 4, 8]
        with pytest.raises(ValueError):
            subdomains_for_level(3)

    def test_fvschemes_limits_grad_u_only(self, built):
        # Gradient limiting is confined to grad(U); the default gradient
        # stays plain Gauss linear so the transition scalars keep full
        # second-order behaviour (the quantity this phase validates on).
        case_dir, _ = built
        text = (case_dir / "system" / "fvSchemes").read_text(encoding="utf-8")
        assert re.search(r"default\s+Gauss linear;", text)
        assert re.search(r"grad\(U\)\s+cellLimited Gauss linear 1;", text)
        # The pseudo-transient fallback variant must mirror the same split
        # (its header promises identity outside ddtSchemes).
        var = (case_dir / "pimple_overrides" / "fvSchemes").read_text(
            encoding="utf-8")
        assert re.search(r"default\s+Gauss linear;", var)
        assert re.search(r"grad\(U\)\s+cellLimited Gauss linear 1;", var)

    def test_inlet_k_is_decay_boosted(self, built, atmo):
        # 0/k must carry the decay-compensated boundary value, not the raw
        # Mack-level k -- and the boost must be a real (> 1) factor.
        case_dir, plan = built
        t = inlet_turbulence(plan.u_inf, atmo["nu"])
        text = (case_dir / "0" / "k").read_text(encoding="utf-8")
        m = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+);", text)
        assert m, "internalField not found in 0/k"
        assert float(m.group(1)) == pytest.approx(t["k"], rel=1e-4)
        assert t["k"] > t["k_target"]

    def test_rebuild_requires_force(self, built):
        case_dir, _ = built
        with pytest.raises(FileExistsError):
            build_case(4.0, RE_REF, 0, YAML_PATH, case_dir.parent)
        # And --force succeeds over the existing tree.
        rebuilt, _ = build_case(4.0, RE_REF, 0, YAML_PATH, case_dir.parent,
                                force=True)
        assert (rebuilt / "system" / "controlDict").is_file()
