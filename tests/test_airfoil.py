"""Unit tests for geometry/airfoil.py.

Exercises the full coordinate-handling contract against the committed
LS(1)-0413 file (Lednicer layout, 45+45 points, blunt TE; source:
https://m-selig.ae.illinois.edu/ads/coord/ls413.dat) plus the NACA 4-digit
generator and the Selig writer round-trip. Reference values:

  * LS(1)-0413 published max t/c = 0.13; the committed coordinate set
    measures 0.1293 near x/c = 0.39 (see the note in aircraft.yaml).
  * Blunt TE gap of the committed set: -0.0016 - (-0.0071) = 0.0055.

The nose-fidelity suite guards the arc-length parametric resampler against
regression toward a y(x) fit, which cannot represent the vertical tangent
at the round LE (undersized nose, pinched LE angle, curvature ringing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from geometry.airfoil import (
    load_airfoil,
    max_thickness,
    naca4_coords,
    resample_airfoil,
    write_selig,
)

# Committed wing-section coordinate file, resolved relative to this test so
# the suite passes regardless of pytest's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
LS413_PATH = REPO_ROOT / "geometry" / "ls413.dat"


# -----------------------------------------------------------------------------
#  Shared fixtures -- module scope: the file parse and resamples are pure and
#  reused across many assertions, no point repeating the work per test.
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ls413() -> np.ndarray:
    """Committed LS(1)-0413 loop as loaded (89 points, Selig order)."""
    return load_airfoil(LS413_PATH)


@pytest.fixture(scope="module")
def ls413_blunt(ls413: np.ndarray) -> np.ndarray:
    """Default 241-point cosine resample with the measured TE gap kept."""
    return resample_airfoil(ls413, n_points=241, te="blunt")


@pytest.fixture(scope="module")
def ls413_sharp(ls413: np.ndarray) -> np.ndarray:
    """241-point resample with the TE closed by linear gap removal."""
    return resample_airfoil(ls413, n_points=241, te="sharp")


def _thickness_at(coords: np.ndarray, x_station: float) -> float:
    """Linear-interpolated thickness at one chord station (test-grade probe).

    Splits the Selig loop at the LE (minimum x) and interpolates each surface
    independently; np.interp needs ascending abscissae, hence the flip of the
    upper surface back to LE -> TE order.
    """
    i_le = int(np.argmin(coords[:, 0]))
    upper = coords[: i_le + 1][::-1]
    lower = coords[i_le:]
    y_up = np.interp(x_station, upper[:, 0], upper[:, 1])
    y_lo = np.interp(x_station, lower[:, 0], lower[:, 1])
    return float(y_up - y_lo)


# -----------------------------------------------------------------------------
#  Loading the committed Lednicer file
# -----------------------------------------------------------------------------

class TestLoadLs413:
    def test_point_count(self, ls413: np.ndarray) -> None:
        # 45 upper + 45 lower with the LE shared exactly once -> 89 points.
        assert ls413.shape == (89, 2)

    def test_chord_normalized(self, ls413: np.ndarray) -> None:
        # File is already unit chord; the loader must not perturb it.
        assert ls413[:, 0].min() == pytest.approx(0.0, abs=1e-12)
        assert ls413[:, 0].max() == pytest.approx(1.0, abs=1e-12)

    def test_selig_loop_order(self, ls413: np.ndarray) -> None:
        # Loop must start and end at the TE, with the upper-surface TE
        # ordinate above the lower one (this section's gap is asymmetric).
        assert ls413[0, 0] > 0.99 and ls413[-1, 0] > 0.99
        assert ls413[0, 1] > ls413[-1, 1]
        # The nose sits mid-array: x decreases then increases through it.
        i_le = int(np.argmin(ls413[:, 0]))
        assert 0 < i_le < len(ls413) - 1

    def test_max_thickness(self, ls413: np.ndarray) -> None:
        # Published t/c = 0.13; committed coordinates measure 0.1293 around
        # x/c ~ 0.39 (cross-checked in aircraft.yaml wing section notes).
        t_over_c, x_at = max_thickness(ls413)
        assert t_over_c == pytest.approx(0.1293, abs=0.003)
        assert 0.30 <= x_at <= 0.45


# -----------------------------------------------------------------------------
#  Cosine resampling -- blunt TE (default CFD geometry path)
# -----------------------------------------------------------------------------

class TestResampleBlunt:
    def test_shape(self, ls413_blunt: np.ndarray) -> None:
        assert ls413_blunt.shape == (241, 2)

    def test_thickness_preserved(self, ls413: np.ndarray, ls413_blunt: np.ndarray) -> None:
        # Resampling is interpolation, not smoothing: the section thickness
        # must survive essentially unchanged.
        t_raw, _ = max_thickness(ls413)
        t_res, _ = max_thickness(ls413_blunt)
        assert abs(t_res - t_raw) < 0.002

    def test_te_gap_preserved(self, ls413_blunt: np.ndarray) -> None:
        # Measured open TE: upper y = -0.0016, lower y = -0.0071 -> 0.0055.
        gap = ls413_blunt[0, 1] - ls413_blunt[-1, 1]
        assert gap == pytest.approx(0.0055, abs=0.0008)

    def test_cosine_clustering(self, ls413_blunt: np.ndarray) -> None:
        # Cosine clustering contract: spacing collapses toward the LE. The
        # mean |dx| of the 10 segments hugging the nose (5 per surface) must
        # be at least 5x finer than the spacing near mid-chord.
        dx = np.abs(np.diff(ls413_blunt[:, 0]))
        i_le = int(np.argmin(ls413_blunt[:, 0]))
        near_le = np.concatenate([dx[i_le - 5 : i_le], dx[i_le : i_le + 5]])
        # Mid-chord segment: the one whose midpoint lies closest to x = 0.5.
        seg_mid_x = 0.5 * (ls413_blunt[:-1, 0] + ls413_blunt[1:, 0])
        mid_dx = dx[int(np.argmin(np.abs(seg_mid_x - 0.5)))]
        assert near_le.mean() * 5.0 <= mid_dx

    def test_input_validation(self, ls413: np.ndarray) -> None:
        # Even point counts cannot share a single LE point; bad TE flags are
        # programming errors -- both must fail loudly, not silently degrade.
        with pytest.raises(ValueError):
            resample_airfoil(ls413, n_points=240)
        with pytest.raises(ValueError):
            resample_airfoil(ls413, te="rounded")


# -----------------------------------------------------------------------------
#  Cosine resampling -- sharp TE (linear gap removal)
# -----------------------------------------------------------------------------

class TestResampleSharp:
    def test_te_closed(self, ls413_sharp: np.ndarray) -> None:
        # Both surfaces must land on the identical TE point.
        assert abs(ls413_sharp[0, 1] - ls413_sharp[-1, 1]) < 1e-9
        assert abs(ls413_sharp[0, 0] - ls413_sharp[-1, 0]) < 1e-9

    def test_le_unchanged(self, ls413_blunt: np.ndarray, ls413_sharp: np.ndarray) -> None:
        # The gap-removal ramp is zero at the nose by construction; the LE
        # point must be bit-identical between blunt and sharp variants.
        i_le_b = int(np.argmin(ls413_blunt[:, 0]))
        i_le_s = int(np.argmin(ls413_sharp[:, 0]))
        assert np.allclose(ls413_blunt[i_le_b], ls413_sharp[i_le_s], atol=1e-12)

    def test_midchord_thickness_shift_small(
        self, ls413_blunt: np.ndarray, ls413_sharp: np.ndarray
    ) -> None:
        # The linear ramp redistributes half the 0.0055 gap per surface;
        # at mid-chord that is 0.00275 of thickness change -- bounded here.
        t_blunt = _thickness_at(ls413_blunt, 0.5)
        t_sharp = _thickness_at(ls413_sharp, 0.5)
        assert abs(t_sharp - t_blunt) < 0.003


# -----------------------------------------------------------------------------
#  Nose fidelity of the arc-length parametric resampler (review regression
#  guards: a y(x) per-surface fit fails every assertion in this section)
# -----------------------------------------------------------------------------

def _dense_loop_reference(coords: np.ndarray, n: int = 20000) -> np.ndarray:
    """Chordal arc-length parametric oracle of a raw loop, as a dense polyline.

    A CubicSpline over cumulative segment length reproduces the measured loop
    with the vertical LE tangent intact. 20000 samples put the polyline
    sagitta error near 1e-7 chord even at the nose (curvature ~ 50 / chord,
    ds ~ 1e-4, so k * ds^2 / 8 ~ 7e-8), far below the 5e-5 bound asserted
    against it -- the polyline itself contributes no measurable slack.
    """
    seg = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return CubicSpline(s, coords)(np.linspace(0.0, s[-1], n))


def _max_distance_to_polyline(points: np.ndarray, poly: np.ndarray) -> float:
    """Largest point-to-polyline distance (normal-deviation measure).

    Each query point is projected onto every polyline segment, the projection
    clamped to the segment span, and the closest hit kept; the maximum of
    those minima over all query points comes back. Vectorized per query
    point, so 241 points against 20000 segments stays well under a second.
    """
    a, b = poly[:-1], poly[1:]
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    worst = 0.0
    for p in points:
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / ab2, 0.0, 1.0)
        d = np.hypot(*(p - (a + t[:, None] * ab)).T)
        worst = max(worst, float(d.min()))
    return worst


def _le_included_angle_deg(loop: np.ndarray) -> float:
    """Included angle at the LE vertex of a discrete loop, in degrees.

    Measured between the two segments leaving the forward-most point; a
    pinched (undersized) nose reads low because both neighbors collapse
    toward the chord line instead of wrapping around the LE radius.
    """
    i = int(np.argmin(loop[:, 0]))
    v1 = loop[i - 1] - loop[i]
    v2 = loop[i + 1] - loop[i]
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _smoothed_curvature(surf: np.ndarray, window: int = 5) -> np.ndarray:
    """Signed finite-difference curvature along a LE -> TE surface polyline.

    Three-point circumcircle curvature (2 * cross / product of the triangle
    side lengths) handles the non-uniform cosine spacing exactly. The
    5-sample moving average then irons out the knot-to-knot steps that a
    piecewise cubic's second derivative carries by construction, which would
    otherwise read as sign jitter rather than geometry; edge padding keeps
    the output aligned with the interior points of the surface.
    """
    p0, p1, p2 = surf[:-2], surf[1:-1], surf[2:]
    a, b, c = p1 - p0, p2 - p1, p2 - p0
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    k = 2.0 * cross / (np.hypot(*a.T) * np.hypot(*b.T) * np.hypot(*c.T))
    pad = window // 2
    return np.convolve(np.pad(k, pad, mode="edge"), np.ones(window) / window, mode="valid")


def _curvature_sign_changes(
    loop: np.ndarray, surface: str, x_lo: float, x_hi: float, deadband: float = 1e-3
) -> int:
    """Count curvature sign changes on one surface over [x_lo, x_hi].

    Samples with |curvature| below the deadband are dropped before counting,
    so a genuine inflection registers exactly once (the sign flips between
    its flanks) instead of chattering through the zero crossing. Tolerance
    choice: real surface curvature of this 13% section stays above ~4e-2 / c
    over the probed ranges away from the inflection itself, while the noise
    floor of the smoothed stencil sits orders of magnitude below 1e-3 / c --
    the deadband can therefore only swallow the crossing band of a genuine
    inflection, never a real surface feature.
    """
    n_side = (len(loop) + 1) // 2
    surf = loop[:n_side][::-1] if surface == "upper" else loop[n_side - 1 :]
    k = _smoothed_curvature(surf)
    x_mid = surf[1:-1, 0]  # curvature exists at interior points only
    k = k[(x_mid >= x_lo) & (x_mid <= x_hi)]
    sig = np.sign(k[np.abs(k) > deadband])
    return int(np.sum(sig[:-1] != sig[1:]))


class TestResampleNoseFidelity:
    """Regression guards for the arc-length parametric fit (review finding).

    The previous per-surface y(x) fit could not represent the vertical LE
    tangent: nose undersized by up to 6.2e-4 c against the parametric
    reference, discretized LE included angle 153.8 deg, spurious curvature
    sign flips at x ~ 0.004-0.006 (upper) and ringing around the genuine
    lower-surface inflection. The bounds below pin the parametric behavior.
    """

    def test_nose_normal_deviation(self, ls413: np.ndarray, ls413_blunt: np.ndarray) -> None:
        # Every resampled point must sit on the arc-length oracle of the raw
        # loop to within 5e-5 chord; the y(x) fit missed by 6.2e-4 at the
        # nose, the parametric fit lands near 1e-7 -- two-decade margin.
        ref = _dense_loop_reference(ls413)
        assert _max_distance_to_polyline(ls413_blunt, ref) < 5e-5

    def test_le_included_angle(self, ls413_blunt: np.ndarray) -> None:
        # The y(x) fit pinched the nose to 153.8 deg; the parametric fit
        # recovers ~165 deg at 241-point clustering. The 162 deg floor sits
        # between the two with margin against either drifting.
        assert _le_included_angle_deg(ls413_blunt) >= 162.0

    def test_upper_curvature_sign_discipline(self, ls413_blunt: np.ndarray) -> None:
        # The upper surface is convex through the whole suction-peak region,
        # so any sign change over [0.001, 0.5] is interpolation ringing (the
        # y(x) fit flipped sign around x ~ 0.004-0.006 right at the nose).
        assert _curvature_sign_changes(ls413_blunt, "upper", 0.001, 0.5) == 0

    def test_lower_curvature_single_inflection(self, ls413_blunt: np.ndarray) -> None:
        # The aft-loaded lower surface carries exactly one genuine inflection
        # (x ~ 0.72 on the committed set) where the camber recovery begins;
        # more flips mean ringing, fewer mean the recovery got smoothed away.
        assert _curvature_sign_changes(ls413_blunt, "lower", 0.001, 0.9) == 1

    def test_sharp_te_requires_matching_abscissae(self, ls413: np.ndarray) -> None:
        # Sharp closure shears ordinates only; surfaces ending at different x
        # cannot be closed by it and must be rejected, not papered over.
        skewed = ls413.copy()
        skewed[-1, 0] = 0.999  # pull the lower-surface TE forward in x
        with pytest.raises(ValueError):
            resample_airfoil(skewed, n_points=241, te="sharp")
        # The blunt path carries no closure obligation: it must keep mapping
        # each surface onto its own [LE, TE] span and accept the same input.
        out = resample_airfoil(skewed, n_points=241, te="blunt")
        assert out.shape == (241, 2)


# -----------------------------------------------------------------------------
#  NACA 4-digit generator (NACA 0010 = empennage placeholder section)
# -----------------------------------------------------------------------------

class TestNaca0010:
    @pytest.fixture(scope="class")
    def n0010(self) -> np.ndarray:
        return naca4_coords("0010", n_points=241, te="sharp")

    def test_symmetry(self, n0010: np.ndarray) -> None:
        # A '00xx' section must mirror exactly: identical abscissae, equal
        # and opposite ordinates between the two surfaces.
        n_side = (len(n0010) + 1) // 2
        upper = n0010[:n_side][::-1]   # LE -> TE
        lower = n0010[n_side - 1 :]    # LE -> TE (shares the nose row)
        assert np.max(np.abs(upper[:, 0] - lower[:, 0])) < 1e-12
        assert np.max(np.abs(upper[:, 1] + lower[:, 1])) < 1e-9

    def test_max_thickness(self, n0010: np.ndarray) -> None:
        # 10% section, peak near x/c = 0.30 for the 4-digit family.
        t_over_c, x_at = max_thickness(n0010)
        assert t_over_c == pytest.approx(0.10, abs=0.002)
        assert 0.25 <= x_at <= 0.35

    def test_sharp_te_closed(self, n0010: np.ndarray) -> None:
        # a4 = -0.1036 zeroes the thickness polynomial at x = 1 exactly.
        assert abs(n0010[0, 1] - n0010[-1, 1]) < 1e-9

    def test_blunt_te_open(self) -> None:
        # The published a4 = -0.1015 leaves the classical finite TE: for
        # t = 0.10 the gap is 2 * 5t * 0.0021 = 0.0021 chord.
        blunt = naca4_coords("0010", n_points=241, te="blunt")
        gap = blunt[0, 1] - blunt[-1, 1]
        assert gap == pytest.approx(0.0021, abs=2e-4)


# -----------------------------------------------------------------------------
#  Selig writer round-trip
# -----------------------------------------------------------------------------

def test_write_selig_roundtrip(tmp_path: Path, ls413_blunt: np.ndarray) -> None:
    # Write the resampled loop and reload it through the full parser; the
    # only loss allowed is the %.6f quantization (<= 5e-7 per coordinate).
    out_path = tmp_path / "ls413_resampled.dat"
    write_selig(out_path, ls413_blunt, "LS(1)-0413 resampled 241pt")
    reloaded = load_airfoil(out_path)
    assert reloaded.shape == ls413_blunt.shape
    assert np.max(np.abs(reloaded - ls413_blunt)) < 1e-6
