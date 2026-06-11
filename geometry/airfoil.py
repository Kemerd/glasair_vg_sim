"""Airfoil coordinate ingestion, resampling, and section generation.

Single entry point for 2D section geometry in the VG placement study: reads
measured coordinate files (Selig and Lednicer layouts), refines them onto
CFD-grade cosine-clustered distributions, generates NACA 4-digit placeholder
sections for the empennage, and writes Selig-format files consumed by the
STL extrusion path (geometry/stl_gen.py).

Coordinate conventions (enforced everywhere in this repo):
  * Arrays are (N, 2) float64 with columns (x/c, y/c), chord-normalized.
  * Loop order is Selig: trailing edge along the upper surface to the
    leading edge, then back along the lower surface to the trailing edge
    (TE-upper -> LE -> TE-lower). The LE point appears exactly once.

Data provenance:
  * Committed wing section geometry/ls413.dat is the NASA/Langley
    LS(1)-0413 (GA(W)-2) coordinate set from the UIUC Airfoil Coordinate
    Database: https://m-selig.ae.illinois.edu/ads/coord/ls413.dat
    (Lednicer layout, 45 upper + 45 lower points, blunt trailing edge:
    upper TE y/c = -0.0016, lower TE y/c = -0.0071).
  * NACA 4-digit closed forms follow Abbott & von Doenhoff, "Theory of
    Wing Sections" (Dover, 1959), thickness polynomial and mean-line
    equations of section 6; closed-TE quartic coefficient -0.1036 per
    common CFD practice (open-TE published value is -0.1015).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# Anything accepted as a filesystem path by the public API.
PathLike = Union[str, Path]

# Two leading-edge points closer than this (per component) are treated as the
# same physical point. Lednicer files list the LE once per surface; the shared
# point must survive assembly into the Selig loop exactly once.
_LE_MATCH_TOL: float = 1e-8

# A file whose chord differs from unity by more than this is assumed to be in
# percent-chord or dimensional units and is rescaled. Properly normalized
# archive sets never miss unity by anywhere near 1e-3, so this threshold
# cleanly separates "needs rescaling" from float round-off in the data.
_CHORD_NORM_TOL: float = 1e-3

# Sharp-TE closure shears ordinates only, so it is only well-posed when both
# surfaces terminate at the same abscissa. The committed coordinate sets close
# in x to machine precision; anything beyond this tolerance is a bad file and
# must be rejected rather than silently emitted as a loop still open in x.
_TE_X_MATCH_TOL: float = 1e-9


# =============================================================================
#  File ingestion
# =============================================================================

def load_airfoil(path: PathLike) -> np.ndarray:
    """Read an airfoil coordinate file and return a normalized Selig loop.

    Auto-detects the two layouts found in the UIUC archive:

      * Selig:    title line, then one continuous sweep TE -> LE -> TE.
      * Lednicer: title line, a point-count line such as ``45.  45.``, the
        upper surface LE -> TE, a blank separator, the lower surface
        LE -> TE. Converted to Selig order with the LE shared exactly once.

    Blank lines and lines starting with '#' are skipped; the title (any
    non-numeric line) is discarded. Output is chord-normalized: LE shifted
    to x = 0, both columns scaled by the chord when the file is not already
    unit-chord.

    :param path: coordinate file path.
    :returns: (N, 2) float64 array, Selig loop order, chord-normalized.
    :raises ValueError: malformed file (too few points, count mismatch,
        or degenerate chord).
    """
    text = Path(path).read_text(encoding="utf-8")

    # Harvest every line that parses as a numeric pair. Titles and free-text
    # headers fail the float conversion and drop out here; blank separator
    # lines inside Lednicer files carry no information once counts are known.
    pairs: List[Tuple[float, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        # Coordinate rows carry at least two columns; lone tokens can only
        # be part of a title and are ignored outright.
        if len(tokens) < 2:
            continue
        try:
            pairs.append((float(tokens[0]), float(tokens[1])))
        except ValueError:
            continue

    if len(pairs) < 5:
        raise ValueError(
            f"airfoil file {path}: only {len(pairs)} coordinate pairs found; "
            "not a usable coordinate file"
        )

    # ---- Layout detection ---------------------------------------------------
    # Lednicer files carry a count line "N_upper  N_lower" right after the
    # title. Abscissae of a normalized airfoil never exceed x/c = 1, so a
    # first pair of integer-valued numbers both above 1 is unambiguous; the
    # claimed counts are then cross-checked against the pairs actually read.
    first_a, first_b = pairs[0]
    looks_like_counts = (
        first_a > 1.0 + 1e-6
        and first_b > 1.0 + 1e-6
        and abs(first_a - round(first_a)) < 1e-6
        and abs(first_b - round(first_b)) < 1e-6
    )

    if looks_like_counts:
        n_up = int(round(first_a))
        n_lo = int(round(first_b))
        if len(pairs) - 1 != n_up + n_lo:
            raise ValueError(
                f"airfoil file {path}: Lednicer count line claims "
                f"{n_up}+{n_lo} points but {len(pairs) - 1} were read"
            )
        # Both surfaces are listed nose-to-tail in Lednicer layout.
        upper = np.array(pairs[1 : 1 + n_up], dtype=float)            # LE -> TE
        lower = np.array(pairs[1 + n_up : 1 + n_up + n_lo], dtype=float)  # LE -> TE
        # The Selig loop wants the upper surface flipped (TE -> LE) and then
        # the lower surface appended. Drop the lower-surface LE copy when it
        # coincides with the upper-surface LE so the nose appears once.
        if np.all(np.abs(upper[0] - lower[0]) < _LE_MATCH_TOL):
            lower = lower[1:]
        coords = np.vstack([upper[::-1], lower])
    else:
        # Selig layout: the pairs already form the loop in the order needed.
        coords = np.array(pairs, dtype=float)

    # ---- Chord normalization --------------------------------------------------
    # Downstream code assumes unit chord with the LE at x = 0. Rescale only
    # when clearly required (percent-chord or dimensional input); y scales by
    # the same factor as x so thickness and camber ratios are preserved.
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    chord = x_max - x_min
    if chord <= 0.0:
        raise ValueError(f"airfoil file {path}: degenerate chord (x range is zero)")
    if abs(chord - 1.0) > _CHORD_NORM_TOL or abs(x_min) > _CHORD_NORM_TOL:
        coords = coords.copy()
        coords[:, 0] = (coords[:, 0] - x_min) / chord
        coords[:, 1] = coords[:, 1] / chord

    return coords


# =============================================================================
#  Loop splitting helpers (shared by resampling and thickness probing)
# =============================================================================

def _strictly_increasing_x(surface: np.ndarray) -> np.ndarray:
    """Drop points that do not advance in x along a LE -> TE surface.

    Measured coordinate sets occasionally repeat a station, or back-track by
    a print-precision epsilon near the nose. The root-bracketing tables in
    resample_airfoil and the y(x) probe splines in max_thickness both demand
    strictly increasing surface abscissae (which also guarantees nonzero
    segment lengths for the arc-length parameter), so the surface is walked
    once and only forward steps are kept; ordering is preserved (no sort,
    which could scramble a genuinely bad file rather than flag it).
    """
    keep = [0]
    for i in range(1, surface.shape[0]):
        if surface[i, 0] > surface[keep[-1], 0]:
            keep.append(i)
    return surface[keep]


def _split_surfaces(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a Selig loop into (upper, lower) surfaces, both LE -> TE.

    The LE is taken as the forward-most loop point (minimum x); for the
    chord-normalized sections used here this is robust, and both returned
    surfaces include the shared LE point as their first row.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 5:
        raise ValueError("coords must be an (N, 2) array with N >= 5")
    # Index of the nose: everything before it (inclusive) is the upper
    # surface running TE -> LE, everything after (inclusive) is the lower
    # surface already running LE -> TE.
    i_le = int(np.argmin(arr[:, 0]))
    upper = arr[: i_le + 1][::-1]
    lower = arr[i_le:]
    return _strictly_increasing_x(upper), _strictly_increasing_x(lower)


# =============================================================================
#  Spline resampling
# =============================================================================

def resample_airfoil(coords: np.ndarray, n_points: int = 241, te: str = "blunt") -> np.ndarray:
    """Resample a Selig loop onto a cosine-clustered distribution.

    The loop is fitted parametrically in cumulative (chordal) arc length s,
    one cubic per coordinate -- x(s) and y(s) -- with s = 0 anchored on the
    shared LE knot (s < 0 along the upper surface, s > 0 along the lower).
    A direct y(x) fit cannot represent the vertical tangent dy/dx -> inf at
    a round leading edge: it undersizes the nose and rings the curvature
    there. Both x(s) and y(s) stay smooth through the nose (dx/ds passes
    through zero), and because the LE is an interior knot of the loop fit
    rather than a surface endpoint, no boundary condition has to guess the
    vertical tangent -- it emerges from the data on both sides. This
    geometry feeds transition-sensitive CFD (kOmegaSSTLM), where
    suction-peak fidelity at the nose matters.

    Cosine clustering contract (identical to the previous y(x) version):
    each surface is evaluated at abscissae
        x_j = x_le + (1 - cos(theta_j)) / 2 * (x_te - x_le),
    theta uniform on [0, pi]. Point density scales as 1/sin(theta), packing
    points into the high-curvature LE region (and the TE) so the nose radius
    of this 13% section is resolved without inflating the total count. The
    arc station s_j with x(s_j) = x_j is recovered by brentq inside the
    bracketing knot interval -- x is strictly monotone in s along each
    surface of the sections used here, so each interval holds exactly one
    root -- and the output abscissae are exactly the cosine targets: the
    clustering contract is preserved bit-for-bit, only the ordinates come
    from the parametric fit. The LE itself is the shared s = 0 station.
    n_points must be odd so the LE is a single shared point.

    TE treatment:
      * ``'blunt'``: the measured open TE gap is preserved as-is.
      * ``'sharp'``: linear gap removal. With
            y_te_mean = (y_te_upper + y_te_lower) / 2
        each surface is sheared by a ramp proportional to chord fraction f:
            y'(x) = y(x) - f * (y_te_surface - y_te_mean)
        The ramp vanishes at the LE (f = 0), leaving the nose untouched,
        and both surfaces land on (x_te, y_te_mean) at f = 1, closing the
        gap without the curvature kink a local TE patch would introduce.
        Closure is only well-posed when both surfaces terminate at the same
        abscissa; inputs whose TE x differ by more than 1e-9 raise
        ValueError instead of emitting a loop that is still open in x.

    :param coords: (N, 2) Selig loop, chord-normalized.
    :param n_points: total output points (odd, >= 9).
    :param te: ``'blunt'`` or ``'sharp'``.
    :returns: (n_points, 2) Selig loop.
    :raises ValueError: bad ``te`` flag, even/short ``n_points``, or sharp
        closure requested for surfaces ending at different abscissae.
    """
    if te not in ("blunt", "sharp"):
        raise ValueError(f"te must be 'blunt' or 'sharp', got {te!r}")
    if n_points < 9 or n_points % 2 == 0:
        raise ValueError("n_points must be odd (single shared LE point) and >= 9")

    upper, lower = _split_surfaces(coords)

    # Sharp closure shears ordinates only; it cannot reconcile surfaces that
    # stop at different abscissae, so such inputs fail loudly here instead of
    # producing a loop the meshing path would treat as closed but is not.
    te_x_skew = abs(upper[-1, 0] - lower[-1, 0])
    if te == "sharp" and te_x_skew > _TE_X_MATCH_TOL:
        raise ValueError(
            f"te='sharp' needs coincident TE abscissae, but the surfaces end "
            f"{te_x_skew:.3e} apart in x (tolerance {_TE_X_MATCH_TOL:.0e})"
        )

    # Rebuild the loop from the filtered surfaces so the spline knots and the
    # per-surface bracketing tables below index one and the same point set
    # (the splitter may have dropped non-advancing stations from either side).
    loop = np.vstack([upper[::-1], lower[1:]])
    i_le = upper.shape[0] - 1

    # Chordal arc-length parameter, zeroed on the shared LE knot, and the two
    # coordinate splines of the parametric fit (rationale in the docstring).
    seg = np.hypot(np.diff(loop[:, 0]), np.diff(loop[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s -= s[i_le]
    spl_x = CubicSpline(s, loop[:, 0])
    spl_y = CubicSpline(s, loop[:, 1])

    # Unit cosine distribution shared by both surfaces. It doubles as the
    # chord-fraction ramp for the sharp-TE gap removal further down.
    n_side = (n_points + 1) // 2
    f = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_side)))

    def _sample(surface: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate one surface at its cosine abscissae via arc-length roots.

        ``surface`` runs LE -> TE; ``step`` is the loop-index stride from the
        LE toward that surface's TE (-1 upper, +1 lower). Surface knot m sits
        at loop index i_le + step * m, so each interior target abscissa is
        bracketed by the s values of its enclosing knot interval and brentq
        sees exactly one sign change of x(s) - x_j.
        """
        x_k = surface[:, 0]
        x_t = x_k[0] + f * (x_k[-1] - x_k[0])
        s_t = np.empty_like(x_t)
        # Endpoints need no root-finding: the LE is the shared s = 0 station
        # and the TE is this surface's terminal knot of the loop parameter.
        s_t[0] = 0.0
        s_t[-1] = s[i_le + step * (x_k.size - 1)]
        for j in range(1, x_t.size - 1):
            # Knot interval [m-1, m] containing the target; interior targets
            # lie strictly inside (x_le, x_te) so m lands in [1, n-1] -- the
            # clamp only guards against pathological float collapse.
            m = int(np.searchsorted(x_k, x_t[j]))
            m = min(max(m, 1), x_k.size - 1)
            a = s[i_le + step * (m - 1)]
            b = s[i_le + step * m]
            # s decreases toward the upper TE, so order the bracket for brentq.
            lo_s, hi_s = (b, a) if step < 0 else (a, b)
            s_t[j] = brentq(lambda ss, xt=x_t[j]: spl_x(ss) - xt, lo_s, hi_s)
        return x_t, spl_y(s_t)

    # Sample both surfaces; the abscissae returned are the exact cosine
    # targets, the ordinates come from the parametric fit at the root arcs.
    xs_up, ys_up = _sample(upper, -1)
    xs_lo, ys_lo = _sample(lower, +1)

    if te == "sharp":
        # Linear gap removal (formula in the docstring). Capture the TE
        # ordinates before shearing so both corrections see the same gap.
        y_te_up = float(ys_up[-1])
        y_te_lo = float(ys_lo[-1])
        y_te_mean = 0.5 * (y_te_up + y_te_lo)
        ys_up = ys_up - f * (y_te_up - y_te_mean)
        ys_lo = ys_lo - f * (y_te_lo - y_te_mean)

    # Reassemble the Selig loop: upper surface flipped back to TE -> LE,
    # lower surface contributing everything aft of the shared LE. The nose
    # point therefore appears exactly once and the total length is n_points.
    out = np.empty((n_points, 2), dtype=float)
    out[:n_side, 0] = xs_up[::-1]
    out[:n_side, 1] = ys_up[::-1]
    out[n_side:, 0] = xs_lo[1:]
    out[n_side:, 1] = ys_lo[1:]
    return out


# =============================================================================
#  NACA 4-digit generator (empennage placeholder sections)
# =============================================================================

def naca4_coords(designation: str, n_points: int = 241, te: str = "sharp") -> np.ndarray:
    """Generate NACA 4-digit section coordinates in Selig loop order.

    Closed-form generator covering both symmetric ('00xx', needed for the
    NACA 0010 tail placeholders in aircraft.yaml) and cambered sections.
    Output is cosine-clustered with a single shared LE point, matching the
    convention of :func:`resample_airfoil`.

    Thickness polynomial (half-thickness, Abbott & von Doenhoff):
        y_t = 5 t (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 + a4 x^4)
    with a4 = -0.1036 for ``te='sharp'`` (zeroes y_t at x = 1, closed TE)
    or the published open-TE value a4 = -0.1015 for ``te='blunt'``.

    :param designation: four digits, e.g. ``'0010'`` or ``'2412'``.
    :param n_points: total output points (odd, >= 9).
    :param te: ``'sharp'`` or ``'blunt'``.
    :returns: (n_points, 2) Selig loop, unit chord.
    """
    if te not in ("blunt", "sharp"):
        raise ValueError(f"te must be 'blunt' or 'sharp', got {te!r}")
    if n_points < 9 or n_points % 2 == 0:
        raise ValueError("n_points must be odd (single shared LE point) and >= 9")

    # Decode the designation: max camber m (fraction of chord), camber
    # position p (tenths of chord), thickness ratio t (percent of chord).
    digits = designation.strip()
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"NACA 4-digit designation must be 4 digits, got {designation!r}")
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:4]) / 100.0
    if t <= 0.0:
        raise ValueError(f"NACA {digits}: zero thickness is not a section")
    # Camber with no position (e.g. '2012') or vice versa has no defined
    # mean line in the 4-digit system; reject rather than guess.
    if (m > 0.0) != (p > 0.0):
        raise ValueError(f"NACA {digits}: camber magnitude/position digits are inconsistent")

    # Cosine-clustered chord stations, LE to TE (see resample_airfoil for
    # the rationale; same clustering keeps mesh density consistent between
    # measured and generated sections).
    n_side = (n_points + 1) // 2
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_side)))

    # Half-thickness polynomial; the quartic coefficient selects the TE
    # closure per the docstring.
    a4 = -0.1036 if te == "sharp" else -0.1015
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        + a4 * x ** 4
    )

    if m > 0.0:
        # Cambered: piecewise-parabolic mean line fore/aft of x = p, with
        # the thickness applied perpendicular to the local camber slope
        # (the classical construction; abscissae shift off the chord line).
        yc = np.where(
            x < p,
            m / p ** 2 * (2.0 * p * x - x ** 2),
            m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2),
        )
        dyc = np.where(
            x < p,
            2.0 * m / p ** 2 * (p - x),
            2.0 * m / (1.0 - p) ** 2 * (p - x),
        )
        theta = np.arctan(dyc)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
    else:
        # Symmetric: surfaces mirror exactly about the chord line. Sharing
        # the abscissae gives callers bit-exact upper/lower symmetry, which
        # the tail-section meshing relies on.
        xu, yu = x, yt
        xl, yl = x, -yt

    # Selig assembly, identical layout to resample_airfoil: upper TE -> LE,
    # then lower LE -> TE skipping the duplicated nose point.
    out = np.empty((n_points, 2), dtype=float)
    out[:n_side, 0] = xu[::-1]
    out[:n_side, 1] = yu[::-1]
    out[n_side:, 0] = xl[1:]
    out[n_side:, 1] = yl[1:]
    return out


# =============================================================================
#  Output
# =============================================================================

def write_selig(path: PathLike, coords: np.ndarray, name: str) -> None:
    """Write a coordinate loop as a plain Selig file (title + x y rows).

    Six decimal places match the UIUC archive convention and bound the
    write/read round-trip error at 5e-7 chord, comfortably below any
    meshing tolerance in this pipeline.

    :param path: output file path (written UTF-8).
    :param coords: (N, 2) coordinate loop.
    :param name: title line content (section name).
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be an (N, 2) array")
    # Build the whole file in memory; coordinate files are tiny and a single
    # write keeps the on-disk state atomic enough for our purposes.
    lines = [str(name).strip()]
    for x, y in arr:
        lines.append("%.6f %.6f" % (x, y))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
#  Thickness probe (used by tests and later boundary-layer / VG sizing)
# =============================================================================

def max_thickness(coords: np.ndarray) -> Tuple[float, float]:
    """Return (max t/c, x/c at the maximum) for a Selig coordinate loop.

    Both surfaces are splined and probed on a common dense grid; thickness
    is measured perpendicular to the chord line, t(x) = y_up(x) - y_lo(x).
    Results are normalized by the actual chord so the utility stays correct
    even for inputs that are not perfectly unit-chord.

    :param coords: (N, 2) Selig loop.
    :returns: (t_over_c, x_at_max) tuple of floats.
    """
    upper, lower = _split_surfaces(coords)
    spl_up = CubicSpline(upper[:, 0], upper[:, 1])
    spl_lo = CubicSpline(lower[:, 0], lower[:, 1])

    # Thickness is only defined where both surfaces exist; intersect the two
    # spans (they share the LE, but measured TEs can end at different x).
    x_lo_bound = max(upper[0, 0], lower[0, 0])
    x_hi_bound = min(upper[-1, 0], lower[-1, 0])

    # Dense cosine probe grid. The t(x) peak of the sections in this study
    # sits near mid-chord, but LE clustering costs nothing here and keeps
    # the same probe usable for nose-radius checks later on.
    sp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 1001)))
    xp = x_lo_bound + sp * (x_hi_bound - x_lo_bound)
    tx = spl_up(xp) - spl_lo(xp)
    i_max = int(np.argmax(tx))

    # Normalize by the true chord (full x extent across both surfaces).
    x_min = min(upper[0, 0], lower[0, 0])
    chord = max(upper[-1, 0], lower[-1, 0]) - x_min
    return float(tx[i_max] / chord), float((xp[i_max] - x_min) / chord)
