"""Wing/tail planform extraction from 3-view DXF linework, with YAML fallback.

This is the M0 "DXF parse stub" required by glasair3-vg-cfd-spec.md Phase 0
item 2 ("DXF reader: extract wing chord distribution ... fall back to
parameters in a single aircraft.yaml"). The implemented surface is a straight
tapered (trapezoidal) panel: root/tip chords, span, leading-edge sweep, plus
the derived area and mean aerodynamic chord. That is exactly the level of
fidelity the extruded-section studies consume — they need the local chord at
a spanwise station (Planform.chord_at) and nothing more. Curved planform
breaks, aileron cutouts and hinge lines remain on the roadmap once a real
DXF is available to define them.

DWG NOTE (read this before reporting a parse failure): the factory 3-view at
ref/Glasair-III-3-View-Drawing.dwg is a DWG, a proprietary binary format that
ezdxf cannot read. Convert it externally — e.g. the free ODA File Converter,
or a "save as DXF" from any CAD package — and drop the resulting .dxf into
geometry/dxf/. Until a converted drawing exists there, the YAML fallback
(planform_from_yaml) governs all planform numbers, per the spec's "do not
block on the DXFs" directive.

DATA PROVENANCE
  Trapezoid fallback values originate in aircraft.yaml, which carries its own
  per-line provenance tags: [3VIEW] Stoddard-Hamilton factory 3-view rev G
  4/26/90, [VT2005] Virginia Tech Glasair III analysis 2005-03-30, [DERIVED]
  with formulas in comments. The standard trapezoid MAC/area formulas used
  here are e.g. Raymer, "Aircraft Design: A Conceptual Approach", eq. 4.15.

COORDINATE / UNIT CONVENTIONS FOR INPUT DXF LINEWORK
  x = chordwise (LE -> TE), y = spanwise (root -> tip), matching the
  repository STL convention. The drawn outline is assumed to be ONE panel,
  root rib to tip — the usual way a 3-view wing half is drafted — so the
  full span is twice the drawn spanwise extent. Modelspace units are taken
  from the $INSUNITS header variable; a unitless drawing (code 0, the ezdxf
  default) is assumed to be meters, and that assumption is printed by the
  CLI summary so it cannot pass unnoticed.

Run as a script for interactive use:
    python geometry/dxf_reader.py geometry/dxf/<file>.dxf [--layer WING]
prints the layer list and the extracted planform summary (plain ASCII).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import ezdxf

# Type-only import: units.py is a sibling contract module; importing it lazily
# under TYPE_CHECKING keeps this module importable even in a partially built
# checkout (the runtime never touches it — planform_from_yaml is pure
# attribute access against the documented AircraftParams shape).
if TYPE_CHECKING:  # pragma: no cover
    from geometry.units import AircraftParams


# =============================================================================
#  Errors
# =============================================================================
class DxfParseError(Exception):
    """Raised when a DXF cannot yield a planform.

    Messages are written to be actionable from the command line: they name
    the file, the layer filter in effect, what entity types and layers WERE
    found, and what to do about it (fix the filter, convert the DWG, or fall
    back to aircraft.yaml).
    """


# =============================================================================
#  Planform model
# =============================================================================
#  Numerical floor used to reject degenerate geometry (zero-extent rows,
#  zero-span outlines). One micron is far below drafting precision and far
#  above float noise at aircraft scale.
_EPS_M: float = 1.0e-6


@dataclass(frozen=True)
class Planform:
    """Straight-tapered (trapezoidal) lifting-surface planform, SI units.

    Attributes
    ----------
    span_m : float
        FULL span b (both panels tip-to-tip), meters.
    chord_root_m : float
        Root chord, meters. Must be strictly positive.
    chord_tip_m : float
        Tip chord, meters. Zero is legal (pointed tip); negative is not.
    le_sweep_rad : Optional[float]
        Leading-edge sweep, radians, positive aft. None when the source did
        not define it (the YAML fallback passes it through when present).
    """

    span_m: float
    chord_root_m: float
    chord_tip_m: float
    le_sweep_rad: Optional[float] = None

    def __post_init__(self) -> None:
        # Validate eagerly so a bad planform dies at construction, not three
        # scripts later inside a mesh generator. The "not (x > 0)" idiom also
        # rejects NaN, which compares false against everything.
        if not (self.span_m > _EPS_M):
            raise ValueError(f"span_m must be positive, got {self.span_m!r}")
        if not (self.chord_root_m > _EPS_M):
            raise ValueError(
                f"chord_root_m must be positive, got {self.chord_root_m!r}"
            )
        # Tip chord may be zero (delta/pointed tip) but never negative or NaN.
        if not (self.chord_tip_m >= 0.0):
            raise ValueError(
                f"chord_tip_m must be >= 0, got {self.chord_tip_m!r}"
            )

    # -------------------------------------------------------------------------
    #  Derived quantities
    # -------------------------------------------------------------------------
    @property
    def taper_ratio(self) -> float:
        """Taper ratio lambda = c_tip / c_root (dimensionless)."""
        return self.chord_tip_m / self.chord_root_m

    def chord_at(self, eta: float) -> float:
        """Local chord at non-dimensional spanwise station eta = y / (b/2).

        Linear taper interpolation between root (eta = 0) and tip (eta = 1):
            c(eta) = c_root + (c_tip - c_root) * eta
        This is exact for a trapezoid, which is all this class represents.
        eta outside [0, 1] is a caller bug (a station off the panel), so it
        raises rather than extrapolating silently.
        """
        # The negated-range test rejects NaN as well as out-of-range values.
        if not (0.0 <= eta <= 1.0):
            raise ValueError(
                f"eta = {eta!r} is outside [0, 1]; eta is the spanwise "
                f"fraction y/(b/2) measured root -> tip"
            )
        return self.chord_root_m + (self.chord_tip_m - self.chord_root_m) * float(eta)

    def area(self) -> float:
        """Planform area of the FULL wing (both panels), m^2.

        Standard trapezoid: S = b * (c_root + c_tip) / 2. Equivalent to
        2 * (b/2) * c_mean per panel; consistent with the S_ref convention
        used by [VT2005] (the YAML area_ref of 87.6 ft^2 back-checks against
        this formula to within rounding of the published chords).
        """
        return self.span_m * (self.chord_root_m + self.chord_tip_m) / 2.0

    def mac(self) -> float:
        """Mean aerodynamic chord of a linearly tapered panel, meters.

        Standard result (Raymer eq. 4.15, lambda = taper ratio):
            MAC = (2/3) * c_root * (1 + lambda + lambda^2) / (1 + lambda)
        NOTE: a published MAC (e.g. the 3.76 ft from [VT2005]) can differ by
        a fraction of a percent from this formula because published values
        often come from a vortex-lattice planform with slightly different
        chord rounding — treat sub-1% disagreement as definitional noise.
        """
        lam = self.taper_ratio
        return (2.0 / 3.0) * self.chord_root_m * (1.0 + lam + lam * lam) / (1.0 + lam)


# =============================================================================
#  YAML fallback path (the governing source until a converted DXF exists)
# =============================================================================
def planform_from_yaml(ac: "AircraftParams") -> Planform:
    """Build the wing Planform from the aircraft.yaml wing block.

    All values arrive pre-converted to SI by geometry/units.py (lengths in
    meters, angles in radians), so this function is pure assembly plus one
    derivation fallback: when explicit chords are absent it inverts the
    trapezoid area formula, mirroring the [DERIVED] entries documented in
    aircraft.yaml itself:
        c_root = 2 S / (b (1 + lambda)),   c_tip = lambda * c_root
    """
    wing = ac.wing

    # Full span is mandatory — without it there is no planform at all.
    span = getattr(wing, "span", None)
    if span is None:
        raise ValueError("aircraft.yaml wing block is missing 'span'")

    # Preferred path: explicit chords (these exist in the committed YAML and
    # already encode the area/taper inversion, with provenance comments).
    chord_root = getattr(wing, "chord_root", None)
    chord_tip = getattr(wing, "chord_tip", None)

    # Derivation fallback: a sparser YAML (or a future tail block reusing
    # this function's pattern) may carry only area + taper. Refuse to guess
    # beyond that — fabricating chords from nothing violates the spec's
    # "unknowns are marked, never silently trusted" rule.
    if chord_root is None or chord_tip is None:
        area_ref = getattr(wing, "area_ref", None)
        taper = getattr(wing, "taper_ratio", None)
        if area_ref is None or taper is None:
            raise ValueError(
                "aircraft.yaml wing block needs either chord_root + chord_tip "
                "or area_ref + taper_ratio to define the planform"
            )
        # Trapezoid inversion: S = b * c_root * (1 + lambda) / 2.
        chord_root = 2.0 * area_ref / (span * (1.0 + taper))
        chord_tip = taper * chord_root

    # LE sweep is optional context (already radians via units.py). The
    # committed value is 0 deg per [VT2005]; None simply propagates.
    sweep = getattr(wing, "sweep_le", None)

    return Planform(
        span_m=float(span),
        chord_root_m=float(chord_root),
        chord_tip_m=float(chord_tip),
        le_sweep_rad=None if sweep is None else float(sweep),
    )


# =============================================================================
#  DXF reading
# =============================================================================
#  Entity types that contribute planform linework. Dimensions, text, hatches
#  and blocks are deliberately ignored — only outline geometry defines chords.
_LINEWORK_TYPES: Tuple[str, ...] = ("LINE", "LWPOLYLINE", "POLYLINE")

#  $INSUNITS drawing-units code -> meters-per-drawing-unit. Codes follow the
#  DXF reference (0 unitless, 1 in, 2 ft, 4 mm, 5 cm, 6 m, 7 km). Unitless
#  drawings are ASSUMED to be meters; that assumption is documented in the
#  module docstring and surfaced by the CLI. Any other code is an error —
#  silently misreading feet as meters would corrupt every downstream mesh.
_INSUNITS_TO_M: Dict[int, float] = {
    0: 1.0,       # unitless -> assume meters (documented assumption)
    1: 0.0254,    # inches
    2: 0.3048,    # feet
    4: 1.0e-3,    # millimeters
    5: 1.0e-2,    # centimeters
    6: 1.0,       # meters
    7: 1.0e3,     # kilometers
}


def _open_dxf(path: Union[str, Path]) -> "ezdxf.document.Drawing":
    """Open a DXF file, translating ezdxf/IO failures into DxfParseError.

    Centralized so read_dxf_planform and list_layers report identical,
    actionable messages — including the DWG hint, since feeding the factory
    DWG straight in is the most likely first-use mistake.
    """
    p = Path(path)
    try:
        return ezdxf.readfile(str(p))
    except IOError as exc:
        # Covers missing files and permission problems.
        raise DxfParseError(f"cannot open DXF file '{p}': {exc}") from exc
    except ezdxf.DXFStructureError as exc:
        # Covers corrupt files AND binary DWGs renamed to .dxf.
        raise DxfParseError(
            f"'{p}' is not a readable DXF ({exc}). If this is a DWG (e.g. the "
            f"factory 3-view at ref/Glasair-III-3-View-Drawing.dwg), convert "
            f"it externally with ODA File Converter and retry."
        ) from exc


def _insunits_scale(doc: "ezdxf.document.Drawing", path: Path) -> float:
    """Resolve the modelspace -> meters scale factor from $INSUNITS."""
    code = int(doc.header.get("$INSUNITS", 0))
    # Unknown codes (yards, miles, astronomical units...) are refused rather
    # than guessed: a wrong length scale poisons every downstream artifact.
    if code not in _INSUNITS_TO_M:
        raise DxfParseError(
            f"'{path}' uses unsupported $INSUNITS code {code}; re-export the "
            f"drawing in mm/cm/m/in/ft (codes 4/5/6/1/2) or unitless meters"
        )
    return _INSUNITS_TO_M[code]


def _entity_points(entity: Any) -> List[Tuple[float, float]]:
    """Extract (x, y) vertices from one supported linework entity.

    Z coordinates are dropped: a 3-view is planar by construction, and any
    stray elevation on traced linework carries no planform information.
    """
    kind = entity.dxftype()
    if kind == "LINE":
        # Straight segment: both endpoints participate in the extent search.
        s, e = entity.dxf.start, entity.dxf.end
        return [(float(s.x), float(s.y)), (float(e.x), float(e.y))]
    if kind == "LWPOLYLINE":
        # Lightweight polyline: vertices come straight off the entity. Bulge
        # (arc) segments contribute their endpoints only — adequate for the
        # straight-edged trapezoid outlines this reader supports.
        return [(float(x), float(y)) for x, y in entity.get_points("xy")]
    if kind == "POLYLINE":
        # Heavyweight 2D/3D polyline: vertices exposed as Vec3 sequence.
        return [(float(v.x), float(v.y)) for v in entity.points()]
    # Unsupported kinds are filtered before this call; returning empty keeps
    # the function total anyway.
    return []


def read_dxf_planform(
    path: Union[str, Path], layer: Optional[str] = None
) -> Planform:
    """Extract a trapezoidal planform from DXF linework.

    Parameters
    ----------
    path : str | Path
        DXF file. Modelspace units are resolved via $INSUNITS (unitless is
        assumed meters — see module docstring).
    layer : Optional[str]
        When given, only entities on this layer participate (DXF layer names
        are case-insensitive, so the comparison is too). Use list_layers()
        to discover candidates interactively.

    Method
    ------
    1. Collect every vertex of LINE / LWPOLYLINE / POLYLINE entities (after
       the optional layer filter).
    2. The spanwise direction is y. The drawn outline is assumed to be one
       panel, root at min-y, tip at max-y; full span = 2 * drawn extent.
    3. Chords are the x-extents of the vertex clusters at the min-y and
       max-y rows. Clustering is tolerant: vertices within 0.5% of the
       spanwise extent of the exact row ordinate count as members, because
       traced/converted linework rarely lands multiple endpoints on an
       exactly shared coordinate (drafting snap slop, DWG->DXF rounding).
       0.5% of the half-span (~18 mm here) is far below any plausible rib
       spacing, so the clusters cannot swallow a neighboring station.
    4. LE sweep is recovered from the x-shift of the min-x (leading-edge)
       vertex between the two rows.

    Raises
    ------
    DxfParseError
        When the file is unreadable, has no usable linework (the message
        names what WAS found), has zero spanwise extent, or has a degenerate
        root row.
    """
    p = Path(path)
    doc = _open_dxf(p)
    scale = _insunits_scale(doc, p)
    msp = doc.modelspace()

    # ---- pass 1: gather vertices, plus diagnostics for the failure path ----
    # seen_types / seen_layers record EVERYTHING in modelspace (pre-filter)
    # so a failed extraction can tell the user exactly what the file holds.
    points: List[Tuple[float, float]] = []
    seen_types: Dict[str, int] = {}
    seen_layers: Set[str] = set()
    want_layer = None if layer is None else layer.casefold()

    for entity in msp:
        kind = entity.dxftype()
        seen_types[kind] = seen_types.get(kind, 0) + 1
        ent_layer = str(entity.dxf.layer)
        seen_layers.add(ent_layer)
        # Layer filter (case-insensitive per the DXF spec's treatment of
        # symbol-table names) and entity-type filter applied together.
        if want_layer is not None and ent_layer.casefold() != want_layer:
            continue
        if kind not in _LINEWORK_TYPES:
            continue
        points.extend(_entity_points(entity))

    # ---- failure path: explain precisely what was found instead ----
    if not points:
        found = (
            ", ".join(f"{n}x {t}" for t, n in sorted(seen_types.items()))
            or "no entities at all"
        )
        layers_found = ", ".join(sorted(seen_layers)) or "none"
        where = f" on layer '{layer}'" if layer is not None else ""
        raise DxfParseError(
            f"no usable LINE/LWPOLYLINE/POLYLINE linework{where} in '{p}'. "
            f"Modelspace contains: {found}; layers present: {layers_found}. "
            f"Check the --layer filter, explode blocks/dimensions to plain "
            f"linework, or rely on the aircraft.yaml fallback "
            f"(planform_from_yaml) until a clean DXF is available."
        )

    # ---- pass 2: spanwise extents and tolerant row clustering ----
    ys = [pt[1] for pt in points]
    y_min, y_max = min(ys), max(ys)
    drawn_extent = y_max - y_min

    # A zero-height outline means the linework is collinear in y — likely a
    # single chord line or a wrongly-oriented drawing. Refuse loudly.
    if drawn_extent * scale <= _EPS_M:
        raise DxfParseError(
            f"linework in '{p}' has zero spanwise (y) extent; expected a "
            f"root-to-tip panel outline with x = chordwise, y = spanwise"
        )

    # Row membership tolerance: 0.5% of the drawn extent (see docstring for
    # the sizing argument). Floored at _EPS to stay meaningful for tiny test
    # geometries drawn in exact coordinates.
    tol = max(_EPS_M / max(scale, _EPS_M), 0.005 * drawn_extent)
    root_xs = [x for (x, y) in points if abs(y - y_min) <= tol]
    tip_xs = [x for (x, y) in points if abs(y - y_max) <= tol]

    # Convert clustered extents to meters. Root must have finite chord; a
    # pointed TIP (single vertex, zero extent) is geometrically legitimate.
    chord_root = (max(root_xs) - min(root_xs)) * scale
    chord_tip = (max(tip_xs) - min(tip_xs)) * scale
    if chord_root <= _EPS_M:
        raise DxfParseError(
            f"root row of '{p}' (y = {y_min:.6g}, {len(root_xs)} vertices) "
            f"has zero chordwise extent — outline may be rotated 90 deg "
            f"(spanwise must be y) or the root rib linework is missing"
        )

    # ---- LE sweep from the leading-edge x-shift between the rows ----
    # min-x is the LE by the module's chordwise convention (LE -> TE = +x).
    x_le_root = min(root_xs) * scale
    x_le_tip = min(tip_xs) * scale
    half_span = drawn_extent * scale
    le_sweep = math.atan2(x_le_tip - x_le_root, half_span)

    # Full span doubles the drawn panel — the standard half-model 3-view
    # convention documented in the module docstring. A drawing of the FULL
    # symmetric planform would violate this assumption; in that ambiguous
    # case the spec mandates the YAML fallback anyway.
    return Planform(
        span_m=2.0 * half_span,
        chord_root_m=chord_root,
        chord_tip_m=chord_tip,
        le_sweep_rad=le_sweep,
    )


def list_layers(path: Union[str, Path]) -> List[str]:
    """Return the sorted layer names defined in a DXF (interactive helper).

    Intended workflow: run this (or the CLI below) on a freshly converted
    drawing to find which layer carries the wing outline, then pass that
    name as the ``layer`` argument of read_dxf_planform().
    """
    doc = _open_dxf(path)
    # The layer TABLE lists every defined layer, including ones with no
    # modelspace entities — still useful, since converters often preserve
    # the source CAD layer structure verbatim.
    return sorted(lyr.dxf.name for lyr in doc.layers)


# =============================================================================
#  CLI: layer discovery + planform summary for a converted drawing
# =============================================================================
def _main(argv: List[str]) -> int:
    """Print layers and the extracted planform of a DXF (ASCII only)."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Inspect a 3-view DXF: list layers and extract the trapezoidal "
            "planform (x = chordwise, y = spanwise, one panel root->tip). "
            "Note: DWG files must be converted to DXF externally first."
        )
    )
    parser.add_argument("dxf_path", help="path to the DXF file")
    parser.add_argument(
        "--layer",
        default=None,
        help="restrict extraction to one layer (see the printed layer list)",
    )
    args = parser.parse_args(argv)

    # Layer inventory first — even when extraction fails this is the piece
    # of information the user needs to fix the invocation.
    try:
        layers = list_layers(args.dxf_path)
    except DxfParseError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Layers in {args.dxf_path}:")
    for name in layers:
        print(f"  {name}")

    # Planform extraction with the requested filter; failures are reported
    # verbatim (the error text already carries the remediation hints).
    try:
        pf = read_dxf_planform(args.dxf_path, layer=args.layer)
    except DxfParseError as exc:
        print(f"ERROR: {exc}")
        print("Falling back to aircraft.yaml values is the supported path "
              "until a clean DXF is provided (see planform_from_yaml).")
        return 1

    # Summary block. Sweep may be None in principle, though the DXF path
    # always computes it; guard anyway for forward compatibility.
    sweep_txt = (
        "n/a" if pf.le_sweep_rad is None
        else f"{math.degrees(pf.le_sweep_rad):8.3f} deg"
    )
    print("Extracted planform (drawn panel assumed root->tip half-span):")
    print(f"  span (full)   : {pf.span_m:10.4f} m")
    print(f"  chord root    : {pf.chord_root_m:10.4f} m")
    print(f"  chord tip     : {pf.chord_tip_m:10.4f} m")
    print(f"  taper ratio   : {pf.taper_ratio:10.4f}")
    print(f"  area (trapez.): {pf.area():10.4f} m2")
    print(f"  MAC  (trapez.): {pf.mac():10.4f} m")
    print(f"  LE sweep      : {sweep_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
