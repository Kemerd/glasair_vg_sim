"""2.5D extruded-section STL generation for snappyHexMesh wall geometry.

Builds the watertight solids consumed by the spanwise-periodic section studies
(glasair3-vg-cfd-spec.md, Phase 0 item 2(c) and Phases 3-5):

  (a) single-element extruded wing section (LS(1)-0413 at the aileron station),
  (b) two-element stab section with TE-up elevator and an OPEN hinge gap,
  (c) two-element fin section with deflected rudder and the same gap machinery.

COORDINATE CONVENTION (fixed across the whole pipeline)
  x = chordwise, LE -> TE, meters          (airfoil arrays arrive chord-normalized)
  y = thickness direction, +y "up"         (suction side of the wing)
  z = spanwise extrusion, midspan at z = 0 (periodic-domain convention: blockMesh
      cyclic patches sit at z = +/- span/2, so centering the solid keeps the
      refinement boxes and VG line definitions symmetric about the domain origin)

CONTROL-SURFACE SIGN CONVENTION (flight-mechanics standard)
  positive deflection  =  trailing edge DOWN  (+y is up)
  An elevator nose-up command is therefore a NEGATIVE deflection (TE up); the
  stab study per the spec deflects TE-up, so its callers pass negative radians.

HINGE-GAP PROVENANCE
  Gap width 1/16 in = 1.5875 mm exactly, per the Stoddard-Hamilton factory
  drawing, carried per surface in aircraft.yaml (wing.aileron.hinge_gap,
  horizontal_tail.elevator.hinge_gap, vertical_tail.rudder.hinge_gap). Gaps
  are never sealed: the gap is the leak path that drives hinge-line
  separation, which is exactly the flow feature the VG study must capture.

GAPPED BASELINES vs CLEAN SOLIDS (include_gap)
  The leak path exists on the real aircraft at ALL deflections, including
  zero, so spec Phase 3 runs the 0 deg baseline WITH the open gap and the
  generators default to include_gap=True: every case - baseline included -
  is a main+control pair sharing the identical split topology, keeping the
  whole sweep in one mesh family so case-to-case deltas are not polluted by
  topology changes. include_gap=False is the deliberate opt-out that emits
  one gapless solid for the Phase-1 clean validation geometry only, and it
  refuses a nonzero deflection (a deflected control has no gapless form).

ACHIEVED OPENING vs NOMINAL GAP (read before using sweep results)
  The clearance cove (see split_at_hinge) keeps the deflected control from
  fouling the fixed element, but it makes the achieved external opening
  DEFLECTION-DEPENDENT: always >= the nominal gap, measured at 2.3-3.3 mm
  across the 10-25 deg stab sweep against the 1.5875 mm nominal. Every
  two-element case is therefore audited (measure_gap_metrics) and the
  numbers are printed and returned in SectionCaseResult so post-processing
  can correlate aerodynamic results against the gap geometry actually built.

Watertightness is a hard requirement (these are viscous walls for snappy);
every generator validates and reports it before returning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

PathLike = Union[str, Path]

# Deflections smaller than this are treated as exactly zero; the only consumer
# is the include_gap=False guard, which must reject any real deflection.
_ZERO_DEFLECTION_TOL: float = 1.0e-12


# =============================================================================
#  Result containers — what a sweep runner needs to log per generated case
# =============================================================================

@dataclass(frozen=True)
class GapMetrics:
    """2D clearance audit of one main+control hinge case (all meters).

    nominal_gap_m    -- the gap width the half-plane cuts were built with
                        (1/16 in = 1.5875e-3 from aircraft.yaml)
    min_clearance_m  -- tightest main/control separation anywhere in the
                        channel (shapely distance); the cove construction
                        guarantees this is >= nominal_gap_m
    opening_upper_m  -- external slot width where the gap meets the UPPER
                        outer mold line: distance from the upper cove lip
                        (aft-most fixed-element point above the hinge
                        waterline) to the control surface
    opening_lower_m  -- same measurement at the lower outer mold line

    The cove makes the achieved opening deflection-dependent (always >= the
    nominal gap): on the stab section it measures 2.3-3.3 mm across the
    10-25 deg sweep vs the 1.5875 mm nominal. These numbers exist so sweep
    post-processing can correlate results against the gap actually built,
    instead of assuming the drawing value survived the construction.
    """

    nominal_gap_m: float
    min_clearance_m: float
    opening_upper_m: float
    opening_lower_m: float


@dataclass(frozen=True)
class SectionCaseResult:
    """Summary of one generated section case (returned by every gen_*).

    paths          -- exported STL files: [single] for include_gap=False,
                      [main, control] for a gapped two-element case
    chord_m        -- section chord the airfoil loop was scaled to
    span_m         -- spanwise extrusion length (periodic-domain width)
    deflection_rad -- control deflection used (positive = TE down)
    hinge_frac     -- hinge station as x/c actually used for the split
    include_gap    -- True when the open hinge gap was built
    gap            -- GapMetrics for the two-element pair, None when the
                      case is the single-solid clean baseline
    """

    paths: List[Path]
    chord_m: float
    span_m: float
    deflection_rad: float
    hinge_frac: float
    include_gap: bool
    gap: Optional[GapMetrics]


# =============================================================================
#  Shapely helpers — polygon hygiene
# =============================================================================

def _as_single_polygon(geom: BaseGeometry, label: str) -> Polygon:
    """Coerce a shapely result into one valid, non-empty Polygon.

    Boolean ops on digitized airfoil loops can return GeometryCollections or
    MultiPolygons carrying degenerate slivers (zero-width artifacts where a
    cut line grazes a vertex). The aerodynamic element is always the largest
    area component; anything else is numerical debris and is dropped here.

    :param geom:  raw output of an intersection/difference call
    :param label: element name for error messages ('main', 'control', ...)
    :raises ValueError: if no usable polygon remains after cleanup
    """
    # Empty output means the cut removed the entire element - always a setup
    # error (e.g. hinge_frac outside [0,1]) that must not pass silently.
    if geom.is_empty:
        raise ValueError(f"{label}: cut produced an empty geometry")

    # Collect polygonal components; lines/points from tangent contacts are
    # discarded because they carry no area and cannot be extruded.
    if isinstance(geom, Polygon):
        candidates = [geom]
    else:
        candidates = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    if not candidates:
        raise ValueError(f"{label}: cut produced no polygonal area")

    # Keep the dominant component, then repair residual invalidity (bow-tie
    # self-touch etc.) with buffer(0): it re-nodes the ring and rebuilds a
    # clean valid polygon without moving any vertex - the standard shapely
    # idiom for healing near-degenerate digitized outlines.
    poly = max(candidates, key=lambda g: g.area)
    if not poly.is_valid:
        poly = poly.buffer(0)
        # buffer(0) may itself split a bow-tie into parts; recurse once to
        # pick the dominant piece of the repaired output.
        if not isinstance(poly, Polygon):
            return _as_single_polygon(poly, label)
    if poly.is_empty or poly.area <= 0.0:
        raise ValueError(f"{label}: repaired polygon has no area")
    return poly


def section_polygon(coords: np.ndarray, chord_m: float) -> Polygon:
    """Scale a chord-normalized airfoil loop to meters and return a Polygon.

    :param coords:  (N,2) Selig loop (TE-upper -> LE -> TE-lower), x/c in [0,1]
    :param chord_m: physical chord in meters; uniform scale on both axes so
                    the thickness distribution stays geometrically similar
    :returns: valid shapely Polygon in meters, exterior oriented CCW
    """
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError(f"airfoil loop must be (N,2) with N >= 3, got {pts.shape}")
    if chord_m <= 0.0:
        raise ValueError(f"chord must be positive, got {chord_m}")

    # Shapely closes the ring implicitly; a duplicated first/last vertex from
    # the Selig file is harmless. Scale is applied before construction so the
    # validity predicate runs on the physical geometry.
    poly = Polygon(pts * chord_m)

    # Digitized loops occasionally self-intersect by sub-micron amounts at a
    # blunt TE closure; buffer(0) re-nodes the ring without moving vertices,
    # which is why it is the repair of choice (no smoothing, no offset).
    if not poly.is_valid:
        poly = _as_single_polygon(poly.buffer(0), "section")

    # Enforce CCW exterior so the extrusion triangulation yields outward
    # normals and a positive enclosed volume.
    return orient(poly, sign=1.0)


def _mid_thickness_y(poly: Polygon, x: float) -> float:
    """Mid-thickness y of a section polygon at chordwise station x (meters).

    Probes the section with a vertical line; the in-polygon part runs from
    the lower to the upper surface, so the midpoint of its y-bounds is the
    local camber-line height. Shared by the hinge-point construction and the
    gap audit so both reference the identical waterline.
    """
    minx, miny, maxx, maxy = poly.bounds
    # One chord of overhang guarantees the probe pierces both surfaces.
    pad = maxx - minx
    probe = LineString([(x, miny - pad), (x, maxy + pad)])
    xsec = poly.intersection(probe)
    if xsec.is_empty:
        raise ValueError(f"station x={x:.6f} m lies outside the section")
    return 0.5 * (xsec.bounds[1] + xsec.bounds[3])


# =============================================================================
#  Extrusion — 2.5D solids on the periodic-domain z convention
# =============================================================================

def _extrude_centered(poly: Polygon, span_m: float) -> trimesh.Trimesh:
    """Extrude a section polygon along z and center it on the midspan plane.

    trimesh extrudes from z=0 to z=span; the half-span shift afterwards puts
    midspan at z=0, which is the periodic-domain convention used by every
    spanwise-periodic study in this repo (cyclic patches at z = +/- span/2).
    """
    if span_m <= 0.0:
        raise ValueError(f"span must be positive, got {span_m}")

    # earcut handles the concave airfoil outline; the polygon arrives CCW so
    # the cap normals come out pointing away from the solid.
    mesh = trimesh.creation.extrude_polygon(poly, height=span_m)
    mesh.apply_translation([0.0, 0.0, -0.5 * span_m])

    # Fuse coincident cap/wall vertices so trimesh sees closed fans at every
    # edge - this is what makes is_watertight report True for snappy intake.
    mesh.merge_vertices()
    if not mesh.is_watertight:
        # Fallback for pathological triangulations: stitch any open edge loop
        # and re-orient. Reaching this branch is unexpected for clean input.
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    return mesh


def extrude_section(coords: np.ndarray, chord_m: float, span_m: float) -> trimesh.Trimesh:
    """Build a watertight single-element 2.5D solid from an airfoil loop.

    :param coords:  chord-normalized Selig loop, see section_polygon()
    :param chord_m: chord in meters (x extent of the result)
    :param span_m:  spanwise extrusion length in meters (z extent), solid
                    centered so midspan sits at z = 0
    """
    return _extrude_centered(section_polygon(coords, chord_m), span_m)


def extrude_two_element(
    main_poly: Polygon,
    ctrl_poly: Polygon,
    span_m: float,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Extrude a fixed element + control surface pair into one case geometry.

    Both solids share the identical z convention (midspan at z=0) so they drop
    into the same periodic domain; snappy receives them as two named STL
    regions and meshes the open hinge gap between them.
    """
    return _extrude_centered(main_poly, span_m), _extrude_centered(ctrl_poly, span_m)


# =============================================================================
#  Hinge split — vertical cut, open gap, rigid rotation, clearance cove
# =============================================================================

def split_at_hinge(
    coords: np.ndarray,
    hinge_frac: float,
    deflection_rad: float,
    gap_m: float,
    chord_m: float,
) -> Tuple[Polygon, Polygon]:
    """Split a section at the hinge line and rigidly deflect the control.

    Construction (spec Phase 0 item 2(c)):
      * vertical cut plane at x = hinge_frac * chord
      * FIXED element  = everything forward of (hinge_x - gap_m)
      * CONTROL surface = everything aft of hinge_x
      * the gap_m strip between them stays OPEN - never sealed, because the
        gap leak is the very mechanism behind hinge-line separation
      * hinge point = mid-thickness of the section at hinge_x
      * control rotates rigidly about the hinge point by deflection_rad

    SIGN CONVENTION (do not change): positive deflection = trailing edge DOWN
    with +y up. Elevator nose-up command is a NEGATIVE deflection (TE up);
    the stab study therefore supplies negative values.

    CLEARANCE COVE: a plain vertical cut would foul at deflection - the
    control's nose corner sits ~half-thickness t from the hinge and sweeps
    t*sin(delta) forward of the cut plane (about 9 mm at 25 deg on a 0.7 m
    NACA 0010 section), far more than the 1.59 mm gap. Real elevators solve
    this with a cove in the fixed surface; the same is done here: a circular
    arc centered on the hinge point, radius = (largest hinge-distance of any
    control boundary point that ends up forward of the cut plane) + gap_m,
    is subtracted from the fixed element. Because every forward-swung control
    point then lies at least one full gap width inside the cove, the minimum
    main/control clearance is >= gap_m at the requested deflection and the
    leak path stays open everywhere.

    ACHIEVED OPENING: the cove radius tracks the deflected control, so the
    external opening at the outer mold line is deflection-dependent and
    always >= gap_m (measured 2.3-3.3 mm across 10-25 deg on the stab
    section vs the 1.5875 mm nominal). Callers that need the as-built
    numbers should run measure_gap_metrics on the returned pair - the
    gen_* entry points do this automatically and return the audit.

    :param coords:         chord-normalized Selig loop
    :param hinge_frac:     hinge chordwise station as x/c in (0,1)
    :param deflection_rad: control deflection, positive = TE down (radians)
    :param gap_m:          physical hinge gap in meters (open leak path)
    :param chord_m:        section chord in meters
    :returns: (main_polygon, control_polygon), both valid shapely Polygons
    """
    if not 0.0 < hinge_frac < 1.0:
        raise ValueError(f"hinge_frac must be in (0,1), got {hinge_frac}")
    if gap_m <= 0.0:
        raise ValueError(f"gap must be positive (open gap required), got {gap_m}")

    # Physical-scale section and a padding margin for the half-plane boxes;
    # one chord of pad guarantees the boxes envelop the section completely.
    poly = section_polygon(coords, chord_m)
    minx, miny, maxx, maxy = poly.bounds
    pad = chord_m
    hinge_x = hinge_frac * chord_m

    # Hinge point: mid-thickness at the hinge station, via the shared probe
    # helper so the gap audit later references the identical waterline.
    hinge_y = _mid_thickness_y(poly, hinge_x)

    # ---- half-plane cuts -----------------------------------------------------
    # Fixed element keeps x <= hinge_x - gap; control keeps x >= hinge_x.
    # The strip between the two cut planes is discarded: that is the open gap.
    main = _as_single_polygon(
        poly.intersection(box(minx - pad, miny - pad, hinge_x - gap_m, maxy + pad)),
        "main",
    )
    ctrl = _as_single_polygon(
        poly.intersection(box(hinge_x, miny - pad, maxx + pad, maxy + pad)),
        "control",
    )

    # ---- rigid control rotation ----------------------------------------------
    # shapely's rotate() is CCW-positive in the x-y plane. With +y up, TE-down
    # (positive deflection) is a CLOCKWISE rotation of the aft element, hence
    # the sign flip here. This is the single place the convention is encoded.
    ctrl = affinity.rotate(
        ctrl, -deflection_rad, origin=(hinge_x, hinge_y), use_radians=True
    )

    # ---- clearance cove in the fixed element ----------------------------------
    # Clip the rotated control boundary to the forward side of the cut plane.
    # The clip inserts exact plane-crossing vertices, so the largest hinge
    # distance over the clipped coordinates bounds EVERY forward control point
    # (segment points never exceed the farther endpoint's distance, and the
    # forward region sits inside the convex hull of its boundary).
    fwd = ctrl.exterior.intersection(box(minx - pad, miny - pad, hinge_x, maxy + pad))
    r_fwd = 0.0
    if not fwd.is_empty:
        # The clip may return a LineString, MultiLineString or stray Points;
        # walk whatever came back and track the max distance from the hinge.
        parts = list(getattr(fwd, "geoms", [fwd]))
        for part in parts:
            for vx, vy in getattr(part, "coords", []):
                r_fwd = max(r_fwd, math.hypot(vx - hinge_x, vy - hinge_y))

    # At zero deflection the control's blunt face lies exactly on the cut
    # plane and the clip returns it, so r_fwd equals the nose-corner radius
    # and the cove is carved consistently across the whole deflection family.
    r_cove = r_fwd + gap_m
    # 64 segments per quarter circle keeps the polygonized-arc sagitta error
    # below ~2 microns at these radii - negligible against the gap width.
    cove = Point(hinge_x, hinge_y).buffer(r_cove, quad_segs=64)
    main = _as_single_polygon(main.difference(cove), "main")

    return main, ctrl


# =============================================================================
#  Gap audit — measure the channel actually built, never assume the nominal
# =============================================================================

def measure_gap_metrics(
    main: Polygon,
    ctrl: Polygon,
    hinge_y: float,
    nominal_gap_m: float,
) -> GapMetrics:
    """Measure the as-built gap channel of a two-element hinge case.

    Two observables, both in the 2D section plane:

      (a) min_clearance  -- shapely distance(main, ctrl): the tightest point
          anywhere between the elements. The cove guarantees >= nominal gap.
      (b) opening widths -- the external slot at each outer-mold-line lip.
          The cove disk always swallows the entire blunt cut face of the
          fixed element (both face corners sit within r_fwd + gap of the
          hinge, inside the convex cove), so the aft-most fixed-element
          vertex on each side of the hinge waterline is exactly the point
          where the cove cut breaks through the OML. The opening is the
          distance from that lip to the control surface.

    The opening is deflection-dependent by construction (the cove radius
    tracks the swung control): 2.3-3.3 mm measured across the 10-25 deg
    stab sweep vs the 1.5875 mm nominal. Report, do not hide.

    :param main:          fixed-element polygon from split_at_hinge
    :param ctrl:          control polygon from split_at_hinge
    :param hinge_y:       hinge waterline (mid-thickness y at the hinge x)
    :param nominal_gap_m: the gap the cuts were built with, for the record
    :returns: GapMetrics with all four values in meters
    """
    # (a) global minimum clearance between the two wall polygons.
    min_clearance = float(main.distance(ctrl))

    # (b) locate the OML lips: split the fixed-element boundary vertices at
    # the hinge waterline and take the aft-most vertex on each side. The
    # cut/cove booleans insert exact crossing vertices, so the lips are real
    # boundary points, not interpolation guesses.
    xy = np.asarray(main.exterior.coords, dtype=float)
    upper = xy[xy[:, 1] >= hinge_y]
    lower = xy[xy[:, 1] < hinge_y]
    if upper.size == 0 or lower.size == 0:
        # A main element entirely on one side of the waterline means the
        # split geometry is broken; refuse to report meaningless numbers.
        raise ValueError("main element does not straddle the hinge waterline")
    lip_upper = upper[int(np.argmax(upper[:, 0]))]
    lip_lower = lower[int(np.argmax(lower[:, 0]))]

    # Slot width = lip-to-control distance; the control surface is the other
    # side of the external opening at each lip.
    opening_upper = float(Point(lip_upper).distance(ctrl))
    opening_lower = float(Point(lip_lower).distance(ctrl))

    return GapMetrics(
        nominal_gap_m=float(nominal_gap_m),
        min_clearance_m=min_clearance,
        opening_upper_m=opening_upper,
        opening_lower_m=opening_lower,
    )


def gap_report(metrics: GapMetrics) -> str:
    """One-line ASCII summary of a GapMetrics audit (millimeters)."""
    # Millimeters keep these small numbers human-readable next to the meter
    # scale bbox figures in mesh_report.
    return (
        f"gap nominal={metrics.nominal_gap_m * 1e3:.4f} mm "
        f"min_clearance={metrics.min_clearance_m * 1e3:.3f} mm "
        f"opening_upper={metrics.opening_upper_m * 1e3:.3f} mm "
        f"opening_lower={metrics.opening_lower_m * 1e3:.3f} mm"
    )


# =============================================================================
#  Mesh QA helpers
# =============================================================================

def check_watertight(mesh: trimesh.Trimesh) -> bool:
    """True if the mesh is a closed 2-manifold suitable as a snappy wall."""
    return bool(mesh.is_watertight)


def mesh_report(mesh: trimesh.Trimesh) -> str:
    """One-line ASCII QA summary: watertight flag, face count, volume, bbox."""
    # extents is the axis-aligned bounding-box size; volume is only meaningful
    # when watertight, so the flag leads the line for quick log scanning.
    dx, dy, dz = (float(e) for e in mesh.extents)
    return (
        f"watertight={mesh.is_watertight} faces={len(mesh.faces)} "
        f"volume={float(mesh.volume):.6e} m^3 "
        f"bbox dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} m"
    )


def _write_stl(mesh: trimesh.Trimesh, path: Path) -> Path:
    """Export one solid as ASCII STL and log its QA line.

    ASCII (not binary) STL is deliberate: snappyHexMesh accepts both, but
    ASCII files diff cleanly in git and the section solids are tiny (a few
    thousand facets), so the size penalty is irrelevant.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="stl_ascii")
    print(f"[stl_gen] wrote {path} | {mesh_report(mesh)}")
    return path


def _generate_section_case(
    coords: np.ndarray,
    chord_m: float,
    span_m: float,
    deflection_rad: float,
    hinge_frac: float,
    gap_m: float,
    out_dir: PathLike,
    stem: str,
    include_gap: bool = True,
) -> SectionCaseResult:
    """Shared worker behind the three gen_* entry points.

    include_gap=True (default) ALWAYS builds the main+control pair with the
    open hinge gap - zero deflection included, because the leak path exists
    on the real aircraft at all deflections and the 0 deg baseline must share
    the mesh family of the deflected cases (spec Phase 3). The pair is
    audited with measure_gap_metrics before extrusion and the numbers ride
    back on the result.

    include_gap=False is the deliberate single-solid opt-out for the Phase-1
    clean validation geometry: one gapless solid, no hinge machinery. It is
    only defined at zero deflection - a deflected control with no gap is not
    a geometry this pipeline can honestly produce, so that combination raises.
    """
    out = Path(out_dir)

    # ---- clean single-solid opt-out (Phase-1 validation geometry only) -------
    if not include_gap:
        # Refuse a deflected gapless request outright: silently dropping the
        # gap was exactly the failure mode this flag exists to eliminate.
        if abs(deflection_rad) >= _ZERO_DEFLECTION_TOL:
            raise ValueError(
                "include_gap=False is the clean single-solid baseline and "
                "cannot represent a deflected control (got "
                f"{math.degrees(deflection_rad):+.3f} deg); use "
                "include_gap=True for any deflected case"
            )
        mesh = extrude_section(coords, chord_m, span_m)
        paths = [_write_stl(mesh, out / f"{stem}.stl")]
        return SectionCaseResult(
            paths=paths, chord_m=chord_m, span_m=span_m,
            deflection_rad=deflection_rad, hinge_frac=hinge_frac,
            include_gap=False, gap=None,
        )

    # ---- gapped two-element case (the standard path, baseline included) ------
    main_poly, ctrl_poly = split_at_hinge(
        coords, hinge_frac, deflection_rad, gap_m, chord_m
    )

    # Audit the channel actually built: the cove makes the achieved opening
    # deflection-dependent, so the as-built numbers are measured and logged
    # for every case rather than trusting the nominal drawing value.
    hinge_y = _mid_thickness_y(
        section_polygon(coords, chord_m), hinge_frac * chord_m
    )
    metrics = measure_gap_metrics(main_poly, ctrl_poly, hinge_y, gap_m)
    print(f"[stl_gen] {gap_report(metrics)}")

    # Extrude both elements on the shared midspan-centered z convention and
    # write one named STL per element (snappy consumes them as two regions).
    main_mesh, ctrl_mesh = extrude_two_element(main_poly, ctrl_poly, span_m)
    paths = [
        _write_stl(main_mesh, out / f"{stem}_main.stl"),
        _write_stl(ctrl_mesh, out / f"{stem}_control.stl"),
    ]
    return SectionCaseResult(
        paths=paths, chord_m=chord_m, span_m=span_m,
        deflection_rad=deflection_rad, hinge_frac=hinge_frac,
        include_gap=True, gap=metrics,
    )


# =============================================================================
#  YAML-driven generators (one per study surface)
# =============================================================================

def _naca_designation(label: str) -> str:
    """Extract the 4-digit code from a label like 'NACA 0010'.

    aircraft.yaml stores tail sections as human-readable strings; only the
    digit group feeds naca4_coords(). A missing code is a data error, not
    something to paper over with a silent default.
    """
    m = re.search(r"(\d{4})", label)
    if m is None:
        raise ValueError(f"cannot parse NACA 4-digit designation from {label!r}")
    return m.group(1)


def _require(value, key: str):
    """Reject a YAML null on a parameter with no defensible fallback.

    aircraft.yaml v2 carries DXF-measured values for everything these
    generators consume; a null here means the master file regressed, and
    fabricating a stand-in would silently violate the single-source-of-truth
    rule. Fail with the exact key so the YAML author can fix the file.
    """
    if value is None:
        raise ValueError(
            f"aircraft.yaml entry '{key}' is null; the schema v2 file carries "
            "a measured value here - refusing to substitute a guess"
        )
    return value


def _resolve_repo_path(yaml_path: PathLike, rel: str) -> Path:
    """Resolve a repo-relative path from aircraft.yaml against the yaml's dir.

    aircraft.yaml references files like 'geometry/ls413.dat' relative to the
    repo root (where the yaml itself lives), so anchoring on the yaml parent
    keeps the generators working from any working directory.
    """
    p = Path(rel)
    return p if p.is_absolute() else Path(yaml_path).resolve().parent / p


def _fmt_defl(deflection_rad: float) -> str:
    """Filename-safe deflection tag in degrees, e.g. 'defl-20.0deg'."""
    return f"defl{math.degrees(deflection_rad):+.1f}deg"


def gen_wing_section_stl(
    yaml_path: PathLike,
    out_dir: PathLike,
    aileron_deflection_rad: float = 0.0,
    span_m: Optional[float] = None,
    hinge_frac: Optional[float] = None,
    include_gap: bool = True,
) -> SectionCaseResult:
    """Study-1 wing section: LS(1)-0413 at the aileron mid-span station.

    Chord comes straight from the DXF-measured wing.aileron.chord_at_mid_station
    (0.9022 m at eta = 0.835); the chord_root/chord_tip trapezoid is evaluated
    at the same station purely as a consistency gate - the two derive from
    independent DXF measurements, so disagreement beyond 1% means the wing
    block was corrupted and generation refuses to proceed (RuntimeError).

    :param yaml_path:              aircraft.yaml location
    :param out_dir:                output directory for STL files
    :param aileron_deflection_rad: positive = TE down (see module convention)
    :param span_m:  spanwise domain width; default = 2 VG-pair spacings at the
                    outboard pitch (spec Phase 3 requires N >= 2 pair spacings)
    :param hinge_frac: override for the hinge station; default is the measured
                    wing.aileron.hinge_chord_fraction (0.8013 [DXF])
    :param include_gap: True (default) builds the main+aileron pair with the
                    open hinge gap at ANY deflection, zero included; False is
                    the single-solid Phase-1 clean baseline (0 deg only).
    :returns: SectionCaseResult with paths and (when gapped) the gap audit
    """
    # Deferred sibling imports keep the pure-geometry helpers in this module
    # importable in isolation (unit tests, ad-hoc scripting) and avoid any
    # import-order coupling between concurrently developed toolkit modules.
    from geometry.airfoil import load_airfoil, resample_airfoil
    from geometry.dxf_reader import planform_from_yaml
    from geometry.units import load_aircraft

    ac = load_aircraft(yaml_path)

    # Spanwise station of the section cut: aileron mid-span ([DXF] measured,
    # 116.622 in from centerline) normalized by the half-span.
    half_span = 0.5 * ac.wing.span
    station = _require(ac.wing.aileron.mid_span_station, "wing.aileron.mid_span_station")
    eta = min(max(station / half_span, 0.0), 1.0)

    # Section chord: the DXF measurement at the station is authoritative; the
    # trapezoid interpolation cross-checks it. Both are independent reads of
    # the same drawing, so >1% disagreement flags a corrupted wing block.
    chord_meas = _require(
        ac.wing.aileron.chord_at_mid_station, "wing.aileron.chord_at_mid_station"
    )
    chord_trap = planform_from_yaml(ac).chord_at(eta)
    if abs(chord_trap - chord_meas) > 0.01 * chord_meas:
        raise RuntimeError(
            "wing planform inconsistency in aircraft.yaml: measured "
            f"chord_at_mid_station = {chord_meas:.4f} m but the chord_root/"
            f"chord_tip trapezoid gives {chord_trap:.4f} m at eta={eta:.4f} "
            "(>1% apart); fix the wing block before generating sections"
        )
    chord_m = chord_meas

    # Section coordinates: published LS(1)-0413 loop, cosine-resampled with a
    # closed blunt TE - the real wing has finite TE thickness and snappy
    # meshes a blunt base far more robustly than a knife edge.
    coords = resample_airfoil(
        load_airfoil(_resolve_repo_path(yaml_path, ac.wing.airfoil_file)),
        n_points=241,
        te="blunt",
    )

    # Periodic-domain width default: two counter-rotating VG pair spacings at
    # the outboard pitch (Strausak 50 mm center), the spec minimum of N = 2.
    if span_m is None:
        span_m = 2.0 * ac.vg_defaults.wing.spacing_outboard

    # Hinge station: the DXF-measured hinge line at the mid-span station,
    # carried directly as wing.aileron.hinge_chord_fraction (0.8013).
    if hinge_frac is None:
        hinge_frac = _require(
            ac.wing.aileron.hinge_chord_fraction, "wing.aileron.hinge_chord_fraction"
        )

    # Aileron gap from the wing block itself. The yaml flags this one entry
    # TODO (assumed to follow the elevator's 1/16 in convention), so it is
    # the single remaining assumption worth announcing on every invocation.
    gap_m = _require(ac.wing.aileron.hinge_gap, "wing.aileron.hinge_gap")
    print(
        "[stl_gen] WARNING: wing.aileron.hinge_gap is TODO-flagged in "
        "aircraft.yaml (assumed 1/16 in, same convention as the elevator); "
        "confirm against the factory drawing."
    )

    print(
        f"[stl_gen] wing section: chord={chord_m:.4f} m at eta={eta:.3f}, "
        f"span={span_m:.4f} m, deflection={math.degrees(aileron_deflection_rad):+.1f} deg, "
        f"include_gap={include_gap}"
    )
    return _generate_section_case(
        coords, chord_m, span_m, aileron_deflection_rad, hinge_frac, gap_m,
        out_dir, f"wing_section_{_fmt_defl(aileron_deflection_rad)}",
        include_gap=include_gap,
    )


def gen_stab_section_stl(
    yaml_path: PathLike,
    out_dir: PathLike,
    elevator_deflection_rad: float,
    span_m: Optional[float] = None,
    hinge_frac: Optional[float] = None,
    include_gap: bool = True,
) -> SectionCaseResult:
    """Study-2 stab section: NACA 0010 placeholder + TE-up elevator with gap.

    The deflection is passed through UNCHANGED: nose-up elevator command is
    TE-up, which is NEGATIVE in this module's sign convention, so the stab
    study caller supplies negative radians (e.g. -0.5 * max_up).

    :param span_m: default = 2 elevator VG spacings (Strausak 30 mm pitch),
                   the N = 2 periodic-domain minimum
    :param hinge_frac: override; default = 1 - elevator.chord_fraction = 0.66
                   from the DXF-measured span-mean elevator chord fraction
    :param include_gap: True (default) builds the gapped pair at any
                   deflection, zero included (the Phase-3 baseline carries
                   the gap); False = single-solid clean geometry, 0 deg only
    :returns: SectionCaseResult with paths and (when gapped) the gap audit
    """
    # Deferred sibling imports - see gen_wing_section_stl for rationale.
    from geometry.airfoil import naca4_coords
    from geometry.units import load_aircraft

    ac = load_aircraft(yaml_path)

    # Section: the actual stab profile is still TODO in aircraft.yaml; the
    # spec mandates a flagged NACA 0010 placeholder. Sharp TE is fine here -
    # the placeholder carries no measured TE thickness to honor.
    print(
        f"[stl_gen] WARNING: horizontal_tail.airfoil ({ac.horizontal_tail.airfoil!r}) "
        "is a TODO placeholder in aircraft.yaml; the as-built stab section is unmeasured."
    )
    coords = naca4_coords(_naca_designation(ac.horizontal_tail.airfoil), te="sharp")

    # Chord: DXF-measured stab root chord; the study section nominally sits
    # at the elevator VG row, refined when the row station is finalized.
    chord_m = _require(ac.horizontal_tail.chord_root, "horizontal_tail.chord_root")

    # Open hinge gap straight from the factory-drawing value in the yaml.
    gap_m = _require(
        ac.horizontal_tail.elevator.hinge_gap, "horizontal_tail.elevator.hinge_gap"
    )

    # Periodic width default: two VG spacings on the elevator row.
    if span_m is None:
        span_m = 2.0 * ac.vg_defaults.elevator.spacing

    # Hinge station from the DXF-measured elevator chord fraction: the yaml
    # stores c_elev/c (span-mean 0.34), so the hinge sits at 1 - 0.34 = 0.66c.
    if hinge_frac is None:
        cf = _require(
            ac.horizontal_tail.elevator.chord_fraction,
            "horizontal_tail.elevator.chord_fraction",
        )
        hinge_frac = 1.0 - cf

    print(
        f"[stl_gen] stab section: chord={chord_m:.4f} m, span={span_m:.4f} m, "
        f"deflection={math.degrees(elevator_deflection_rad):+.1f} deg (TE-up is negative), "
        f"include_gap={include_gap}"
    )
    return _generate_section_case(
        coords, chord_m, span_m, elevator_deflection_rad, hinge_frac, gap_m,
        out_dir, f"stab_section_{_fmt_defl(elevator_deflection_rad)}",
        include_gap=include_gap,
    )


def gen_fin_section_stl(
    yaml_path: PathLike,
    out_dir: PathLike,
    rudder_deflection_rad: float,
    span_m: Optional[float] = None,
    hinge_frac: Optional[float] = None,
    chord_m: Optional[float] = None,
    include_gap: bool = True,
) -> SectionCaseResult:
    """Study-3 fin section: NACA 0010 placeholder + deflected rudder with gap.

    The fin lies in a vertical plane on the aircraft, but the 2.5D section
    machinery is identical: here +y is simply the lateral direction and the
    sweep runner owns the left/right mapping. Positive deflection still means
    TE toward +y per the module convention. The DXF-measured hinge rake
    (vertical_tail.rudder.hinge_rake_from_vertical, 22.1 deg aft lean) is a
    3D planform feature that a constant-section extrusion cannot represent;
    the 2.5D study treats the hinge as normal to the section plane.

    :param chord_m: override for the fin section chord; default is the
                    DXF-measured vertical_tail.chord_root_incl_rudder
                    (44.778 in = 1.1374 m) at the fin root station
    :param hinge_frac: override; default = 1 - rudder.chord_fraction = 0.6288
                    from the DXF-measured rudder chord fraction at the root
    :param include_gap: True (default) builds the gapped pair at any
                    deflection, zero included; False = single-solid clean
                    geometry, 0 deg only
    :returns: SectionCaseResult with paths and (when gapped) the gap audit
    """
    # Deferred sibling imports - see gen_wing_section_stl for rationale.
    from geometry.airfoil import naca4_coords
    from geometry.units import load_aircraft

    ac = load_aircraft(yaml_path)

    # Placeholder symmetric section, same TODO status as the stab study.
    print(
        f"[stl_gen] WARNING: vertical_tail.airfoil ({ac.vertical_tail.airfoil!r}) "
        "is a TODO placeholder in aircraft.yaml; the as-built fin section is unmeasured."
    )
    coords = naca4_coords(_naca_designation(ac.vertical_tail.airfoil), te="sharp")

    # Fin chord: DXF-measured root chord including the rudder. Note this is
    # deliberately NOT vertical_tail.area (still null - the fin planform is
    # curved and its area is computed only when Study 3 needs it); the chord
    # is a direct drawing measurement and needs no area inversion.
    if chord_m is None:
        chord_m = _require(
            ac.vertical_tail.chord_root_incl_rudder,
            "vertical_tail.chord_root_incl_rudder",
        )

    # Same 1/16 in factory gap convention, taken from the rudder entry.
    gap_m = _require(ac.vertical_tail.rudder.hinge_gap, "vertical_tail.rudder.hinge_gap")

    # No rudder-specific VG pitch is defined yet; the elevator row spacing is
    # the closest analog (same Strausak source), so the periodic width default
    # reuses it: two pair spacings = N = 2 minimum.
    if span_m is None:
        span_m = 2.0 * ac.vg_defaults.elevator.spacing

    # Hinge station from the DXF-measured rudder chord fraction at the fin
    # root: c_rudder/c = 0.3712, so the hinge sits at 1 - 0.3712 = 0.6288c.
    if hinge_frac is None:
        cf = _require(
            ac.vertical_tail.rudder.chord_fraction, "vertical_tail.rudder.chord_fraction"
        )
        hinge_frac = 1.0 - cf

    print(
        f"[stl_gen] fin section: chord={chord_m:.4f} m, span={span_m:.4f} m, "
        f"deflection={math.degrees(rudder_deflection_rad):+.1f} deg, "
        f"include_gap={include_gap}"
    )
    return _generate_section_case(
        coords, chord_m, span_m, rudder_deflection_rad, hinge_frac, gap_m,
        out_dir, f"fin_section_{_fmt_defl(rudder_deflection_rad)}",
        include_gap=include_gap,
    )
