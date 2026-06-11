# -*- coding: utf-8 -*-
"""
Extract Glasair III planform/control-surface measurements from the factory
3-view DXFs (converted from ref/Glasair-III-3-View-Drawing.dwg by the owner).

This script is intentionally SPECIFIC to these three files:
    geometry/dxf/glasair_topview.dxf    x = fore-aft (in), y = spanwise (in)
    geometry/dxf/glasair_sideview.dxf   x = fore-aft (in), y = vertical (in)
    geometry/dxf/glasair_frontview.dxf  x = spanwise (in), y = vertical (in)
All drawings verified 1:1 scale in inches ($INSUNITS=1): topview spanwise
extent 279.84 in vs factory span 23' 3-5/16" = 279.31 in (0.2% agreement).

Feature identification strategy (see comments at each step): the views are
line/arc drawings with no layer separation, so features are recognized by
length, orientation, and left/right symmetry about the aircraft centerline.
Outputs geometry/dxf/measured.yaml plus a console report; aircraft.yaml is
updated manually from that file so every change stays reviewable in git.

Run:  python scripts/measure_dxf.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ezdxf
import yaml

REPO = Path(__file__).resolve().parent.parent
DXF_DIR = REPO / "geometry" / "dxf"


# ---------------------------------------------------------------------------
#  Generic line catalog
# ---------------------------------------------------------------------------
@dataclass
class Seg:
    """A LINE entity reduced to what feature-matching needs."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle(self) -> float:
        """Orientation folded into [0, 180): 0 = +x aligned, 90 = +y aligned."""
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0

    def x_at_y(self, y: float) -> float:
        """Linear interpolation/extrapolation of x along the segment at height y."""
        t = (y - self.y1) / (self.y2 - self.y1)
        return self.x1 + t * (self.x2 - self.x1)

    def y_span(self) -> tuple[float, float]:
        return (min(self.y1, self.y2), max(self.y1, self.y2))


def load_lines(name: str) -> list[Seg]:
    """All LINE entities from a DXF modelspace as Seg records."""
    doc = ezdxf.readfile(DXF_DIR / f"{name}.dxf")
    out = []
    for e in doc.modelspace():
        if e.dxftype() == "LINE":
            out.append(Seg(e.dxf.start.x, e.dxf.start.y, e.dxf.end.x, e.dxf.end.y))
    return out


def pick(lines: list[Seg], pred) -> list[Seg]:
    """Filter helper kept tiny on purpose; predicates document the geometry."""
    return [s for s in lines if pred(s)]


# ---------------------------------------------------------------------------
#  Topview: wing LE/TE, flap+aileron hinge, aileron break, stab planform
# ---------------------------------------------------------------------------
def measure_topview() -> dict:
    lines = load_lines("glasair_topview")

    # Aircraft centerline: the drawing is symmetric, so the midpoint of the
    # overall spanwise extent is the fuselage centerline.
    ys = [v for s in lines for v in (s.y1, s.y2)]
    y_cl = 0.5 * (min(ys) + max(ys))
    half_span = max(ys) - y_cl

    # Wing feature lines on the RIGHT side (y > centerline): the three longest
    # near-spanwise lines are, by x position at the root, LE / hinge / TE.
    # (LE sweeps slightly aft, hinge and TE sweep forward on this planform.)
    right_span = pick(
        lines,
        lambda s: s.length > 100 and 75 < s.angle < 105
        and min(s.y1, s.y2) > y_cl + 15,
    )
    right_span.sort(key=lambda s: s.x_at_y(y_cl + 25))   # sort by root-side x
    if len(right_span) != 3:
        raise RuntimeError(f"expected 3 wing spanwise lines, got {len(right_span)}")
    le, hinge, te = right_span

    # Wing/fuselage junction = inboard end of the LE line; tip = outboard end.
    y_root = min(le.y1, le.y2)
    y_tip_line = max(te.y1, te.y2)          # TE line reaches furthest outboard

    def chord_at(y: float) -> float:
        return te.x_at_y(y) - le.x_at_y(y)

    # Aileron inboard edge: a short chordwise line out near the tip that
    # bridges exactly from the hinge line to the TE line (the flap/aileron
    # break rib). Identified by orientation, length, and that bridging fit.
    breaks = pick(
        lines,
        lambda s: 5 < s.length < 12 and (s.angle < 8 or s.angle > 172)
        and min(s.y1, s.y2) > y_cl + 60,
    )
    ail_inb = None
    for s in breaks:
        y = 0.5 * (s.y1 + s.y2)
        xs = sorted((s.x1, s.x2))
        if abs(xs[0] - hinge.x_at_y(y)) < 1.0 and abs(xs[1] - te.x_at_y(y)) < 1.0:
            ail_inb = y
            break
    if ail_inb is None:
        raise RuntimeError("aileron/flap break line not found")

    # Aileron outboard edge: the chordwise tip-joint line spanning LE..TE.
    tip_joints = pick(
        lines,
        lambda s: 25 < s.length < 40 and (s.angle < 5 or s.angle > 175)
        and min(s.y1, s.y2) > y_cl + 120,
    )
    ail_out = max(0.5 * (s.y1 + s.y2) for s in tip_joints)

    # Horizontal stab: hinge is the exactly-spanwise line pair at x ~ 223; LE
    # and TE are the remaining long spanwise lines in the tail region.
    stab_zone = pick(lines, lambda s: 30 < s.length < 60 and 80 < s.angle < 100
                     and min(s.x1, s.x2) > 200)
    stab_hinge = [s for s in stab_zone if abs(s.angle - 90) < 0.05]
    stab_le = [s for s in stab_zone if s.x_at_y(y_cl) < 210]
    stab_te = [s for s in stab_zone if s.x_at_y(y_cl) > 228 and abs(s.angle - 90) > 0.05]
    stab_tip_y = max(max(s.y1, s.y2) for s in stab_hinge)
    s_le, s_te = stab_le[0], stab_te[0]
    hinge_x = stab_hinge[0].x1

    stab_root_chord = s_te.x_at_y(y_cl) - s_le.x_at_y(y_cl)
    stab_tip_chord = s_te.x_at_y(stab_tip_y) - s_le.x_at_y(stab_tip_y)

    # Aileron mid-span station: center of the aileron span; the wing-section
    # study (Phase 3) cuts its 2.5D section here.
    ail_mid = 0.5 * (ail_inb + ail_out)

    return {
        "centerline_y_in": round(y_cl, 3),
        "half_span_in": round(half_span, 3),
        "span_in": round(2 * half_span, 3),
        "wing": {
            # Positive = aft sweep (LE x grows outboard). The LE line angle is
            # measured from +x, so spanwise-perfect is 90; deviation below 90
            # means the line leans aft going outboard.
            "le_sweep_deg": round(90 - le.angle, 3),
            "chord_centerline_in": round(chord_at(y_cl), 3),
            "chord_at_fuselage_junction_in": round(chord_at(y_root), 3),
            "chord_at_tip_in": round(chord_at(y_tip_line), 3),
            "fuselage_junction_eta": round((y_root - y_cl) / half_span, 4),
            "taper_ratio_junction_to_tip": round(chord_at(y_tip_line) / chord_at(y_root), 4),
            "hinge_chord_fraction_at_aileron_mid": round(
                (hinge.x_at_y(ail_mid) - le.x_at_y(ail_mid)) / chord_at(ail_mid), 4),
        },
        "aileron": {
            "inboard_eta": round((ail_inb - y_cl) / half_span, 4),
            "outboard_eta": round((ail_out - y_cl) / half_span, 4),
            "span_in": round(ail_out - ail_inb, 3),
            "mid_station_eta": round((ail_mid - y_cl) / half_span, 4),
            "mid_station_from_cl_in": round(ail_mid - y_cl, 3),
            "chord_at_mid_station_in": round(chord_at(ail_mid), 3),
            "chord_fraction": round(1 - (hinge.x_at_y(ail_mid) - le.x_at_y(ail_mid))
                                    / chord_at(ail_mid), 4),
        },
        "horizontal_tail": {
            "span_in": round(2 * (stab_tip_y - y_cl), 3),
            "root_chord_in": round(stab_root_chord, 3),
            "tip_chord_in": round(stab_tip_chord, 3),
            "taper_ratio": round(stab_tip_chord / stab_root_chord, 4),
            "hinge_x_in": round(hinge_x, 3),
            "elevator_chord_fraction_root": round(
                (s_te.x_at_y(y_cl) - hinge_x) / stab_root_chord, 4),
            "elevator_chord_fraction_tip": round(
                (s_te.x_at_y(stab_tip_y) - hinge_x) / stab_tip_chord, 4),
            "area_trapezoid_ft2": round(
                0.5 * (stab_root_chord + stab_tip_chord) * 2 * (stab_tip_y - y_cl) / 144.0, 3),
        },
    }


# ---------------------------------------------------------------------------
#  Sideview: fin LE, rudder hinge, rudder TE  (true lengths: vertical surface)
# ---------------------------------------------------------------------------
def measure_sideview() -> dict:
    lines = load_lines("glasair_sideview")

    # The three fin/rudder lines live aft of x=200 above the tailcone. They
    # are distinguished by orientation: LE rakes ~40 deg, hinge ~68 deg,
    # rudder TE ~77 deg (all measured from horizontal).
    tail = pick(lines, lambda s: s.length > 35 and min(s.x1, s.x2) > 200
                and max(s.y1, s.y2) > 80)
    fin_le = next(s for s in tail if 30 < s.angle < 50)
    hinge = next(s for s in tail if 60 < s.angle < 72)
    rud_te = next(s for s in tail if 72 < s.angle < 85)

    # Fin root = LE inboard end height; tip = LE top end. The rudder runs
    # deeper than the fin root, down the tailcone to the TE line's bottom.
    y_root = min(fin_le.y1, fin_le.y2)
    y_tip = max(fin_le.y1, fin_le.y2)
    y_rud_bot = min(rud_te.y1, rud_te.y2)

    def x_on(s: Seg, y: float) -> float:
        return s.x_at_y(y)

    root_chord = x_on(rud_te, y_root) - x_on(fin_le, y_root)
    rudder_root = x_on(rud_te, y_root) - x_on(hinge, y_root)
    # Hinge rake measured from vertical: relevant when mirroring VG rows on
    # both fin sides ahead of the hinge (Study 3).
    rake = math.degrees(math.atan2(abs(hinge.x2 - hinge.x1), abs(hinge.y2 - hinge.y1)))

    return {
        "fin_root_y_in": round(y_root, 3),
        "fin_tip_y_in": round(y_tip, 3),
        "fin_height_in": round(y_tip - y_root, 3),
        "rudder_bottom_y_in": round(y_rud_bot, 3),
        "root_chord_incl_rudder_in": round(root_chord, 3),
        "rudder_chord_fraction_root": round(rudder_root / root_chord, 4),
        "hinge_rake_from_vertical_deg": round(rake, 3),
    }


# ---------------------------------------------------------------------------
#  Frontview: dihedral from the long wing projection lines
# ---------------------------------------------------------------------------
def measure_frontview() -> dict:
    lines = load_lines("glasair_frontview")

    # Wing projections are the only >100 in near-horizontal lines. Their
    # slopes give dihedral along the LE and TE respectively (they differ
    # because the section twists the projection; the factory figure is 3 deg).
    wings = pick(lines, lambda s: s.length > 100 and (s.angle < 6 or s.angle > 174))
    angles = sorted(round(min(s.angle, 180 - s.angle), 3) for s in wings)
    return {"projection_dihedral_deg": angles, "factory_value_deg": 3.0}


# ---------------------------------------------------------------------------
def main() -> None:
    report = {
        "_provenance": "measured from owner-converted factory 3-view DXFs, inches, 1:1",
        "topview": measure_topview(),
        "sideview": measure_sideview(),
        "frontview": measure_frontview(),
    }
    out = DXF_DIR / "measured.yaml"
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
