"""M0 acceptance smoke test for the Glasair III VG geometry toolkit.

End-to-end exercise of the Phase-0 geometry pipeline, run as a single
self-checking script (spec milestone M0):

  (a) load the master parameter file aircraft.yaml through
      geometry.units.load_aircraft and echo a few key SI values, proving
      the unit-conversion choke point works on the committed file;
  (b) ingest the committed LS(1)-0413 coordinates (geometry/ls413.dat) and
      cosine-resample them to the CFD-grade 241-point blunt-TE loop;
  (c) drive the YAML-backed STL generators end to end:
        * clean wing section (include_gap=False, the deliberate single-solid
          Phase-1 validation geometry -> ONE solid),
        * stab section at 0 deg WITH the open gap (include_gap=True, the
          Phase-3 gapped baseline -> TWO solids sharing the deflected cases'
          mesh family),
        * stab section with the elevator at -20 deg (TE up, the negative
          sign per the stl_gen flight-mechanics convention) and the open
          1/16 in = 1.5875 mm factory hinge gap -> TWO solids;
      everything lands in results/smoke/;
  (d) reload every STL that was written FROM DISK (not the in-memory mesh,
      so the exported artifact itself is what gets judged) and print an
      ASCII summary table of watertight status, enclosed volume, bounding
      box extents, and - for gapped cases - the as-built gap audit columns
      (minimum main/control clearance plus the external opening width at
      each outer-mold-line lip; the cove construction makes these
      deflection-dependent and >= the nominal gap, so they are reported
      per case rather than assumed).

Exit status is the acceptance verdict: 0 only when every solid on disk is
watertight AND every gapped case respects the cove clearance guarantee
(min clearance >= nominal gap); any failure (or any pipeline exception)
exits nonzero so CI and the milestone checklist can gate on this script.

Run from anywhere:  python scripts/smoke_m0.py
Plain-ASCII output only (safe on the default Windows console).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
#  Repo-root bootstrap
# -----------------------------------------------------------------------------
#  This script lives at <root>/scripts/smoke_m0.py. When invoked as
#  "python scripts/smoke_m0.py", Python puts scripts/ (not the repo root) on
#  sys.path, so the geometry package would be invisible. Inserting the parent
#  of this file's directory mirrors what conftest.py does for pytest and makes
#  the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import trimesh  # noqa: E402  (import after the path bootstrap on purpose)

from geometry.airfoil import load_airfoil, resample_airfoil  # noqa: E402
from geometry.stl_gen import gen_stab_section_stl, gen_wing_section_stl  # noqa: E402
from geometry.units import load_aircraft  # noqa: E402

# Fixed locations for this smoke run; results/smoke is disposable output and
# is recreated by the generators (mkdir parents=True) on every invocation.
YAML_PATH = REPO / "aircraft.yaml"
OUT_DIR = REPO / "results" / "smoke"

# The M0 acceptance deflection: elevator 20 deg trailing-edge UP. The module
# sign convention is positive = TE down, so TE-up is NEGATIVE radians.
ELEVATOR_DEFLECTION_RAD = math.radians(-20.0)

# Factory hinge gap every gapped case must carry: 1/16 in per the drawing,
# which is EXACTLY 1.5875e-3 m (0.0625 * 0.0254). The value is read from
# aircraft.yaml by the generators and cross-checked here so a YAML edit that
# silently changes any of the three gap entries fails the smoke test loudly.
EXPECTED_GAP_M = 1.5875e-3


def _banner(title: str) -> None:
    """Print a fixed-width ASCII section banner (keeps the log scannable)."""
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    """Run the full M0 smoke sequence; return the process exit code."""

    # ---- (a) aircraft.yaml through the units choke point ---------------------
    _banner("[1/4] load aircraft.yaml via geometry.units.load_aircraft")
    ac = load_aircraft(YAML_PATH)

    # Echo a handful of representative SI conversions: one length (ft -> m),
    # one angle (deg -> rad), one small length (in -> m). Seeing plausible
    # magnitudes here is the fastest human check that the parse succeeded.
    print(f"  wing.span                      = {ac.wing.span:.4f} m")
    print(f"  wing.mac                       = {ac.wing.mac:.4f} m")
    print(f"  wing.aileron.chord_at_mid      = {ac.wing.aileron.chord_at_mid_station:.4f} m")
    print(f"  horizontal_tail.chord_root     = {ac.horizontal_tail.chord_root:.4f} m")
    print(f"  vertical_tail.chord_root_incl  = {ac.vertical_tail.chord_root_incl_rudder:.4f} m")
    print(f"  vg_defaults.vane_incidence     = {ac.vg_defaults.vane_incidence:.6f} rad")

    # Hard-stop if any of the three per-surface gap entries drifted from the
    # factory 1/16 in callout; the gapped cases below would otherwise be
    # generated with the wrong leak path. All three surfaces carry the same
    # convention in the schema-v2 file.
    gaps = {
        "wing.aileron.hinge_gap": ac.wing.aileron.hinge_gap,
        "horizontal_tail.elevator.hinge_gap": ac.horizontal_tail.elevator.hinge_gap,
        "vertical_tail.rudder.hinge_gap": ac.vertical_tail.rudder.hinge_gap,
    }
    for key, gap_m in gaps.items():
        print(f"  {key:<38} = {gap_m * 1000.0:.4f} mm")
        if abs(gap_m - EXPECTED_GAP_M) > 1e-9:
            print(f"  FAIL: {key} {gap_m} m != expected {EXPECTED_GAP_M} m")
            return 1

    # ---- (b) LS(1)-0413 ingestion + 241-point blunt resample ------------------
    _banner("[2/4] resample geometry/ls413.dat (blunt TE, 241 points)")
    raw = load_airfoil(REPO / "geometry" / "ls413.dat")
    coords = resample_airfoil(raw, n_points=241, te="blunt")

    # Contract checks: exact point count, and a genuinely open (blunt) TE.
    # The TE gap is the |y| difference between the first and last loop points
    # (Selig order starts and ends at the trailing edge).
    te_gap = abs(float(coords[0, 1] - coords[-1, 1]))
    print(f"  raw points = {raw.shape[0]}, resampled points = {coords.shape[0]}")
    print(f"  blunt TE gap = {te_gap:.6f} c (must be > 0 for 'blunt')")
    if coords.shape != (241, 2):
        print(f"  FAIL: resample returned shape {coords.shape}, expected (241, 2)")
        return 1
    if te_gap <= 0.0:
        print("  FAIL: blunt resample produced a closed trailing edge")
        return 1

    # ---- (c) STL generation into results/smoke -------------------------------
    _banner("[3/4] generate STL solids into results/smoke/")
    cases = []

    # Clean wing section: include_gap=False is the deliberate single-solid
    # opt-out for Phase-1 validation geometry (no hinge split, no gap).
    cases.append(gen_wing_section_stl(
        YAML_PATH, OUT_DIR, aileron_deflection_rad=0.0, include_gap=False
    ))

    # Gapped stab BASELINE: 0 deg WITH the open gap (include_gap defaults to
    # True). The leak path exists at zero deflection on the real aircraft,
    # so the baseline must be the same two-element mesh family as the
    # deflected cases - this case proves that path works end to end.
    cases.append(gen_stab_section_stl(
        YAML_PATH, OUT_DIR, elevator_deflection_rad=0.0
    ))

    # Stab section, elevator -20 deg (TE up): two solids with the open
    # 1.5875 mm gap between them, hinge from the DXF-measured elevator
    # chord fraction (1 - 0.34 = 0.66c).
    cases.append(gen_stab_section_stl(
        YAML_PATH, OUT_DIR, elevator_deflection_rad=ELEVATOR_DEFLECTION_RAD
    ))

    # ---- (d) reload from disk and render the acceptance table -----------------
    _banner("[4/4] watertightness + gap audit of the exported STL files")

    # Column layout sized so typical values stay aligned; volumes are tiny
    # (1e-3..1e-1 m^3) hence scientific notation, bbox in plain meters, and
    # the three gap-audit columns in millimeters (single-solid rows show '-'
    # because a clean baseline has no channel to measure).
    header = (
        f"  {'file':<38} {'watertight':>10} {'volume_m3':>13} "
        f"{'dx_m':>8} {'dy_m':>8} {'dz_m':>8} "
        f"{'clr_mm':>7} {'opn_up':>7} {'opn_lo':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_ok = True
    for case in cases:
        # Pre-format the per-case gap columns once; both files of a pair
        # report the same channel audit (it is a property of the case, not
        # of either solid alone).
        if case.gap is not None:
            clr = f"{case.gap.min_clearance_m * 1e3:7.3f}"
            opu = f"{case.gap.opening_upper_m * 1e3:7.3f}"
            opl = f"{case.gap.opening_lower_m * 1e3:7.3f}"
        else:
            clr = opu = opl = f"{'-':>7}"

        for path in case.paths:
            # Reload the artifact from disk so the verdict applies to what
            # snappy will actually consume, not the pre-export in-memory
            # mesh. trimesh's default processing merges the duplicated
            # ASCII-STL facet vertices, which is required for the
            # closed-manifold test to be meaningful.
            mesh = trimesh.load(str(path), force="mesh")
            ok = bool(mesh.is_watertight)
            all_ok = all_ok and ok

            # Volume is only physically meaningful on a closed mesh; it is
            # still printed for leaky meshes because the magnitude hints at
            # the defect.
            dx, dy, dz = (float(e) for e in mesh.extents)
            print(
                f"  {Path(path).name:<38} {('YES' if ok else 'NO'):>10} "
                f"{float(mesh.volume):>13.6e} {dx:>8.4f} {dy:>8.4f} {dz:>8.4f} "
                f"{clr} {opu} {opl}"
            )

        # Gapped-case acceptance: the cove guarantee (min clearance >= the
        # nominal gap) must hold or the leak-path geometry is broken.
        if case.gap is not None and case.gap.min_clearance_m < case.gap.nominal_gap_m:
            all_ok = False
            print(
                f"  FAIL: min clearance {case.gap.min_clearance_m * 1e3:.3f} mm "
                f"< nominal gap {case.gap.nominal_gap_m * 1e3:.4f} mm"
            )

    print("  " + "-" * (len(header) - 2))
    n_files = sum(len(c.paths) for c in cases)
    if not all_ok:
        print("  RESULT: FAIL - watertightness or gap-clearance check failed")
        return 1
    print(
        f"  RESULT: PASS - all {n_files} exported solids are watertight and "
        "every gapped case respects the cove clearance guarantee"
    )
    return 0


if __name__ == "__main__":
    # Exit code is the acceptance verdict (0 = M0 smoke passed); exceptions
    # propagate naturally and yield a nonzero exit through the traceback.
    sys.exit(main())
