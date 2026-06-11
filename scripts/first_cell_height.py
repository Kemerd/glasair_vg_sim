# -*- coding: utf-8 -*-
"""
First-cell height calculator for y+ targeted wall-resolved meshes.

The project meshing rule (spec section 4) is y+ < 1 on all viscous walls,
which requires choosing the wall-adjacent cell height BEFORE meshing. This
tool turns a flight condition (speed, chord, air properties) into:

    * Reynolds number
    * estimated wall shear / friction velocity (flat-plate correlation)
    * the first-cell height that hits a requested y+
    * a turbulent boundary-layer thickness estimate plus a prism-layer
      schedule (count needed to bury the BL at a given growth ratio)

Correlations used (documented so reviewers can audit the arithmetic):
    Cf  = (2*log10(Re) - 0.65)^-2.3        Schlichting, turbulent flat plate
    tau = 0.5 * Cf * rho * U^2
    u_t = sqrt(tau / rho)
    y1  = y_plus * nu / u_t
    d99 = 0.37 * c / Re^(1/5)              classic 1/7th-power-law estimate
These are sizing estimates only — the actual y+ achieved is verified per
case from the solution (yPlus function object) and the case fails the gate
if max y+ > 2 over more than 1% of wall faces.

Defaults come from aircraft.yaml (approach speed, MAC, ISA sea level), so
running with no arguments answers the Phase-1 question directly. All YAML
access goes through geometry.units.load_aircraft — the single conversion
choke point — so this script honors whatever unit tags the master file
carries instead of assuming them. The only conversion done locally is the
mph <-> m/s handling for the human-facing CLI flag, and that too uses the
geometry.units factor table rather than a private constant.

Run:  python scripts/first_cell_height.py [--speed-mph V] [--chord-m C]
      [--yplus Y] [--growth G] [--layers N]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
#  Repo-root bootstrap (same pattern as scripts/smoke_m0.py)
# -----------------------------------------------------------------------------
#  Invoked as "python scripts/first_cell_height.py", Python puts scripts/ on
#  sys.path rather than the repo root, which would hide the geometry package.
#  Inserting this file's grandparent directory mirrors conftest.py and keeps
#  the script runnable from any working directory.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geometry.units import UNIT_TO_SI, load_aircraft, parse_quantity  # noqa: E402


def load_defaults() -> dict:
    """Pull approach speed, MAC, and air properties from aircraft.yaml.

    load_aircraft already returns SI floats for every unit-tagged quantity,
    so no factors are applied here — if the YAML author switches a tag (say
    mac from ft to in), the defaults below stay correct automatically.
    """
    ac = load_aircraft(REPO / "aircraft.yaml")
    return {
        "speed_mps": ac.conditions.approach_speed,      # m/s (tag-converted)
        "chord_m": ac.wing.mac,                         # m   (tag-converted)
        "rho": ac.atmosphere.density,                   # kg/m^3
        "nu": ac.atmosphere.kinematic_viscosity,        # m^2/s
    }


def first_cell_height(speed: float, chord: float, rho: float, nu: float,
                      y_plus: float) -> dict:
    """Flat-plate sizing chain; returns every intermediate for the report."""
    re = speed * chord / nu
    cf = (2.0 * math.log10(re) - 0.65) ** -2.3          # Schlichting fit
    tau_w = 0.5 * cf * rho * speed * speed
    u_tau = math.sqrt(tau_w / rho)
    y1 = y_plus * nu / u_tau
    d99 = 0.37 * chord / re ** 0.2                       # BL thickness at TE
    return {"re": re, "cf": cf, "tau_w": tau_w, "u_tau": u_tau,
            "y1": y1, "d99": d99}


def layers_to_cover(y1: float, growth: float, target: float) -> tuple[int, float]:
    """Prism layers (count, total thickness) to bury `target` from y1 at ratio."""
    total, h, n = 0.0, y1, 0
    while total < target:
        total += h
        h *= growth
        n += 1
    return n, total


def main() -> None:
    d = load_defaults()
    p = argparse.ArgumentParser(description="y+ first-cell height calculator")
    # The CLI keeps mph for the speed flag (pilots and the flight-test notes
    # speak mph); the default shown in --help is the YAML approach speed
    # expressed back in mph through the same factor table used at load time.
    p.add_argument("--speed-mph", type=float,
                   default=d["speed_mps"] / UNIT_TO_SI["mph"],
                   help="freestream speed in mph (default: approach from aircraft.yaml)")
    p.add_argument("--chord-m", type=float, default=d["chord_m"],
                   help="reference chord in meters (default: MAC from aircraft.yaml)")
    p.add_argument("--yplus", type=float, default=1.0, help="target y+ (default 1)")
    p.add_argument("--growth", type=float, default=1.2,
                   help="prism growth ratio (spec cap 1.2)")
    p.add_argument("--rho", type=float, default=d["rho"], help="density kg/m3")
    p.add_argument("--nu", type=float, default=d["nu"], help="kinematic viscosity m2/s")
    a = p.parse_args()

    # Convert the human-units CLI value through the choke-point helper so the
    # mph factor lives in exactly one place in the repository.
    speed = parse_quantity({"value": a.speed_mph, "unit": "mph"},
                           key_path="cli.speed_mph")
    r = first_cell_height(speed, a.chord_m, a.rho, a.nu, a.yplus)
    n_layers, covered = layers_to_cover(r["y1"], a.growth, r["d99"])

    # Plain-ASCII report, aligned for quick reading in a Windows console.
    print(f"speed            : {a.speed_mph:10.1f} mph  ({speed:.2f} m/s)")
    print(f"chord            : {a.chord_m:10.4f} m")
    print(f"Reynolds number  : {r['re']:10.3e}")
    print(f"flat-plate Cf    : {r['cf']:10.5f}")
    print(f"friction velocity: {r['u_tau']:10.4f} m/s")
    print(f"y+ target        : {a.yplus:10.2f}")
    print(f"first cell height: {r['y1']*1000:10.4f} mm   ({r['y1']:.3e} m)")
    print(f"est. BL d99 (TE) : {r['d99']*1000:10.2f} mm")
    print(f"prism schedule   : {n_layers} layers @ growth {a.growth} covers "
          f"{covered*1000:.2f} mm (spec minimum 30 layers)")


if __name__ == "__main__":
    main()
