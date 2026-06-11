# -*- coding: utf-8 -*-
"""
Generate the 'speck control' test article: the clean slice wing plus ONE tiny
bump at mid-span on the suction surface.

Purpose (experimental control, owner's idea): the span-periodic slice lets
the separating shear layer roll into artificially coherent 2D vortex rollers
that pump extra lift into the clean baseline. A single small 3D disturbance
seeds spanwise instability and breaks that coherence while adding essentially
no aerodynamic device of its own -- so (clean+speck vs clean) measures the
ROLL ARTIFACT, separating it from the genuine VG effects in the deltas.

The speck: an 8 mm cube, half-sunk into the skin at 30% chord, mid-span.
Small enough to be aerodynamically boring, big enough (~5 lattice cells) for
the lattice to feel it.

Run:  python gpu/fluidx3d/make_speck_wing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import trimesh

from geometry.airfoil import load_airfoil, resample_airfoil
from geometry.stl_gen import extrude_section
from geometry.units import load_aircraft

ASSETS = REPO / "gpu" / "fluidx3d" / "assets"
SPAN = 0.25                 # slice span, m (matches the suite articles)
SPECK = 0.008               # speck edge length, m
X_FRAC = 0.30               # chordwise station of the speck


def main() -> None:
    ac = load_aircraft(REPO / "aircraft.yaml")
    chord = ac.wing.aileron.chord_at_mid_station

    coords = resample_airfoil(load_airfoil(REPO / "geometry" / "ls413.dat"),
                              n_points=241, te="blunt")
    wing = extrude_section(coords, chord, SPAN)

    # Upper-surface height at the speck station (same interpolation approach
    # as the VG generator: loop is Selig order, upper surface first).
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1][::-1]
    y_surf = float(np.interp(X_FRAC, upper[:, 0], upper[:, 1])) * chord

    speck = trimesh.creation.box(extents=(SPECK, SPECK, SPECK))
    # Half-sunk into the skin at mid-span (z=0 -- extrude_section centers the
    # span on z=0) so the union is robust and the protrusion is ~4 mm.
    speck.apply_translation((X_FRAC * chord, y_surf, 0.0))

    model = trimesh.boolean.union([wing, speck])
    out = ASSETS / "wing_speck_s0.25m.stl"
    model.export(out)
    print(f"wrote {out.name}: clean wing + {SPECK*1000:.0f} mm speck at "
          f"{X_FRAC*100:.0f}%c mid-span | watertight={model.is_watertight} "
          f"faces={len(model.faces)}")


if __name__ == "__main__":
    main()
