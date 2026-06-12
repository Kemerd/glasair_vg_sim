# License

## Original work

Copyright (c) 2026 D Everett Hinton

Except as noted under *Third-party components* below, all original code,
scripts, case definitions, geometry tooling, and documentation in this
repository are licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Third-party components and carve-outs

The MIT grant above does NOT extend to the following; each remains governed
by its own upstream license (see also `docs/CITATIONS.md`):

* **FluidX3D-derived files** (`gpu/fluidx3d/setup_glasair.cpp`,
  `gpu/fluidx3d/defines_glasair.hpp`, and any other file adapted from
  ProjectPhysX/FluidX3D source): FluidX3D is distributed under a custom
  NON-COMMERCIAL license by Dr. Moritz Lehmann. These adaptations inherit
  that restriction — no commercial use without upstream permission.
  https://github.com/ProjectPhysX/FluidX3D

* **OpenFOAM / RapidCFD**: the solvers themselves are not vendored in this
  repository, but case dictionaries reference them and local patches to
  RapidCFD (GPLv3, SimFlowCFD/RapidCFD-dev) are maintained in a separate
  working tree. Any distribution of those patches falls under GPLv3.

* **XFOIL** (`tools/xfoil/`, binary not committed): GPLv2, M. Drela / MIT.
  Provenance and re-download instructions in `tools/xfoil/SOURCE.md`.

* **NASA technical data** (`validation/nasa/`): digitized from NASA
  TM X-72843; U.S. government work, public domain. Digitization errors are
  ours, not NASA's.

* **Aircraft geometry inputs** (DXF measurements, `aircraft.yaml` values):
  derived from Stoddard-Hamilton Glasair drawings for personal,
  experimental-aircraft use by the owner; not cleared for redistribution
  as manufacturer data.

## No airworthiness warranty

Nothing in this repository constitutes aeronautical engineering advice or
certified flight data. Simulation results here inform PERSONAL experimentation
on an Experimental-category aircraft; any aerodynamic modification must be
validated by flight test per the operating limitations of the aircraft.
