# Glasair III Vortex Generator Placement Study

Automated OpenFOAM CFD pipeline to find optimal vortex generator (VG) placement,
height, and spacing on a Stoddard-Hamilton Glasair III, for three surfaces: wing
ahead of the ailerons, horizontal stabilizer underside ahead of the elevator, and
vertical fin (both sides) ahead of the rudder. The pipeline narrows the flight-test
matrix to 1-2 candidate VG layouts per surface; tuft-visualization flight testing
remains the final verification step. **Trust deltas, not absolutes** — RANS near
CLmax carries 5-10% uncertainty on absolute numbers, so every conclusion is framed
as VG-on vs VG-off (or position A vs position B) on identical meshes and settings.

## Repository layout

```
aircraft.yaml        # master parameter file — single source of truth, unit-tagged
glasair3-vg-cfd-spec.md  # full project specification
geometry/            # airfoil coords, units/YAML loader, DXF parsing, STL generation
  dxf/               # factory 3-view DXFs (user-provided)
cases/               # OpenFOAM case templates (one per study type)
sweeps/              # sweep definitions (YAML) + staged sweep runner
scripts/             # mesh/run/post automation
validation/          # XFOIL + NASA experimental comparison data & gate scripts
results/             # parsed outputs, plots, summary CSVs (gitignored except summaries)
tests/               # pytest suite for the geometry toolkit
ref/                 # source documents (3-view, VT slides, Impulse #74 article)
```

## Data sources

| Tag | Source | Used for |
| --- | --- | --- |
| [3VIEW] | Stoddard-Hamilton factory 3-view, drawing rev G, 4/26/90 (`ref/`, DXF/DWG) | Span, length, dihedral; exact chord/control-surface geometry once DXFs are parsed |
| [VT2005] | Virginia Tech Glasair III analysis slides, Carobine/Fitzwater/Jackson, 2005-03-30 | Areas, taper, MAC, weights, stall/cruise conditions |
| [IMP74] | Strausak VG flight-test article, IE Impulse #74, Jan 2021, pp. 7-9 (Lancair Legacy) | Centers of the VG sweep ranges (7% chord, 50/90 mm spacing, 15 deg incidence) |
| [UIUC] | LS(1)-0413 (GA(W)-2) coordinates, UIUC Airfoil Database — <https://m-selig.ae.illinois.edu/ads/coord/ls413.dat> | Wing section (`geometry/ls413.dat`); NASA wind-tunnel data anchors Phase-1 validation |

All dimensional values live in `aircraft.yaml` as `{value, unit}` mappings and are
converted to SI exactly once, by `geometry/units.py`. No script hard-codes an
aircraft parameter.

## Quickstart (Windows, Python 3.10+)

```
pip install -r requirements.txt
python -m pytest
```

Sanity-check the master parameter file (prints an SI summary table):

```
python -m geometry.units
```

## Environment bootstrap (Ubuntu / WSL2)

The geometry/sweep tooling runs natively on Windows; the solver runs under
**WSL2 Ubuntu** with **ESI OpenFOAM >= v2406** (ESI flavor required for the
`kOmegaSSTLM` gamma-Re_theta transition model — mandatory for Phase 1 onward,
since the LS(1)-0413 on a composite skin carries significant laminar flow).

**1. Install WSL2 + Ubuntu** (elevated PowerShell on the Windows host):

```
wsl --install -d Ubuntu
```

**Known issue:** WSL on this workstation currently fails to start with HNS error
`0x8007271d` (Host Network Service). An administrator fix (restart the Host
Network Service / `netsh winsock reset` network reset) or a reboot is required
before the M1 solver bring-up can proceed. Geometry work (M0) is unaffected.

**2. Add the openfoam.com (ESI) Debian repository** (inside Ubuntu):

```
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
```

**3. Install the solver:**

```
sudo apt-get update
sudo apt-get install openfoam2506-default
```

**Version pinning:** the package name above is the current ESI release; the
EXACT version actually installed must be pinned at M1 in BOTH this README and
the `solver` block of `aircraft.yaml` (currently `openfoam_version: null`,
deliberately unpinned until install). Every case directory must be reproducible
from `aircraft.yaml` + sweep YAML + that pinned version.

**4. Activate the OpenFOAM environment** (append to `~/.bashrc` to persist;
substitute the pinned version number):

```
source /usr/lib/openfoam/openfoam2506/etc/bashrc
```

**5. XFOIL + Python tooling on the Linux side** (XFOIL drives the Phase-1
validation baseline; the Python deps mirror `requirements.txt`):

```
sudo apt-get install xfoil python3-pip
pip3 install -r requirements.txt
```

**Python version:** the spec targets **Python 3.11+** for the pre/post stack;
the toolkit is kept compatible with — and currently also runs on — the
workstation's native **Python 3.10.11** (see Quickstart above), so nothing in
this repository may use 3.11-only syntax.

## M1 validation (Phase 1, clean 2D section)

One command drives the whole Phase-1 clean-section validation:

```
./scripts/run_validation.sh            # WSL/Ubuntu side (sources openfoam2506)
python scripts/run_validation.py       # equivalent, any side; same flags
```

The driver (a) ensures the XFOIL baseline polars exist in `validation/xfoil/`
(generating them via `validation/xfoil_polar.py` when missing), (b) builds the
AoA sweep case set through `scripts/build_validation_case.py` — alpha -4 to
+20 deg, 2-deg steps with 1-deg refinement from +8 up, Re 3e6 / mesh level 0
by default (`--re` / `--level` to change), (c) runs blockMesh / checkMesh /
decomposePar / simpleFoam per case when a solver is reachable — two cases in
flight, 8 cores each by default (`--jobs` / `--cores`), logs captured per
tool, convergence gated by `scripts/parse_forces.py`, boundary-layer + Cf
extraction via `scripts/extract_bl.py` on converged cases — and (d) evaluates
the Phase-1 gates with `validation/compare_gate.py`, append-updating
`validation/report.md` (one section per configuration, regenerated in place).

**Current state:** OpenFOAM is not installed yet (WSL repair in progress), so
the driver stops after case generation with a `SOLVER MISSING — generated N
cases ready` summary and exit 0; the gate report is still written, with every
solver-dependent row marked SKIPPED and its reason. Re-running the same
command after the install picks up where it left off (cases with a recorded
convergence PASS are skipped; `--force` re-solves them).

**2D chord/speed convention:** all Phase-1 2D cases use a unit chord
(c = 1.0 m) and set the freestream speed from the target Reynolds number,
`U = Re * nu / c`, with `nu` taken from the `atmosphere` block of
`aircraft.yaml` (ISA sea level, 1.4607e-5 m^2/s):

| Re | U (m/s) | Mach (a = 340.3 m/s) |
| --- | --- | --- |
| 1.5e6 | 21.9 | 0.064 |
| 3e6 | 43.8 | 0.129 |
| 6e6 | 87.6 | 0.258 |

The Re 6e6 point exceeds the M < 0.2 quasi-incompressible guideline at unit
chord. **Decision: keep unit chord and keep the Re 6e6 point as a trend
point.** Both simpleFoam (incompressible) and XFOIL (incompressible at our
settings) ignore compressibility, so the RANS-vs-XFOIL comparison remains
self-consistent, matching common practice for incompressible airfoil
validation studies; absolute force coefficients at Re 6e6 carry the usual
compressibility caveat and are used for Re-trend information only. (The
alternative — scaling the chord up to lower U — was rejected because it
changes nothing in either incompressible code except the bookkeeping.)

**Gates** (spec Phase-1 item 4, evaluated by `validation/compare_gate.py`):

1. dCl/dalpha within 5% of XFOIL, linear fit over alpha in [-2, +6] deg;
2. Clmax within 10% and stall AoA within 2 deg of the NASA experimental
   polar — hand-digitized lift curves in `validation/nasa/digitized/` from
   TM X-72843 fig. 5 (LS(1)-0413, Re 2.2/4.3/6.4e6, transition fixed at
   0.075c) and CR-145139 fig. 2(a) (Re 2.2e6, strips), provenance and
   uncertainty per `validation/nasa/SOURCES.md`; the gate matches the CSV
   to the requested Re by filename token (1% tolerance) and reports an
   explicit SKIPPED reason when no anchor at that Re exists — never
   silently passed;
3. upper-surface transition x/c within 10% chord of XFOIL's e^N `xtr_top`
   at matching Ncrit. The RANS transition pickup is defined as the x/c where
   wall Cf first rises through its post-minimum inflection (first local
   maximum of dCf/dx downstream of the laminar Cf minimum) — implementation
   and guards documented in `validation/compare_gate.py`.

## Milestone status

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Repo scaffold, `aircraft.yaml`, geometry toolkit + unit tests | In progress |
| M1 | Phase-1 clean 2D validation case end-to-end, gate report | In progress (driver + gate ready; solver bring-up blocked on WSL repair) |
| M2 | jBAY fvOptions source + unit validation case | Pending |
| M3 | Study-1 sweep runner, 2-case smoke sweep with auto post | Pending |
| M4 | Full Study-1 screening sweep; Studies 2-3 templates | Pending |
| M5 | Reporting pipeline (`results/REPORT.md` per study) | Pending |
