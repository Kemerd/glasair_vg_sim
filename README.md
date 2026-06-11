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

## OpenFOAM environment plan (M1+)

The geometry/sweep tooling runs natively on Windows; the solver runs under
**WSL2 Ubuntu** with **ESI OpenFOAM >= v2406** (ESI flavor required for the
`kOmegaSSTLM` gamma-Re_theta transition model — mandatory for Phase 1 onward,
since the LS(1)-0413 on a composite skin carries significant laminar flow). The
exact OpenFOAM version gets pinned in the `solver` block of `aircraft.yaml` at
install time; every case directory must be reproducible from `aircraft.yaml` +
sweep YAML + that pinned version.

**Known issue:** WSL on this workstation currently fails to start with HNS error
`0x8007271d` (Host Network Service). An administrator fix (HNS service reset /
network reset) or a reboot is required before the M1 solver bring-up can proceed.
Geometry work (M0) is unaffected.

## Milestone status

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Repo scaffold, `aircraft.yaml`, geometry toolkit + unit tests | In progress |
| M1 | Phase-1 clean 2D validation case end-to-end, gate report | Pending |
| M2 | jBAY fvOptions source + unit validation case | Pending |
| M3 | Study-1 sweep runner, 2-case smoke sweep with auto post | Pending |
| M4 | Full Study-1 screening sweep; Studies 2-3 templates | Pending |
| M5 | Reporting pipeline (`results/REPORT.md` per study) | Pending |
