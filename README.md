# Glasair III Vortex Generator Placement Study

Automated CFD pipeline to find the optimal vortex generator (VG) for a
Stoddard-Hamilton Glasair III — the geometry that buys the most stall recovery
for the least cruise drag, so the owner can 3D-print the part and a placement
jig and install a real set on the airplane. **Trust deltas, not absolutes** —
RANS near CLmax carries 5–10% uncertainty on absolute numbers, so every
conclusion is framed as VG-on vs VG-off (or A vs B) on identical meshes and
settings.

> **Why this repo exists.** The Glasair III is a slick, fast, short-wing
> homebuilt with a relatively sharp stall. Vortex generators — small vanes that
> stir high-energy air down into the boundary layer — can tame that stall and
> lower approach/landing speeds, but the folklore around them (how tall, how
> far apart, what shape, what angle, what they cost in cruise) is inconsistent
> and largely un-quantified for *this* airfoil. This project replaces the
> hangar-talk with numbers: a GPU-accelerated CFD sweep over VG height, shape,
> incidence, spacing, and chord position, scoring each design on the only two
> things that matter to the pilot — **does it un-stall the wing, and what does
> it cost me at cruise.**

---

## ⭐ The VG optimization study (the main event)

> **Status: ~80% complete (2026-06-15), running autonomously on a GPU cluster
> of one (RTX 5090). Live results: [`gpu/rapidcfd/06-14-26_results.md`](gpu/rapidcfd/06-14-26_results.md).
> Final dated report `06-15-26_results.md` + STL/jig + spanwise install plan land
> when the last cases drain.**

### The winner so far 🏆

**A 12 mm-long, 8 mm-tall delta (triangular-ramp) vane, set at 10° incidence,
spaced ~50 mm, counter-rotating pairs, front tips at 7% chord.**

| | result | vs the clean stalled wing |
| --- | --- | --- |
| **Stall recovery** (α = 18°) | Cl 1.748, Cd 0.0926, L/D 18.9 | Cd **−79%**, L/D 3.6 → 18.9 |
| **Cruise tax** (α = 2°, 200 mph) | Cd 0.01232 | **+15.8%** — the lowest of 40+ configs |
| **Cruise speed lost** | **~1–2.6 kt** | (vs ~6–13 kt for the naive config) |

In plain terms: it takes a deeply stalled wing (drag quadrupled, lift collapsing)
and **reattaches the flow** — cutting drag ~79% and roughly **quintupling L/D** —
while costing only a couple of knots at cruise. It's also small and cheap to
3D-print.

### The five things that decided it

| Lever | Finding |
| --- | --- |
| **Shape** | The simple **delta (sharp triangular ramp) beats every fancier shape** tested — rectangular plate, Stolspeed swept fin, trapezoid (cropped delta), gothic (concave swept LE), and a cambered airfoil-section vane. The sharp ramp sheds the tightest, steadiest vortex. (Trapezoid & gothic make *more lift* but cost more drag/cruise; the airfoil-section went unsteady and lost lift.) |
| **Height** | **Micro-VGs work.** 6, 8, and 12 mm all recover the stall (it's a gentle gradient, not a cliff). 8 mm is the sweet spot — nearly the 12 mm's stall recovery, steadier flow, *lower* cruise tax (short vane hides in the thin cruise boundary layer), and a smaller print. 16 mm is *too tall* (spoiler effect, lift −31%). |
| **Incidence** | **8–10° is optimal**, not the 15° the literature inherits. 8° minimizes drag, 10° maximizes lift; 20° over-yaws. |
| **Spacing (pitch)** | Wider is better for cruise (fewer vanes), down to a floor: 50–70 mm all work well, but **35 mm is catastrophic** — crowded vanes merge into one ragged vortex and the wing *re-stalls*. |
| **Chord position** | **7% chord (front tips)** is the sweet spot — matches both the IMP74 flight-test number and Stolspeed's field-proven 8–12% band. Pushing the row aft to 15%+ keeps stall-class drag (the vane ends up inside the separated flow). |

### What was tested (40+ configurations)

A 2D RANS slice of one VG pitch (periodic spanwise) on the gapless
LS(1)-0413 wing section at the aileron station, swept across:

- **Shapes:** rectangular plate · delta ramp · Stolspeed swept fin · trapezoid
  (l = 4h cropped delta) · gothic planform · cambered airfoil-section
- **Heights:** 6 · 8 · 10 · 12 · 16 mm
- **Incidence:** 5 · 8 · 10 · 12 · 15 · 20°
- **Pitch:** 35 · 50 · 60 · 70 · 90 · 110 mm
- **Chord position:** 7 · 15 · 30 · 45%
- **Layout:** counter-rotating pairs vs single-alternating; toe-out vs toe-in
- **Conditions:** stall (α = 18°, Re 2.2e6) and cruise drag-tax (α = 2°,
  200 mph, Re 5.52e6) for every serious contender; stall-onset checks at 16°;
  a stall-development polar (α = 15/16/17/18) on the top configs *(in progress)*

### The cruise-vs-stall tradeoff, honestly

A *passive* VG can only add drag when the flow is attached, so none of these
"raise" cruise speed — the best achievable is a stall fix that costs ~nothing
at cruise. The 8 mm micro-delta gets close: it hides inside the thin cruise
boundary layer (≈ no drag) yet still bites the thick stall boundary layer at
high alpha. The realistic airplane-level penalty for a 25–55% span install is
**~1–2.6 kt** off a ~224 kt true cruise.

### Progressive spanwise stall (the install plan) *(in progress)*

The Glasair III wing is swept and tapered. The plan is one printed vane placed
in **two spacing zones**: **wider pitch inboard** (weaker vortices → root stalls
*first*, so the ailerons stay effective and the nose drops cleanly) and
**tighter pitch outboard** over the ailerons (strongest attachment → roll
control held deepest into the stall). VGs are oriented to the *airflow* (a
spanwise reference line square to the fuselage centerline), not the swept
leading edge. CFD for the inboard "stalls-first" zone is running now.

### How the study is run

The solver is **RapidCFD** (a CUDA/Thrust GPU fork of OpenFOAM 2.3) running on
an RTX 5090, single-precision with double-accumulated reductions for stable
convergence. Meshes are built with OpenFOAM v2506 `snappyHexMesh`
(~2.4–2.7 M cells/case). Each case runs a two-stage scheme ramp (upwind →
limitedLinearV) and is scored on the last-500-iteration mean. The whole sweep is
orchestrated by a chain of GPU-idle-gated queues so exactly one case solves at a
time, with a disk janitor that reclaims finished-case mesh files to keep the run
flat over days. Everything regenerates from `gpu/rapidcfd/build_cases.py`.

```
gpu/rapidcfd/
  build_cases.py        # case factory — all shapes/heights/angles/pitches; `--only NAME...`
  run_all.sh            # WSL runner (mesh → GPU solve → copy results home)
  report.py             # tail-averaged Cl/Cd table + VG-vs-clean deltas
  06-14-26_results.md   # dated study report (the full force table + findings)
  results/<case>/       # per-case forceCoeffs + logs
  assets/               # generated VG + wing STLs
```

**Methodology caveats** (also in the dated report): steady RANS on a separated
flow gives limit-cycle averages, so trust the *ranking* over the third digit;
kOmegaSST is fully turbulent (no transition model) and known to *under*-predict
streamwise-vortex strength, so the real VG benefit is likely a touch *better*
than shown; the periodic slice is an infinite array (no tip/fuselage effects).

---

## The original toolkit & validation pipeline

The sections below document the geometry toolkit, the `aircraft.yaml`
single-source-of-truth parameter system, and the clean-section validation
pipeline (XFOIL + NASA anchors) that underpin the study above. The VG study
also covers the horizontal stabilizer (underside ahead of the elevator) and
vertical fin as future surfaces; the wing is done first.

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
| M0 | Repo scaffold, `aircraft.yaml`, geometry toolkit + unit tests | Done |
| M1 | Phase-1 clean 2D validation (driver + gate + XFOIL/NASA anchors) | Done (toolkit); CPU clean-section cross-check used to validate the GPU pipeline |
| GPU pivot | RapidCFD (OpenFOAM-2.3 CUDA fork) built for the RTX 5090; transform-patch BC bug solved via cyclicAMI sides | Done |
| **VG study** | **40+ resolved configs — shape × height × incidence × pitch × position × layout, at stall and cruise** | **~80% — champion found (8 mm delta, β10, ~50 mm); polar + final report in progress** |
| Deliverables | Final dated report, printable VG STL (curved base), wing-hugging placement jig, progressive spanwise install plan | Pending (after the last cases drain) |

*(The original M2–M5 milestones — jBAY fvOptions, multi-study sweep runner,
per-study reporting — were superseded by the direct GPU-resolved-VG approach,
which models the actual vane geometry rather than a momentum source.)*
