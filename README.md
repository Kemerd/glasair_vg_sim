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

> **Status: ✅ COMPLETE (2026-06-17). 63 GPU-CFD cases run autonomously on a
> cluster of one (RTX 5090), including a full α = 2°→22° stall polar. Final
> report: [`gpu/rapidcfd/06-17-26_results.md`](gpu/rapidcfd/06-17-26_results.md).
> Stall numbers below are the corrected peak-to-peak Clmax method.**

### 🏁 The leaderboard — knots gained vs knots paid

The bottom line for an installer: **how many knots of stall speed do you gain,
and how many knots of cruise do you pay?** Off the Glasair III's ~69.5 kt clean
stall and ~180 kt cruise. Stall reduction is now computed the **honest peak-to-
peak way** — clean's peak Clmax (1.443 at its ~15° stall) vs each VG's peak
Clmax at *its* higher stall angle, since `Vstall ∝ 1/√Clmax`. **Both stall and
cruise are shown at two span coverages** — 40% (outboard, over the ailerons) and
100% (root-to-tip) — because *both* scale with how much wing you cover. Ranked
by stall-per-cruise ratio at 40% span:

| # | VG config | stalls at | stall ↓ 40% | stall ↓ 100% | cruise ↓ 40% | cruise ↓ 100% | ratio |
|---|---|---|---|---|---|---|---|
| **1** | **6 mm delta, 70 mm** ★ | **~18°** | **−2.4 kt** | **−5.6 kt** | **−1.5 kt** | −3.6 kt | **1.6×** |
| 2 | 12 mm delta, 70 mm | ~16° | −0.8 kt | −1.8 kt | −3.5 kt | −8.2 kt | 0.2× |
| 3 | 8 mm parabolic, 50 mm | ~18° | −0.7 kt | −1.8 kt | −3.1 kt | −7.3 kt | 0.2× |
| 4 | 8 mm delta, 50 mm | ~17° | −0.5 kt | −1.1 kt | −2.7 kt | −6.5 kt | 0.2× |

*Clean wing peaks at **Clmax 1.443 @ α ≈ 15°** then its lift collapses — it
stalls ~15°. Every VG holds lift higher and later; the 6 mm delta reaches a
genuine steady peak of **Clmax 1.709 at α 18°**, far above the field, which is
why it wins stall outright. The bigger VGs' steady peaks land lower (~1.49–1.52)
because their high-lift points fall where buffet has already risen. Why not the
~15 kt some pilots quote? This LS(1)-0413 already has a high clean Clmax (~1.44),
so there's less to recover — draggy STOL airfoils starting at ~1.2–1.4 gain more.
**5.6 kt is the truthful number for this wing.***

**How to read it — span coverage is a lever, not just a detail:**
- **Want the safety (lower stall speed)?** Go *wide* — a 100% install gives
  ~2.3× the stall reduction of 40% (6 mm delta: −2.4 → −5.6 kt), because the
  inboard wing gets the benefit too.
- **Want to protect cruise?** Go *narrow* — 40% costs about 40% of the cruise.
- **#1 (6 mm delta @ 70 mm) is the clear winner** — the *only* config that nets
  positive (stall gain > cruise cost) at 40% span, and it also has the lowest
  cruise tax of the whole study. Stall data, cruise data, and steadiness all
  agree on the same single part.

### 🔀 Mixed install — root-first stall + max aileron authority

Your idea: seed a **root-first stall** for warning while keeping the ailerons
attached longest. The peak-to-peak polar changed the answer here in a good way:
**the 6 mm delta is both the cruise *and* the stall champion** (its Clmax 1.709
beats the high-lift shapes' steady peaks), so the cleanest install is **one part
everywhere**, varied only by *spacing*:

| zone | VG | role |
|---|---|---|
| **bare root** (innermost station) | *no VG* | stalls **first** → buffet warning (may replace the stall strip) |
| inboard (root → mid) | 6 mm delta, pitch ramps **110 → 70 mm** | gradient → smooth root→mid stall sweep |
| outboard (mid → tip, ailerons) | 6 mm delta, **70 mm** uniform | **max attachment + aileron authority** deepest into the stall |

**Why one part, not a two-shape mix:** since the 6 mm delta already wins lift
*and* cruise, adding a draggier high-lift vane outboard would only **cost cruise
without buying stall** — the bigger shapes' steady Clmax is actually *lower*.
So the spacing gradient (wide→tight) does all the work: weaker (wider) VGs
inboard let the root give up first, uniform tight VGs outboard hold the ailerons.

*(The spanwise gradient is built by interpolating between the discrete pitch
points the 2D slice measured — wider pitch demonstrably stalls earlier. The
slice cannot directly simulate a continuous spanwise gradient; a true swept-wing
3D run would refine the exact schedule, but the discrete pitch-vs-stall curve
strongly supports it. Stated plainly so nobody mistakes it for a 3D result.)*

### The recommended part 🏆

**A short delta (triangular-ramp) vane, 6 mm tall, 10° incidence, 70 mm
spacing, counter-rotating pairs, front tips at 7% chord** — used as **one part
on all surfaces** (wing, plus elevator/rudder for control authority when slow),
with a **bare patch / wider spacing at the wing root** so the root stalls first
(natural buffet warning, may replace the stall strip). It is the outright winner
on **all four axes**: it pushes the stall from ~15° to ~18° for the biggest
honest stall-speed cut (**−5.6 kt full-span / −2.4 kt at 40%**), costs the
**lowest cruise drag of the entire study** (+6.2%, ~−1.5 kt at 40% span),
reattaches the stalled wing (drag −80%, L/D ~×5), and runs the *steadiest*
through the buffet. No step-up needed — the bigger shapes cost more cruise for
*less* peak lift. One small, cheap, printable part does everything.

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
