# Glasair III Vortex Generator Placement Study — CFD Project Specification

**Purpose:** Build an automated OpenFOAM-based CFD pipeline to determine optimal vortex generator (VG) placement, height, and spacing on a Glasair III experimental aircraft, for three surfaces: wing (ahead of ailerons), horizontal stabilizer underside (ahead of elevator), and vertical fin (both sides, ahead of rudder). The output of this project narrows the flight-test matrix to 1–2 candidate VG layouts per surface. Flight testing (tuft visualization) remains the final verification step; this pipeline is for design-space exploration, not certification-grade absolute numbers.

**Trust deltas, not absolutes.** RANS near CLmax carries 5–10% uncertainty on absolute CLmax/stall-AoA. All conclusions should be framed as comparisons between configurations (VG-on vs VG-off, position A vs position B) run on identical meshes and settings.

---

## 1. Background / Aerodynamic Context

- Aircraft: Stoddard-Hamilton Glasair III. Low-wing, all-composite, 2-seat, high wing loading, retractable gear.
- Wing airfoil: **NASA/Langley LS(1)-0413 (GA(W)-2)**, 13% thick. Published coordinates available (UIUC database / airfoiltools.com `ls413-il`; Glasair Owners Association also hosts a CAD profile). Published NASA wind-tunnel data exists for this section — use it as the clean-section validation anchor.
- Planform reference (from factory 3-view, Stoddard-Hamilton drawing rev G, 4/26/90): span 23' 3-5/16", length 21' 2", 3° dihedral. User has DXF files of the 3-view for exact chord/taper/control-surface geometry — these will be provided in `geometry/dxf/`.
- Known handling problems being addressed:
  1. Aileron effectiveness washes out at low speed / high AoA (outboard separation).
  2. Limited nose-up elevator authority in the flare at forward CG with full flaps (separation on the **underside** of the stab/elevator — the tail lifts downward, so its suction side is the bottom).
  3. Limited rudder authority at large deflection and low airspeed (suction-side separation past the hinge gap).
- Prior art being replicated: Strausak Lancair Legacy VG program (IE Impulse #74, Jan 2021). His tested geometry, to be used as the center of our sweep ranges: wing VGs at 7% chord, 50 mm spacing outboard / 90 mm inboard (coarser inboard so the root stalls first), 15° vane incidence to local flow, counter-rotating; elevator VGs 100 mm ahead of hinge line, 30 mm spacing, on the underside.
- Theory anchor: Lin, *Progress in Aerospace Sciences* 38 (2002) 389–420 — micro-VGs with height h ≈ 0.1–0.5 δ placed 5–10 h upstream of separation onset; conventional VGs h ≈ δ.

## 2. Hardware / Software Constraints

- Workstation: Intel i9-14900 (24C/32T), RTX 5090 (GPU used for post-processing only; solver is CPU), assume 64–96 GB RAM. **Cap any single mesh at ~30 M cells.** Target overnight (≤ 12 h wall clock) per case on 24 cores.
- Solver: **OpenFOAM** (latest ESI release, e.g. v2406+, or foundation release — pick one and pin it; prefer ESI for the `kOmegaSSTLM` transition model). Document the exact version in the README.
- Everything runs on Linux (native or WSL2). Provide install/bootstrap instructions in the README assuming Ubuntu.
- Meshing: `blockMesh` + `snappyHexMesh` (or `cfMesh` if it simplifies the extruded-section topology — developer's choice, but justify).
- Pre/post: Python 3.11+, `numpy`, `pandas`, `matplotlib`, `PyVista` (GPU-accelerated viz welcome), `ezdxf` for DXF parsing.
- All automation in Python; no manual GUI steps anywhere in the loop.

## 3. Project Phases

Build in this order. Each phase has a validation gate; do not proceed past a failed gate.

### Phase 0 — Repo scaffold & geometry toolkit
1. Repo layout:
   ```
   /geometry          # airfoil coords, DXF parsing, loft scripts, STL generation
   /cases             # OpenFOAM case templates (one per study type)
   /sweeps            # sweep definitions (YAML) + runner
   /scripts           # mesh/run/post automation
   /validation        # XFOIL + NASA experimental comparison data & scripts
   /results           # parsed outputs, plots, summary CSVs (gitignored except summaries)
   README.md
   ```
2. Geometry toolkit:
   - Fetch/store LS(1)-0413 coordinates (commit the coordinate file; cite source in a header comment). Provide a spline-refinement function (cosine-clustered resampling, ≥ 200 points, closed sharp or blunt TE — make TE treatment a flag).
   - DXF reader: extract wing chord distribution, aileron span/chord, stab and fin outlines, hinge-line positions from the user's 3-view DXFs. Where the DXF is ambiguous, fall back to parameters in a single `aircraft.yaml` (create it, populate with known values, mark unknowns `TODO` for the user).
   - STL generators: (a) 2.5D extruded wing section; (b) extruded stab section with deflected elevator + **hinge gap** (gap width parameter, default 1/16" = 1.59 mm, per factory drawing); (c) extruded fin section with deflected rudder + same gap. Control-surface deflection implemented as rigid rotation about the hinge line with gap geometry preserved (no sealed gaps).

### Phase 1 — Clean-section validation (2D)
1. XFOIL baseline: script an XFOIL polar sweep of LS(1)-0413 at Re = 1.5 M, 3 M, 6 M (covers approach through cruise; compute Re from chord & speed in `aircraft.yaml` and document). Free transition (e^N, Ncrit = 9) and tripped cases.
2. OpenFOAM 2D case: C-grid or snappy domain, far field ≥ 25 chords. `kOmegaSSTLM` (γ–Re_θ transition) — this is mandatory, not plain SST, because the LS(1)-0413 on a composite skin carries significant laminar flow and δ at the VG station depends on transition location. y+ < 1 everywhere (first-cell height calculator script required), growth ratio ≤ 1.2, ≥ 30 prism layers.
3. AoA sweep −4° to +20° in 2° steps (1° near stall). `simpleFoam`, with a documented fallback to `pimpleFoam` pseudo-transient for post-stall non-convergence.
4. **Gate:** Cl-α slope within 5% of XFOIL/NASA data in the linear range; Clmax within 10% and stall AoA within 2° of NASA experimental data; transition location on the upper surface within 10% chord of XFOIL's e^N prediction at matching Ncrit. Log all comparisons to `validation/report.md` automatically.
5. From the converged clean solutions, extract and tabulate **boundary-layer thickness δ and δ\* vs chord position** on the suction surface at each AoA (write a BL-extraction utility — wall-normal profile sampling). This table sizes the VGs.

### Phase 2 — jBAY VG model implementation
1. Implement the **jBAY** (Jirásek's variant of the Bendiksen–Apsley–Yan BAY) vane source-term model as an OpenFOAM `fvOptions` momentum source, or integrate an existing open implementation if one is found and license-compatible (search first; cite what you use). Inputs per vane: location, chordwise position, height h, vane length l (default l = 3h), incidence angle β (default 15°), orientation (counter-rotating pair handedness), calibration constant.
2. Unit validation case: single vane pair on a flat plate / simple wing section; confirm the model produces a streamwise vortex pair of plausible circulation and that downstream vortex decay is qualitatively sane (compare against published jBAY validation figures — Jirásek AIAA J. Aircraft 2005 — digitize one comparison curve into `validation/`).
3. **Gate:** unit case reproduces the published jBAY behavior trend (vortex strength vs vane angle) and is mesh-converged (3-level refinement study; report GCI on peak streamwise vorticity).

### Phase 3 — Study 1: Wing section VG sweep (the main event)
1. Geometry: 2.5D extruded LS(1)-0413 at the **aileron mid-span station** chord (from DXF), spanwise periodic domain exactly N VG-pair spacings wide (N ≥ 2). Include the aileron as a deflected flap with hinge gap (deflections: 0°, ±15°, and max from `aircraft.yaml`).
2. Mesh: ≤ 10 M cells per case. Local refinement box around VG line and over the aileron. Same wall resolution rules as Phase 1.
3. Sweep matrix (define in `sweeps/wing.yaml`):
   - Chordwise position: 5%, 7%, 10% chord
   - Height: h/δ ∈ {0.2, 0.5, 1.0} where δ is taken from the Phase-1 BL table at the approach-condition AoA
   - Spacing: 50 mm, 70 mm, 90 mm (pair-to-pair)
   - Vane angle: 15° fixed (one sensitivity case at 12° and 18° for the best position)
   - Flow conditions: Re and AoA grid covering approach → stall (use Phase-1 stall AoA to set the upper bound; include at least 3 AoA past clean-section stall)
   - Baseline (no VG) at every flow condition — always rerun baseline on the *same mesh family* as VG cases.
4. Full factorial is large — implement the sweep runner to support staged execution: coarse screening (position × height at one spacing, one aileron deflection), then refinement around the winner. Queue management: run cases sequentially or 2-wide depending on core/RAM budget per case; make this configurable.
5. Outputs per case (automated extraction): Cl, Cd, Cm vs AoA; separation point location (wall-shear-stress sign change); aileron hinge moment; surface Cp and Cf distributions; slice images (PyVista) of streamwise vorticity at 3 downstream stations. Summary CSV + auto-generated comparison plots: ΔClmax, Δ(stall AoA), Δ(separation onset AoA over aileron) vs baseline.
6. **Primary figure of merit:** AoA margin by which attached flow over the deflected aileron is extended versus baseline, at approach Re. Secondary: ΔClmax, ΔCd at cruise AoA (drag penalty check — report it, target < 2 counts increment at cruise CL).

### Phase 4 — Study 2: Stab/elevator underside (run after Study 1 pipeline is proven)
1. Same extruded-section machinery, stab airfoil (symmetric section from plans/`aircraft.yaml`; if unknown, NACA 0010 placeholder flagged `TODO`), elevator deflected trailing-edge-UP (nose-up command) at 50%, 75%, 100% of max deflection, **with hinge gap**.
2. Inflow: sweep local incidence angle from 0° to −16° (flow from below = tail downforce condition) representing the wing-downwash envelope. Add a helper script that estimates the downwash envelope from classical lifting-line (ε ≈ 2·CL/(π·AR) + flap increment) using `aircraft.yaml` numbers, and documents the assumed range.
3. VGs on the **lower (suction) surface**, sweep centered on Strausak's geometry: 100 mm ahead of hinge, 30 mm spacing, h from local δ extracted from the clean stab solution.
4. Figure of merit: maximum local incidence (and deflection) at which elevator-side flow remains attached, VG vs baseline → reported as "recovered elevator authority envelope."

### Phase 5 — Study 3: Fin/rudder
1. Same as Phase 4 structurally: symmetric fin section, rudder deflections to max, hinge gap included, VGs mirrored on **both sides**, sweep inflow sideslip 0–15° combined with deflection.
2. Figure of merit: max deflection × sideslip combination with attached suction-side flow, VG vs baseline; rudder hinge moment trend (informative for pedal-force expectations).

### Phase 6 — Reporting
- Auto-generated `results/REPORT.md` per study: configuration table, validation status, all comparison plots, and a one-page recommendation block ("install row at X% chord, h = Y mm, spacing = Z mm, expected benefit: …, drag penalty: …, confidence/caveats: …").
- Explicit caveats section auto-included: RANS-near-stall uncertainty, transition-model sensitivity (run one transition-model-off sensitivity case per study and report the spread), absence of prop slipstream and fuselage effects, recommendation that flight testing with tufts and GPS speed-truthing (3- or 4-leg groundspeed method) is the verification step.

## 4. Engineering Rules (non-negotiable)

- y+ < 1 on all viscous walls; print a y+ histogram per case and fail the case if max y+ > 2 over more than 1% of wall faces.
- Mesh-independence: for each study, run the chosen baseline at 3 refinement levels and report GCI on Cl and separation location before sweeping.
- Convergence criteria: residuals < 1e-5 AND force coefficients flat (< 0.5% oscillation over last 500 iterations); cases failing this are auto-flagged, not silently included.
- Every case directory must be reproducible from `aircraft.yaml` + sweep YAML + pinned OpenFOAM version. No hand-edited dictionaries in results.
- Units: SI internally everywhere; accept inches/feet in `aircraft.yaml` inputs with explicit unit tags.
- Sea-level ISA default; density/viscosity overridable in `aircraft.yaml`.

## 5. Milestone / Acceptance Order for Claude Code

1. M0: repo scaffold, `aircraft.yaml`, geometry toolkit with unit tests (airfoil resample, DXF parse stub, STL watertightness check via `trimesh`).
2. M1: Phase-1 clean 2D case runs end-to-end from one command (`./scripts/run_validation.sh`), gate report generated.
3. M2: jBAY fvOptions source compiling and passing the unit validation case.
4. M3: Study-1 sweep runner executes a 2-case smoke sweep (baseline + one VG config) end-to-end with auto post-processing.
5. M4: full Study-1 screening sweep definition ready to launch; Studies 2–3 templates cloned and parameterized.
6. M5: reporting pipeline.

Start with M0–M1. Ask the user for: their actual RAM size, OpenFOAM install preference (native Linux vs WSL2), the DXF files, and the stab/fin section designations if known. Do not block on the DXFs — `aircraft.yaml` placeholder values keep everything runnable.
