# RapidCFD GPU study — gapless wing + VG matrix (2026-06-12)

Steady RANS (kOmegaSST, fully turbulent), one-VG-pitch periodic slice of the
no-aileron-gap LS(1)-0413 wing at the aileron station. Re 2.2e6
(U = 36.09 m/s, c = 0.9022 m), alpha baked into the geometry. Solved with
RapidCFD (OpenFOAM 2.3 GPU fork) on an RTX 5090, SP build with
double-accumulated reductions; meshed with OpenFOAM v2506 snappyHexMesh
(~2.4–2.7 M cells/case). Two-stage scheme ramp (upwind → limitedLinearV),
coefficients averaged over the last 500 iterations (mean ± peak-to-peak).

All VG rows: counter-rotating toe-out pairs, ±15° incidence, l = 3h,
50 mm pitch. `p50` = study-default station x/c = 0.07; `xNN` = station
x/c = 0.NN. Aref = chord × pitch; CofR = c/4.

## Force coefficients

| case | α | VG h | x/c | Cl | pk-pk | Cd | pk-pk | L/D |
|---|---|---|---|---|---|---|---|---|
| clean_a08 | 8° | — | — | 1.260 | 0.034 | 0.0236 | 0.0004 | 53.4 |
| clean_a18 | 18° | — | — | 1.605 | 0.276 | 0.4434 | 0.101 | 3.6 |
| **vg12p50_a18** | 18° | 12 mm | 0.07 | **1.533** | 0.183 | **0.1408** | 0.072 | **10.9** |
| vg16p50_a18 | 18° | 16 mm | 0.07 | 1.104 | 0.356 | 0.3726 | 0.150 | 3.0 |
| vg12x15_a18 | 18° | 12 mm | 0.15 | 1.812 | 0.440 | 0.3965 | 0.165 | 4.6 |
| vg12x30_a18 | 18° | 12 mm | 0.30 | 1.801 | 0.502 | 0.4635 | 0.254 | 3.9 |
| vg12x45_a18 | 18° | 12 mm | 0.45 | 1.291 | 0.507 | 0.4349 | 0.238 | 3.0 |

Deltas vs clean_a18: vg12p50 ΔCl −4.5% / ΔCd −68.2%; vg16p50 −31.2% / −16.0%;
vg12x15 +12.9% / −10.6%; vg12x30 +12.2% / +4.5%; vg12x45 −19.5% / −1.9%.

## Reading

- **clean_a08** (sanity point): converged tight; Cl 1.26 matches the v2506
  CPU cross-check (1.26) and sits where fully-turbulent kOmegaSST should
  against the XFOIL/NASA anchors. Pipeline validated.
- **clean_a18**: deep stall — drag is wing-sized (Cd 0.44), lift rides a slow
  separation limit cycle. This is the "before" picture.
- **12 mm @ 7%c is the only configuration that un-stalls the wing.** Drag
  collapses 68% while lift holds within 5% of the (stall-inflated) clean
  value; the residual pk-pk says a small separated pocket still breathes,
  but the flow is substantially reattached. L/D at 18° triples.
- **16 mm @ 7%c hurts.** At 7% chord the boundary layer is millimeters
  thick; a 16 mm plate is ~4–5× the local layer height and behaves like a
  spoiler row — lift −31%, big unsteady wake. Taller is not better when the
  row is far forward.
- **Aft placement (15–45%c) does not fix the stall.** 15% and 30% produce
  more lift than clean (+13%, +12%) but keep stall-class drag and the
  largest oscillations in the study — the vanes energize the front of the
  airfoil yet separation still wins behind them; by 30–45% the row stands
  in (or at the edge of) already-separated flow. 45% loses lift outright.
  Mid-chord rows seen on some fielded Glasairs are plausibly aimed at
  cruise/handling effects, not at maximum-alpha stall recovery — at 18°
  they do nothing good here.

## Caveats

- Steady RANS on a massively separated flow: the stalled cases (clean_a18
  and every aft-row case) are limit-cycle averages, not converged steady
  states — trust the *ranking* and the attached-flow numbers more than the
  third digit of any stalled Cd.
- Fully-turbulent (no transition model); periodic slice = infinite VG array
  (no tip/fuselage effects); smooth skin; one pitch (50 mm) and one pair
  geometry per case.
- 6–8 slightly over-threshold skew faces at the vane–skin junctions in the
  x15/x30 meshes (max ~4.6 vs 4.0 limit) — localized, did not destabilize
  the solves.

Regenerate: `python gpu/rapidcfd/build_cases.py` then
`bash gpu/rapidcfd/run_all.sh <cases>` in WSL;
report: `python gpu/rapidcfd/report.py --tail 20`.
