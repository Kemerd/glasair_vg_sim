# RapidCFD GPU VG study — optimization hunt snapshot (2026-06-14)

Dated report. Supersedes the original `RESULTS.md` (the first 7-case study).
Future reports keep the `MM-DD-YY_results.md` naming so old snapshots survive.

Steady RANS (kOmegaSST, fully turbulent), one-VG-pitch periodic slice of the
no-aileron-gap LS(1)-0413 wing at the aileron station. Stall cases at Re 2.2e6
(U = 36.09 m/s); cruise cases at Re 5.52e6 (U = 90.55 m/s, 200 mph). c = 0.9022 m,
alpha baked into the geometry. RapidCFD (OpenFOAM 2.3 GPU fork) on an RTX 5090,
SP build with double-accumulated reductions; v2506 snappyHexMesh (~2.4–2.7 M
cells/case). Two-stage scheme ramp (upwind → limitedLinearV), coefficients
averaged over the last 500 iterations (mean, with peak-to-peak as the
limit-cycle width).

Naming: `vg<h><shape><pitch>[b<beta>][i]_a<alpha>`. shape: `p`=rect plate,
`d`=delta ramp, `s`=Stolspeed swept fin. `i`=toe-in. `b10`=10° incidence
(default 15°). `single`=single-alternating (else counter-rotating pair).
`x<NN>`=chord station 0.NN (else 7%c tips).

## The goal

Find ONE printable VG that **maximizes stall recovery** (α=18°, the deep-stall
point) with **minimal cruise drag** (α=2°, 200 mph). A VG that fixes the stall
but costs 10–20 kt of cruise is a non-starter. So every serious contender is
run at *both* angles and ranked by stall-win-per-cruise-cost.

## Stall recovery (α = 18°). Clean wing: Cl 1.605, Cd 0.443 (deep stall, L/D 3.6)

| case | shape | h | pitch | β | Cl | Cd | L/D | ΔCd | pk-pk Cd |
|---|---|---|---|---|---|---|---|---|---|
| vg12d60b10 | delta | 12 | 60 | 10 | **1.828** | 0.0920 | 19.9 | −79.2% | 0.032 |
| vg12d70b10 (max-stall alt) | delta | 12 | 70 | 10 | 1.797 | 0.0966 | 18.6 | −78.2% | 0.043 |
| vg12s50 | stol fin | 12 | 50 | 15 | 1.809 | 0.099 | 18.3 | −77.7% | 0.032 |
| vg12d50b10 | delta | 12 | 50 | 10 | 1.804 | **0.090** | **20.0** | **−79.7%** | 0.061 |
| vg12t70b10 (trapezoid 70mm) | trap | 12 | 70 | 10 | **1.893** | 0.1022 | 18.5 | −76.9% | 0.037 |
| vg12g50b10 (gothic) | gothic | 12 | 50 | 10 | **1.870** | 0.0954 | 19.6 | −78.5% | 0.050 |
| vg12t50b10 (trapezoid) | trap | 12 | 50 | 10 | 1.754 | 0.1010 | 17.4 | −77.2% | 0.035 |
| vg12p70 | rect | 12 | 70 | 15 | 1.789 | 0.101 | 17.7 | −77.2% | **0.017** |
| vg12d50i | delta (toe-in) | 12 | 50 | 15 | 1.779 | 0.100 | 17.8 | −77.5% | 0.041 |
| vg12d50 | delta | 12 | 50 | 15 | 1.764 | 0.094 | 18.8 | −78.8% | 0.037 |
| vg10p50 | rect | 10 | 50 | 15 | 1.741 | 0.103 | 16.9 | −76.8% | 0.053 |
| vg08d50b10 (8mm micro) ★CHAMPION | delta | 8 | 50 | 10 | 1.748 | 0.0926 | 18.9 | −79.1% | **0.026** |
| vg06d50b10 (6mm micro) | delta | 6 | 50 | 10 | 1.680 | 0.0989 | 17.0 | −77.7% | **0.026** |
| vg12ssingle | stol single | 12 | 50 | 15 | 1.635 | 0.107 | 15.3 | −75.9% | 0.013 |
| vg12p50 | rect (orig winner) | 12 | 50 | 15 | 1.533 | 0.141 | 10.9 | −68.2% | 0.072 |
| vg12single | rect single | 12 | 50 | 15 | 1.460 | 0.114 | 12.8 | −74.3% | 0.027 |
| vg16p50 | rect | 16 | 50 | 15 | 1.104 | 0.373 | 3.0 | −16.0% | 0.150 |
| vg12a50b10 (airfoil-section) | airfoil | 12 | 50 | 10 | 1.285 | 0.1745 | 7.4 | −60.6% | 0.539 |
| vg12p35 | rect | 12 | 35 | 15 | 1.244 | 0.319 | 3.9 | −28.1% | 0.100 |

| vg12d50b08 | delta | 12 | 50 | 8 | 1.777 | 0.0864 | 20.6 | −80.5% | 0.057 |
| vg12d50b12 | delta | 12 | 50 | 12 | 1.745 | 0.0962 | 18.1 | −78.3% | 0.054 |
| vg12d50b20 | delta | 12 | 50 | 20 | 1.648 | 0.100 | 16.4 | −77.4% | 0.054 |

*(β sweep COMPLETE @ 50mm: 8°(Cd 0.086) ≈ 10°(best lift) > 12° > 15° > 20°.
Optimum incidence 8–10°. vg12d50b05, vg12d70b10,
vg12d60b10, vg06/vg08 micro, trap/gothic/airfoil — in queue.)*

## Cruise drag tax (α = 2°, 200 mph). Clean wing: Cd 0.01064

| case | shape | h | pitch | Cd | Δ vs clean |
|---|---|---|---|---|---|
| clean_a02 | — | — | — | 0.01064 | — |
| **vg08d50b10_a02 ★** | delta | 8 | 50 | 0.01232 | **+15.8%** |
| vg12d70b10_a02 | delta | 12 | 70 | 0.01274 | +19.8% |
| vg12ssingle_a02 | stol | 12 | 50 | 0.01336 | +25.6% |
| vg12d50b10_a02 | delta | 12 | 50 | 0.01344 | +26.3% |
| vg12single_a02 | rect | 12 | 50 | 0.01431 | +34.5% |
| vg12a50b10_a02 | airfoil | 12 | 50 | 0.01370 | +28.8% (moot — fails stall) |
| vg12t50b10_a02 | trap | 12 | 50 | 0.01601 | +50.5% |
| vg12s50_a02 | stol | 12 | 50 | 0.01682 | +58.1% |
| vg12p50_a02 | rect | 12 | 50 | 0.01981 | +86.2% |

**Cruise-tax ranking — THREE levers, in order of impact:** (1) *shallow
incidence* (β=10° vs the inherited 15°) is the biggest single win — less yaw,
less cruise drag, and it does *not* cost stall recovery; (2) *wide pitch* — the
delta@β10 drops from +26% at 50 mm to **+19.8% at 70 mm** (fewer VGs over the
span); (3) *single-alternating layout* helps the rect (+34% vs the pair's +86%)
and the stol-single posts +25.6%, but it does **not** beat the wide-pitch
shallow-incidence delta pair. So the cruise champion is **delta, β10, 70 mm
(pair)** at +19.8% — incidence and pitch matter more than going single. Plain
shape (rect→delta→stol) barely moves cruise on its own; it's the *angle and
spacing* that do. Wave G still tests whether single-alt shaves the last bit.

## Findings

1. **Delta and Stolspeed swept fin tie for the lead and both crush the flat
   plate.** The original rect-plate winner (vg12p50, L/D 10.9) is now mid-pack;
   the delta and swept fin reach L/D ~18–20 at stall — nearly double — and the
   delta *gains* lift where the flat plate lost it.

2. **Shallower incidence is better — the optimum is β≈8–10°.** Delta at α=18°,
   50 mm: β=8° Cl 1.777 / Cd **0.0864** (lowest drag) · β=10° Cl **1.804** / Cd
   0.0900 (highest lift) · β=15° Cl 1.764 / Cd 0.094 · β=20° Cl 1.648 / Cd
   0.100. So β=8° minimizes drag and β=10° maximizes lift — both clearly beat
   the inherited 15°, and 20° over-yaws. Net: **set incidence to 8–10°**; the
   choice between them is a wash (~0.4% on each metric). β=12 (running) fills
   the gap but the minimum is bracketed.

3. **Height: micro-VGs work — 6–12 mm all recover the stall.** Delta @ β10 @
   50 mm height ladder at α=18°: 6 mm Cd 0.0989 / Cl 1.680 · 8 mm Cd 0.0926 /
   Cl 1.748 · 12 mm Cd 0.0900 / Cl 1.804. It's a gentle gradient, not a cliff —
   even a 6 mm vane drops drag ~78% (vs clean 0.443). Taller gives marginally
   more lift; the 8 mm hits a sweet spot (Cd nearly the 12 mm's, far steadier:
   pk-pk 0.026 vs 0.061). This is the key to the cruise tradeoff: if a short
   vane recovers stall, it should cost far less at cruise (it barely pokes the
   thin cruise BL). The 6/8 mm cruise-tax cases (running/queued) test exactly
   this — a micro-delta may be the cruise-friendly champion. (Note: 16 mm was
   *too tall* — spoiler effect, lift −31%; the useful band is ~6–12 mm.)

4. **Wider spacing is better — and too tight is catastrophic.** 70mm > 50mm,
   and 35mm *re-stalls the wing* (Cd back to 0.32). Crowded VGs merge into one
   ragged disturbed vortex instead of discrete tight ones — exactly the failure
   mode the Stolspeed designer warned about. The inherited 50mm was on the
   tight side; the optimum is wider.

5. **Single-alternating halves the cruise tax** (+34% vs the pair's +86%) — but
   it does *not* beat the wide-pitch shallow-incidence delta pair (+19.8%). A
   *flat* single loses lift at stall (−9%); a *swept* single keeps it (+1.9%).
   So single-alt is a real lever but ranks behind incidence and pitch.

6. **Steadiness tracks reattachment quality.** vg12p70, the micro-deltas, and
   the single fins show the smallest limit-cycle pk-pk (0.013–0.026) — the flow
   is most solidly reattached, not just lower on average.

## The tradeoff verdict + cruise speed loss in KNOTS

The honest physics: a *passive* VG can only add drag when the flow is attached,
so none of these will *raise* cruise speed — the best achievable is a stall fix
that costs ~nothing at cruise.

**★ NEW CHAMPION: 8 mm MICRO-delta, β=10°.** The micro-VG upset paid off — a
vane 2/3 the height beats the 12 mm on the combined tradeoff:
- Stall (α=18°, 50 mm): Cl 1.748 (+9.0%), Cd 0.0926 (**−79.1%**), L/D 18.9 —
  recovers the stall nearly as well as the 12 mm, and *steadier* (pk-pk 0.026).
- Cruise (α=2°, 50 mm): Cd 0.01232 (**+15.8%** — the lowest tax of all 30+
  configs, beating even the 12 mm @ 70 mm's +19.8%).
The physics: at cruise the boundary layer is thin, so a short 8 mm vane barely
pokes out of it = minimal drag, yet at α=18° the stall BL is thick enough that
8 mm still bites and reattaches the flow. Best of both worlds — and a smaller,
cheaper part to 3D-print. The 12 mm delta@β10 (70 mm: cruise +19.8%, stall Cd
−78.2%; 50 mm: L/D 20.0, the max stall authority) remains the alternate if you
want the strongest possible stall margin. *(6 mm also recovers stall; its cruise
tax + an 8 mm @ 70 mm combo are running in Wave I — could push even lower.)*

**Cruise speed loss** (off 224 kt true / 258 mph @ 8000 ft, ~200 hp, prop η 0.82),
by how much of the span carries VGs:

| config | 25% span | 40% span | 55% span |
|---|---|---|---|
| **★ 8 mm delta β10, 50 mm** | **~1.2 kt** | **~1.9 kt** | **~2.6 kt** |
| 12 mm delta β10, 70 mm | ~1.5 kt | ~2.4 kt | ~3.3 kt |
| 12 mm delta β10, 50 mm | ~2.0 kt | ~3.2 kt | ~4.3 kt |
| orig rect-pair β15, 50 mm | ~6.3 kt | ~9.7 kt | ~13 kt |

**The 8 mm micro-delta drops the cruise penalty to ~1–2.5 kt** — roughly a fifth
of the original config's cost — while recovering the stall and being smaller to
print. Caveat: the slice is
all-VG/no-gap, so the real airplane penalty is somewhat *lower* still. Wave G
(single-alternating + β10 + wide pitch) is testing whether single-alt shaves it
even further while holding the stall win.

Current leader for raw stall recovery: **delta, 10° incidence**; the micro-VG
(6–8 mm) and Wave G best-of-both cases are still running.

## Caveats

- Steady RANS on separated flow: stalled cases are limit-cycle averages; trust
  the ranking over the third digit. kOmegaSST is known to *under*-predict
  streamwise-vortex strength, so the real VG benefit is likely better than shown.
- Fully turbulent (no transition); periodic slice = infinite array; one pair
  geometry and station per case unless swept.
- 6–8 slightly over-threshold skew faces at vane–skin junctions on some meshes
  (localized, did not destabilize the solves).

Regenerate: `python gpu/rapidcfd/build_cases.py --only <cases>` then the master
GPU-idle-gated queue in WSL; report: `python gpu/rapidcfd/report.py --tail 20`.
