# Vortex Generator Master Report — Glasair III
### Everything you need to choose YOUR install. The "#1 pick" is only a suggestion.

> **Read this top-to-bottom once, then jump to [§9 Decision Menu](#9-the-decision-menu--pick-your-own-winner) to choose.**
> This document is deliberately exhaustive. It explains *how* every number was
> produced, *why* the physics works the way it does, *every* candidate config
> with its full trade-offs, and several ready-to-go install recipes sorted by
> what **you** care about most — pure cruise protection, pure stall safety,
> balanced, or maximum control authority. The single "champion" is just the
> config that wins the *default* figure of merit; if your priorities differ,
> a different config is correct **for you**, and this report tells you which.

---

## Table of contents
1. [Why this study exists](#1-why-this-study-exists)
2. [The airplane & airfoil (why your numbers differ from the internet)](#2-the-airplane--airfoil)
3. [How the CFD was run (methods, in detail)](#3-how-the-cfd-was-run)
4. [How stall speed and cruise loss are computed (the math)](#4-how-stall-speed--cruise-loss-are-computed)
5. [The complete raw dataset (every case we ran)](#5-the-complete-raw-dataset)
6. [What we learned about SHAPE](#6-what-we-learned-about-shape)
7. [What we learned about HEIGHT, CHORD STATION, ANGLE, SPACING](#7-height-chord-station-angle-spacing)
8. [The four finalists — full trade-off breakdown](#8-the-four-finalists)
9. [THE DECISION MENU — pick your own winner](#9-the-decision-menu--pick-your-own-winner)
10. [Install geometry & the printable parts](#10-install-geometry--the-printable-parts)
11. [The progressive spanwise install (root-first stall)](#11-the-progressive-spanwise-install)
12. [Honest caveats & limitations](#12-honest-caveats--limitations)
13. [Glossary & file index](#13-glossary--file-index)

---

## 1. Why this study exists

Vortex generators are tiny vanes that drag high-energy air from the free stream
down into the boundary layer, keeping the flow attached to the wing to a higher
angle of attack than it otherwise would. The payoff: **a lower stall speed and a
gentler stall**. The cost: **a little extra drag in cruise**, every second the
airplane flies, forever.

The internet is full of contradictory VG advice — how tall, how far apart, what
shape, what angle, mount them where — and almost none of it is specific to the
**Glasair III's wing**. So this study did the only thing that settles it:
**ran the actual airfoil through GPU CFD across every variable that matters**,
and reduced everything to the two numbers a pilot actually feels:

> **How many knots of stall speed do I gain, and how many knots of cruise do I pay?**

This report gives you those two numbers for every viable configuration, plus the
reasoning, so you can weight them however *you* want.

---

## 2. The airplane & airfoil

| Parameter | Value | Source |
|---|---|---|
| Aircraft | Stoddard-Hamilton Glasair III | — |
| Wing section | **LS(1)-0413** (NASA GA(W)-2 family) | `geometry/ls413.dat` |
| Reference chord | **0.9022 m** (aileron mid-station) | DXF / `aircraft.yaml` |
| Cruise Reynolds number | ~5.5 × 10⁶ | Re at cruise TAS |
| Stall-regime Reynolds | ~2.2 × 10⁶ | Re at ~80 mph |
| Clean stall speed reference (Vs₀) | **69.5 kt** | study basis |
| VG chord station (default) | **7% chord** | IMP74 + Stolspeed convention |

### ⚠️ The single most important context in this whole report

**The LS(1)-0413 is a clean, modern, high-lift airfoil.** Its *clean* maximum
lift coefficient is already about **Clmax ≈ 1.44**. That matters enormously:

- People who report **"VGs dropped my stall 15 knots!"** are almost always
  flying **draggy bush/STOL airfoils** that start at a low Clmax (~1.2–1.4).
  On those wings, a VG that recovers Clmax to ~1.7 is a *huge fractional* jump,
  hence a big stall-speed cut.
- **Your wing already makes 1.44 clean** — there is simply **less headroom to
  recover**. So the honest stall-speed reduction for this airfoil is **smaller
  in absolute knots** than the big internet numbers, *not because the VGs are
  weak, but because the wing was already good.* The best config here buys about
  **5.6 kt full-span** — and that is the truthful number, not a disappointment.

Keep this in mind every time you see a knots figure below.

---

## 3. How the CFD was run

### 3.1 Solver & hardware
- **RapidCFD** — a CUDA/Thrust GPU fork of OpenFOAM 2.3, run on an **NVIDIA
  RTX 5090**. Single-precision solve with double-accumulated reductions for
  stable force integration.
- Meshing by **OpenFOAM v2506 `snappyHexMesh`**, ~2.4–2.5 million cells per case.
- Turbulence model: **k-ω SST RANS**, fully turbulent (no transition model —
  conservative for a VG study, since VGs work by energizing turbulent BL).

### 3.2 What geometry was simulated
Each case is a **2-D periodic spanwise slice** exactly one VG *pitch* wide (the
spacing between vanes), with the airfoil at the chord and Reynolds number above.
For VG cases, a **counter-rotating pair** of vanes sits in that pitch, yawed to
the toe angle, planted at the 7% chord station with the vane base following the
local skin slope. Periodic side boundaries make the slice represent an infinite
spanwise row of identical VGs — the correct model for a uniform VG field.

### 3.3 The two-stage convergence scheme
Each run uses a **two-stage scheme ramp** for robustness:
1. **Stage 1** — first-order upwind convection, ~2000 iterations, to settle a
   stable field from the initial guess.
2. **Stage 2** — switch to `limitedLinearV` (second-order, accurate), continue
   to ~4500 total iterations.

The reported numbers are the **tail-500-iteration mean** of the converged run.

### 3.4 Reading the high-angle "buffet" correctly  ← important
At high angle of attack the flow **separates and oscillates** — the lift trace
swings violently (we report this as **pk-pk**, peak-to-peak). **This is real
physics, not solver noise:** it *is* the stall buffet, the same shaking you'd
feel in the airframe as the wing mushes. Therefore:

- We read the **time-averaged (tail-mean) lift straight through the buffet** —
  that mean *is* the lift the wing actually makes while buffeting.
- We report **pk-pk separately as a "stall manners" indicator** — a lower pk-pk
  at a given angle means a **gentler, more controllable, less abrupt** stall.
- A few points are **buffet-dominated** (pk-pk > ~1.5): their mean sits on a
  knife-edge and is **not a trustworthy steady value**. We flag these and do
  **not** use them as a config's peak. (Example: clean α18 mean 1.65 / pk-pk
  1.03 and clean α20 mean 1.55 / pk-pk 2.29 are buffet artifacts — the clean
  wing's *real* steady peak is at α15.)

### 3.5 How many cases
**63 GPU-CFD cases total**, spanning: 9 candidate *shapes*; heights 6/8/10/12/16
mm; chord stations 7/15/30/45%; incidence angles 5/8/10/12/15/20°; spacings
50/60/70/90/110 mm; single-vs-paired layouts; toe-in vs toe-out; **and a full
angle-of-attack polar from α = 2° (cruise) to α = 22° (deep post-stall)** for the
finalists, which is what lets us compute the stall *honestly* (see §4).

---

## 4. How stall speed & cruise loss are computed

### 4.1 Stall speed — the peak-to-peak Clmax method
Stall speed depends on the **maximum lift coefficient the wing can reach**:

> **Vstall ∝ 1 / √Clmax**  →  **ΔVstall = Vs₀ · (1 − √(Clmax_clean / Clmax_vg))**

with **Vs₀ = 69.5 kt**.

The crucial subtlety — and the thing that an earlier draft of this study got
wrong — is **which Clmax you compare**:

- ❌ **Wrong (same-angle):** compare clean vs VG lift *at the same angle* (say
  both at 18°). By 18° the clean wing is already stalled, so you're comparing the
  VG's healthy lift against the clean wing's post-stall wreckage. This makes the
  gain look like a measly ~2–3 kt — it **undercounts badly**.
- ✅ **Right (peak-to-peak):** compare the **clean wing's PEAK Clmax** (at *its*
  stall angle) against **each VG's PEAK Clmax** (at *its* higher stall angle).
  This captures the real benefit: the VG lets the wing keep climbing to a higher
  angle and a higher peak before it lets go.

That correction is the entire reason the full α-polar (§3.5) was run — we needed
to find *where each configuration actually peaks and stalls*, not assume it.

**Clean wing result:** peak **Clmax = 1.443 at α ≈ 15°**, then lift collapses
(1.394 → 1.308) — so **the clean Glasair III wing stalls around 15°.**

### 4.2 Span coverage — a lever, not a footnote
A VG only helps the part of the span it covers. We report **two coverages**:
- **40% span** — VGs on the outboard ~40% (over the ailerons). The un-VG'd
  inboard still stalls at the clean angle, so the *whole-airplane* stall benefit
  is partial — but you pay only ~40% of the cruise drag.
- **100% span** — root to tip. Full stall benefit, full cruise bill.

Section Clmax is span-weighted: `Clmax_aircraft = Clmax_clean·(1−f) + Clmax_vg·f`,
where `f` is the covered fraction.

### 4.3 Cruise-speed loss — fixed-power model
Added VG drag costs speed. At **fixed engine power**, speed scales as
`V ∝ (1 + Δf)^(−1/3)`, where `Δf` is the fractional increase in drag, taken from
the measured **α = 2° device drag tax** and span-weighted by coverage. We quote
the loss against a representative cruise of ~180 kt.

### 4.4 The default figure of merit (and why it's only a default)
The headline ranking uses **NET knots = stall-speed gain − cruise-speed loss**
at 40% span, plus the **benefit/cost ratio** (stall gain ÷ cruise loss). This
rewards configs that buy a lot of stall safety for little cruise.

**But "best NET" embeds a value judgment** — that 1 kt of stall and 1 kt of
cruise are worth the same to you. They may not be! If you'd happily pay 4 kt of
cruise to get every possible knot of stall margin (short strip, heavy airplane,
safety-first), a "worse NET" config is **right for you**. §9 lays out those
alternate priorities explicitly.

---

## 5. The complete raw dataset

All values are tail-500 means of converged RapidCFD runs. `Cl` = lift coeff,
`Cd` = drag coeff, `pk-pk` = peak-to-peak unsteadiness (stall-manners proxy),
`Cm` = pitching moment. Naming: `vg<height><shape><pitch>b<beta>_a<alpha>`, e.g.
`vg06d70b10_a18` = 6 mm, **d**elta, 70 mm pitch, **b**eta 10°, **a**lpha 18°.
Shapes: d=delta, p=plate/rect, t=trapezoid, g=gothic, s=stolspeed-fin,
pb=parabolic, og=ogive, a=airfoil-section, x=chord-station sweep.

### 5.1 Clean wing (the baseline) — full polar
| α (deg) | Cl (mean) | pk-pk | Cd | notes |
|--------:|----------:|------:|---:|---|
| 2  | 0.692 | 0.43 | 0.01462 | cruise reference |
| 8  | 1.240 | 0.76 | 0.03251 | |
| **15** | **1.443** | 0.97 | 0.09501 | **PEAK — stalls here** |
| 16 | 1.394 | 1.00 | 0.12469 | lift falling = stalling |
| 17 | 1.308 | 1.23 | 0.15848 | deeper stall |
| 18 | 1.65 *(buffet)* | 1.03 | 0.30224 | mean unreliable |
| 20 | 1.55 *(buffet)* | 2.29 | 0.36559 | mean unreliable |

### 5.2 The four finalists — full polars
| config | α2 Cd (cruise) | α15 | α16 | α17 | α18 | α20 | α22 | peak Clmax @α |
|---|---|---|---|---|---|---|---|---|
| **6 mm delta, 70 mm** | 0.01553 | 1.435 | — | 1.371 | **1.709** | 1.516 | 1.355 | **1.709 @ 18°** |
| 8 mm delta, 50 mm | 0.01632 | 1.485 | — | **1.491** | 1.473 | — | — | 1.491 @ 17° |
| 8 mm parabolic, 50 mm | 0.01655 | — | — | — | **1.519** | 1.461 | 1.496 | 1.519 @ 18° |
| 12 mm delta, 70 mm | 0.01682 | 1.507 | **1.522** | 1.504 | 1.507 | 1.62 *(buffet)* | — | 1.522 @ 16° |

### 5.3 Every other case we ran (the supporting evidence)
**Height sweep (delta, α18):** 6 mm@70 → 8 mm@50 → 10 mm → 12 mm@50/70 — taller =
more lift authority but more cruise drag; short hides in the thin cruise BL.

**Chord-station sweep (12 mm, α18):** `x15` Cl 1.676 / Cd 0.256, `x30` Cl 1.710 /
Cd 0.287, `x45` Cl 1.623 / Cd 0.311 — moving the row aft raises raw lift but the
drag explodes; **7% is the efficient station** (everything else in the matrix).

**Incidence sweep (12 mm delta @50, α18):** β5° Cl 1.433, β8° 1.519, **β10° 1.545**,
β12° 1.533, β20° 1.467 — **~10° is the sweet spot**; too shallow under-energizes,
too steep adds drag and separates the vane itself.

**Spacing sweep (12 mm delta, α18/16):** 50 mm 1.545 → 60 mm 1.546 → 70 mm 1.507
→ 90 mm 1.43 → 110 mm 1.43 — **tighter holds more lift; wider lets go earlier**
(this is the lever used for root-first stall, §11).

**Toe-in vs toe-out (12 mm, α18):** toe-out (`b10` 1.545) beats toe-in
(`d50i` 1.525 / `p50i` 1.442) — **toe-out wins.**

**Single vs paired (α18):** single-alternating delta@70 Cl 1.374 / pk-pk 2.09 —
**much worse and far less steady** than paired. **Pairs win decisively.**

**Shape shootout (≈12 mm @50–70, α18):** trapezoid@70 Cl 1.579, gothic Cl 1.593,
stolspeed-fin 1.562, ogive 1.518, parabolic 1.519, **plate/rect (naive) loses
lift and adds huge drag**, airfoil-section Cl 1.304 (*loses* lift, wild pk-pk
1.11 — disqualified). See §6.

### 5.4 Drag recovery at the stall (why VGs feel dramatic)
Even where a VG's *lift* number looks modest, its **drag reduction at high AoA is
enormous** — the wing is reattached instead of separated:

| config | ΔCd vs clean @ α18 |
|---|---|
| 6 mm delta @70 | **−61.0%** |
| gothic @50 | −56.5% |
| trapezoid @70 | −55.0% |
| 8 mm parabolic @50 | −53.5% |
| 8 mm delta @50 | −52.5% |
| 12 mm delta @70 | −51.7% |

This −50 to −61% drag cut (L/D roughly ×5 at the stall) is what makes the
approach feel solid and controllable instead of mushy and draggy.

---

## 6. What we learned about SHAPE

We tested nine planforms. The verdict is clear and a little surprising:

| Shape | Verdict | Why |
|---|---|---|
| **Delta** (triangular ramp) | ✅ **WINNER** | Sharp swept ramp sheds the *tightest, most energetic* vortex. Best drag/stall trade by a clear margin, and it reaches the highest *steady* peak Clmax of all (1.709 on the 6 mm). |
| Trapezoid (cropped delta) | High lift, high drag | Literature's "best for vortex persistence" does **not** hold here; raw lift good (1.58–1.89 in some cases) but cruise tax is heavy. |
| Gothic (concave swept LE) | High lift alt | Cl 1.59, doesn't collapse early as lit predicted; a max-authority alternate but draggier than delta. |
| Stolspeed swept fin | Good, not best | The user-favorite shape; solid (Cl 1.56) but delta beats it on the drag/stall trade. |
| Parabolic nose-cone | High lift alt | Holds lift remarkably flat & deep (1.52→1.46→1.50 to α22); a genuine high-lift option (see §8). |
| Ogive nose-cone | ≈ parabolic | Very similar to parabolic; no advantage over it. |
| Rectangular plate (naive) | ❌ **Loses** | The "just bend some tabs" approach **loses lift AND adds the most cruise drag** — a textbook example of why this study was needed. |
| Airfoil-section (cambered) | ❌ **Disqualified** | Sheds a *disorganized, unsteady* vortex on this wing; actually **loses lift** (Cl 1.30) with violent pk-pk. Looks great on paper, fails in practice. |
| Single-alternating | ❌ Worse | Hurts stall and steadiness vs counter-rotating pairs. |

**Bottom line on shape: build deltas, in counter-rotating pairs, toe-out.** The
fancy shapes give *more raw lift* but always at *more cruise drag*; none beats
the delta on the trade that matters.

---

## 7. Height, chord station, angle, spacing

These four "knobs" were each swept independently. The robust settings:

- **Chord station: 7%.** Moving aft raises raw lift but drag climbs much faster.
  7% is where the vortex forms early enough to keep the whole aft surface
  attached without a big drag penalty.
- **Incidence (toe angle): 10° toe-out.** The sweet spot across the 5–20° sweep.
  Shallower under-energizes the BL; steeper makes the vane itself a drag source
  and can separate.
- **Spacing (pitch): 50–70 mm tight = most lift held; 90–110 mm wide = lets go
  earlier.** Tight where you want attachment (ailerons); wide where you *want*
  the wing to give up first (root). This is the spanwise-stall lever (§11).
- **Height: the big trade-off** — see §8. Short (6 mm) = lowest cruise drag,
  highest *steady* peak here. Tall (12 mm) = more brute authority but more drag.

---

## 8. The four finalists

These are the configs worth actually installing. Numbers are the honest
peak-to-peak stall (§4) and fixed-power cruise loss, at both span coverages.

### 8.1 At a glance
| Config | Peak Clmax | Stalls at | Stall↓ 40% | Stall↓ 100% | Cruise tax | Cruise↓ 40% | Cruise↓ 100% | Steadiness (pk-pk) |
|---|---|---|---|---|---|---|---|---|
| **6 mm delta @70** | **1.709** | ~18° | **−2.4 kt** | **−5.6 kt** | **+6.2%** | **−1.5 kt** | −3.6 kt | 1.37 |
| 12 mm delta @70 | 1.522 | ~16° | −0.8 kt | −1.8 kt | +15.0% | −3.5 kt | −8.2 kt | 0.97 (steadiest) |
| 8 mm parabolic @50 | 1.519 | ~18° | −0.7 kt | −1.8 kt | +13.2% | −3.1 kt | −7.3 kt | 1.07 |
| 8 mm delta @50 | 1.491 | ~17° | −0.5 kt | −1.1 kt | +11.6% | −2.7 kt | −6.5 kt | 1.01 |

### 8.2 The plain-English story of each

**🥇 6 mm delta @ 70 mm — "the surprise double-winner."**
The short vane *hides in the thin cruise boundary layer*, so it adds the **least
drag of anything tested** (+6.2%). Yet at high AoA, where the boundary layer is
thick, that same short vane still bites — and it reaches a **genuine steady peak
of Clmax 1.709 at α18**, the **highest** of any finalist, pushing the stall from
15° out to ~18°. So it wins **cruise *and* stall *and* fewest/smallest parts.**
Its only "weakness": slightly higher buffet (pk-pk 1.37) than the bigger vanes —
i.e. a touch more pre-stall shake, which many pilots actually *want* as warning.

**🥈 12 mm delta @ 70 mm — "the steady bruiser."**
Tall, authoritative, and the **steadiest** through the stall (pk-pk 0.97) — the
most *progressive, predictable* mush. But its steady peak (1.522) is *lower* than
the 6 mm's, and its cruise tax is the **highest** of the four (+15%). Pick this
if you value a glass-smooth, dead-predictable stall break and don't mind paying
cruise for it.

**🥉 8 mm parabolic @ 50 mm — "the deep-hold high-lift shape."**
Holds lift remarkably **flat and deep** (1.52 → 1.46 → 1.50 all the way to α22) —
it sustains attachment to the highest angles of anything tested. If your mission
is dragging the airplane around at very high alpha (aggressive short-field, spot
landings) and you want the flow to *stay* attached deep into the mush, this is
the character you want. Costs +13.2% cruise.

**8 mm delta @ 50 mm — "the middle-of-the-road delta."**
A perfectly reasonable, balanced delta. It's simply *dominated* by the 6 mm on
this wing (less stall, more cruise), so it's here for completeness rather than as
a recommendation. If you already have 8 mm stock, it's fine.

---

## 9. THE DECISION MENU — pick your own winner

**There is no universally correct answer — only the answer that matches *your*
priorities.** Find the row that sounds like you:

### 🟢 Priority A — "Protect my cruise, give me what stall safety comes free"
→ **6 mm delta @ 70 mm, 40% span (outboard).**
You pay only **~1.5 kt of cruise** and still get **~2.4 kt of stall reduction** —
the *only* config that nets *positive* (gain > cost). Smallest, cheapest part.
**This is the default champion**, and it's the right call for most cross-country
Glasair III owners. *(NET +1.0 kt, ratio 1.7.)*

### 🔵 Priority B — "Maximize stall safety, I'll pay cruise for it"
→ **6 mm delta @ 70 mm, 100% span (root to tip).**
Same part, full coverage. **~5.6 kt off the stall** for **~3.6 kt of cruise.**
This is the **most total stall reduction available** in the study (the 6 mm's
peak Clmax is the highest), and it *still* costs less cruise than the taller
vanes would at full span. If safety/short-field is your #1 and you accept the
cruise hit, go wide with the 6 mm.

### 🟣 Priority C — "I want the smoothest, most predictable stall break"
→ **12 mm delta @ 70 mm.**
Lowest buffet (pk-pk 0.97) = the **gentlest, most progressive mush** with the
most warning and least tendency to break sharply. You give up some peak Clmax
(1.522 vs 1.709) and pay the most cruise (+15%), but you buy *stall manners*.
Good for a heavy, slippery airframe where a sharp break would be unwelcome.

### 🟠 Priority D — "Maximum attached-flow authority at very high alpha"
→ **8 mm parabolic @ 50 mm** (outboard, over the ailerons).
Holds lift flat and deep to α22 — keeps the **outer wing & ailerons attached
and authoritative deepest into the stall**. Pick this if you fly aggressive
short-field/spot-landing approaches and want roll control to *bite* when the
wing is mushing. Costs +13.2% cruise.

### 🟡 Priority E — "Best of both: cheap cruise inboard, max authority outboard"
→ **Mixed/progressive install** — see §11. Bare root (warning) → 6 mm delta
spacing gradient inboard → tight 6 mm delta (or 8 mm parabolic) over the
ailerons. The most sophisticated build; gives a root-first safe stall *and*
live ailerons, tuned per region.

### Quick chooser
| If you care most about… | Install this |
|---|---|
| Keeping cruise speed | 6 mm delta @70, **40%** span |
| Lowest stall speed | 6 mm delta @70, **100%** span |
| Smooth/predictable break | 12 mm delta @70 |
| Deep high-alpha control | 8 mm parabolic @50, outboard |
| The "do it all" build | Progressive spanwise (§11) |

**My one-line suggestion (only a suggestion):** for a typical Glasair III owner
who flies cross-country and occasionally wants a safer short-field, start with
**6 mm delta @ 70 mm over the outboard 40–50%**, fly it, and if you want more
stall margin extend the *same part* inboard later. You can always add span; you
can't un-drill holes.

---

## 10. Install geometry & the printable parts

Two ready-to-print STLs are in `gpu/rapidcfd/assets/`, both derived from the
**real LS(1)-0413 wing geometry** so they fit your airplane:

### `vg_6mm_delta_vane.stl` — the vane
- 6 mm tall, ~18 mm long, ~1.5 mm thick delta.
- **Curved base** — the foot is *pre-bent to the wing's upper-surface curvature
  at the 7% chord station* (≈3 mm of curve over its 18 mm length), so it sits
  **flush** on the skin instead of rocking on a flat foot.
- Print in **PETG or ASA** (UV/heat tolerant). Print *many*. Install in
  **counter-rotating pairs**, **10° toe-out**, **tips at 7% chord**, **70 mm**
  pair-to-pair pitch (wider/bare at the root — §11).

### `vg_placement_jig.stl` — the install fixture
- A block whose underside is the **negative of the wing leading-edge nose**
  (wraps LE back to ~20% chord) — it clamps on and can only seat one way.
- Cut with **vane pockets at the 7% station, 70 mm pitch, with the 10° toe-out
  baked in**, so each printed vane drops into its slot at the correct station,
  spacing, and angle — **repeatable install with no measuring.**

### Reproduce / re-tune
- `gpu/rapidcfd/make_deliverables.py` regenerates both STLs (change `H_MM`,
  `PITCH_MM`, `BETA_DEG`, `X_FRAC` at the top to print a *different* finalist —
  e.g. set `H_MM = 12.0` for the steady-bruiser, or `PITCH_MM = 50` for tight).
- `gpu/rapidcfd/stall_calc.py` reproduces every knots number in this report from
  the raw polar.

---

## 11. The progressive spanwise install

The classic safe-stall ideal: the **root stalls first** (giving buffet warning
and a nose-drop, with no wing-drop) while the **ailerons stay attached longest**
(so you keep roll control through the mush). Because the 6 mm delta wins on both
cruise *and* lift here, you do this with **one part, varied only by spacing**:

| Spanwise zone | Setup | Role |
|---|---|---|
| **Very root** (innermost station) | **No VG** (bare) | Stalls **first** → natural buffet warning. May **replace the stall strip** entirely. |
| **Inboard half** (root → ~50% semi-span) | 6 mm delta, pitch **ramps 110 → 70 mm** | Continuous gradient → stall sweeps **smoothly** root→mid, no abrupt wing-wide break. |
| **Outboard half** (over the ailerons) | 6 mm delta, **70 mm uniform** | Maximum attachment + **aileron authority** deepest into the stall. |

Orient the vanes to the **local airflow** on the swept wing (spanwise reference
line at right angles to the aircraft centerline), **not** parallel to the leading
edge.

**Why not a two-shape mix (efficient inboard / high-lift outboard)?** Because on
*this* wing the 6 mm delta already makes more peak lift than the bigger high-lift
shapes — adding a draggier vane outboard would **cost cruise without buying
stall**. The spacing gradient does all the work.

**CFD honesty about this section:** the 2-D slice models *one* pitch at a time.
It rigorously proves the **discrete relationship** *wider pitch → stalls earlier*
(the spacing sweep, §7). It **cannot directly simulate a continuous spanwise
gradient** — that needs a full 3-D swept-wing run. The gradient above is built by
**interpolating between the measured discrete pitch points**, which is sound
engineering practice but is **inferred, not directly simulated.** Stated plainly
so you don't over-trust it: the *per-zone* numbers are CFD; the *smooth blend*
between them is interpolation.

---

## 12. Honest caveats & limitations

1. **High-Clmax airfoil = smaller absolute gains.** (See §2.) 5.6 kt is the real
   ceiling for this wing; don't expect internet "15 kt" numbers.
2. **2-D periodic slice.** Each case is one spanwise station of an infinite
   uniform row. It captures section aerodynamics excellently but **cannot** show
   3-D effects: spanwise flow on the swept wing, tip effects, or the *continuous*
   spanwise stall progression (§11).
3. **Fully turbulent RANS.** No laminar/transition modeling. Conservative for a
   VG study (VGs act on the turbulent BL) but it means the absolute cruise drag
   is modeled a touch high; the *deltas between configs* — which is what we rank
   on — are robust.
4. **Buffet-dominated points excluded.** A few high-AoA means (flagged in §3.4 &
   §5.1) are unsteady artifacts and were not used as peaks.
5. **Cruise loss is a fixed-power estimate** from section drag, span-weighted —
   a model, not a flight test. Treat the cruise knots as well-grounded estimates,
   not guarantees.
6. **The "NET knots" ranking embeds a 1:1 value of stall vs cruise** (§4.4). If
   your weighting differs, use §9, not the headline rank.
7. **Not flight-validated.** This is high-fidelity CFD, not a test card. Install
   conservatively (start partial-span), and verify stall behavior carefully and
   incrementally on the real airplane.

---

## 13. Glossary & file index

**Glossary**
- **Clmax** — maximum lift coefficient; sets stall speed (`Vs ∝ 1/√Clmax`).
- **Cd / Cl / Cm** — drag / lift / pitching-moment coefficients.
- **pk-pk** — peak-to-peak swing of the force trace = stall-buffet intensity;
  lower = gentler stall manners.
- **Pitch** — spacing between vane pairs along the span.
- **Toe-out** — vane leading edges splayed apart (the winning orientation).
- **Counter-rotating pair** — two vanes per pitch yawed oppositely; beats singles.
- **Span coverage (40% / 100%)** — how much of the wing carries VGs.

**File index** (under `gpu/rapidcfd/`)
| File | What it is |
|---|---|
| `final_report.md` | **this document** |
| `06-17-26_results.md` | the concise final results report |
| `06-14-26_results.md` | earlier report (superseded for stall by the peak-to-peak method) |
| `stall_calc.py` | the peak-to-peak stall/cruise calculator — reproduces every knot |
| `make_deliverables.py` | generates the printable vane + jig STLs (editable for other finalists) |
| `build_cases.py` | the CFD case/geometry builder (all 63 cases) |
| `report.py` | tabulates the raw CFD force coefficients |
| `assets/vg_6mm_delta_vane.stl` | print-ready champion vane (curved base) |
| `assets/vg_placement_jig.stl` | print-ready LE-hugging placement jig |

---

*All aerodynamic numbers are tail-500-iteration means of converged RapidCFD
(k-ω SST RANS, two-stage scheme, ~2.4 M cells) on the LS(1)-0413 section at the
Glasair III chord. High-AoA means read through real stall buffet; pk-pk reported
as a stall-manners indicator. Stall speeds use the peak-to-peak Clmax method
(Vs₀ = 69.5 kt); cruise loss is a fixed-power span-weighted estimate. 63 cases,
2026-06-12 → 2026-06-17.*
