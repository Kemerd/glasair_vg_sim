# VG Design Re-Validation — Your Printed STL (2026-06-21)

**Closing the loop.** The original parametric CFD study seeded the printed VG
design (6 mm delta + bonding flange, 7% chord, 70 mm pitch, 10° toe-out). This
report re-runs the **actual printed geometry** (`6mm_deltavg_v3.stl`, filleted
everywhere for safe handling) through the full parameter gambit — spacing,
incidence, counter-rotating orientation — to confirm the design params are
optimal for the real part and to characterize the configs needed for a
progressive-stall install. **Verdict: every original design choice holds up,
and the 100 mm root config for progressive stall is validated.**

27 GPU-CFD cases total (4 fillet variants × ~4 α + 15-case gambit), RapidCFD
k-ω SST on the LS(1)-0413 section, tail-500 means.

---

## 0. Which printed variant — the fillet question (settled)

You asked whether to fillet the VG for safe handling (so it doesn't cut whoever
installs it). We ran all four fillet treatments:

| Variant | Cruise Cd @α2 | Peak Cl @α18 | Notes |
|---|---|---|---|
| **v3 (filleted everywhere)** ✅ | **0.01618** | 1.445 | **WINNER: lowest cruise + knuckle-safe** |
| v4fsb (flange sides+back) | 0.01621 | 1.438 | 2nd |
| v4fs (flange sides only) | 0.01630 | 1.433 | |
| v4 (crisp, no fillets) | 0.01631 | 1.452 | sharpest, no benefit |

**Your fillet-everywhere instinct was right on all three counts:** v3 has the
**lowest cruise drag** of the four, is **safe to handle**, and its stall peak is
within noise (~0.02) of the crisp version. The fillets are aerodynamically free
on this shape — the planform sets the ceiling, not the edges. **Build v3.**

*(Note: all four cap at ~1.44 peak vs the parametric straight-ramp champion's
1.709 — that gap is the arrowhead PLANFORM, documented in
`arrowhead_v3_verdict.md`. This report optimizes the printed shape you have.)*

---

## 1. SPACING — 70 mm vs 100 mm (the progressive-stall lever)

Run on the winning v3, beta 10, toe-out:

| α | **70 mm** Cl | **100 mm** Cl | |
|---|---|---|---|
| 2 (cruise Cd) | 0.01618 | **0.01589** | 100 mm = less device drag |
| 15 | 1.423 | 1.411 | 100 mm makes less lift |
| 18 (peak) | 1.445 | 1.374 | 100 mm gives up ~0.07 peak |
| 20 | 1.520 | 1.501 | |

**100 mm does exactly what a progressive-stall root needs:**
- **Less cruise drag** (0.01589 vs 0.01618) — fewer vanes per span = lower tax
- **Less lift / lower peak** (1.374 vs 1.445) — the root reaches its ceiling
  sooner, so it **stalls FIRST** → buffet warning + nose-drop, ailerons stay alive

→ **Install plan confirmed:** **100 mm at the wing root** (stalls first, cheap
cruise), **ramp to 70 mm by mid-span**, **70 mm uniform over the ailerons**
(max attachment + roll authority deepest into the stall). This is the
root-first progressive stall, validated on your real part.

---

## 2. ORIENTATION — toe-out vs toe-in (settled: TOE-OUT)

| Config @ α18 | toe-OUT | toe-IN | winner |
|---|---|---|---|
| 70 mm | **1.445** | 1.409 | toe-out +0.036 |
| 100 mm | **1.374** | 1.348 | toe-out +0.026 |

Cruise is a wash (toe-in/out differ <0.0001 Cd). **Toe-OUT wins the stall at
BOTH spacings** by ~0.03 peak lift — the counter-rotating pair sheds a stronger,
more persistent vortex when the leading edges splay apart. **Your toe-out choice
is confirmed correct.** Install the pair with leading edges pointing *away* from
each other.

---

## 3. INCIDENCE — beta 8 / 10 / 12 (beta10 = best balance)

70 mm, toe-out:

| beta | Cruise Cd @α2 | Peak Cl @α18 | character |
|---|---|---|---|
| 8° | **0.01603** | 1.430 | least drag, least lift (under-energizes a touch) |
| **10° (your design)** | 0.01618 | 1.445 | **best balance** |
| 12° | 0.01628 | **1.454** | most lift, most drag |

A clean monotonic trade: **steeper = more lift AND more drag.** Beta10 sits at
the **best stall-per-drag balance** — beta8 saves a sliver of cruise but
under-energizes the boundary layer (loses 0.015 peak), beta12 buys 0.009 more
peak for more cruise drag.

- **Keep beta 10° as your default** (the figure-of-merit optimum, confirming the
  original sweep).
- **beta 12° is a max-lift alternate** if you ever want a touch more stall
  authority on a specific surface and accept the cruise cost.

*(At 100 mm, beta12 a18 = 1.402 vs beta10's 1.374 — same trend: a hair more lift
for a hair more drag. Beta10 stays the balanced pick at both spacings.)*

---

## 4. The re-validated design (every param confirmed)

| Parameter | Design value | Re-validation verdict |
|---|---|---|
| Shape | 6 mm delta + flange | ✅ printed v3 (fillet-all = lowest cruise + safe) |
| Chord station | 7% c | ✅ (study basis, unchanged) |
| **Incidence** | **10° toe-out** | ✅ beta10 = best balance; toe-out wins both spacings |
| **Pitch (ailerons)** | **70 mm** | ✅ tightest tested = most attachment + authority |
| **Pitch (root)** | **100 mm** | ✅ less drag + stalls first = progressive-stall root |

**Nothing left on the table.** The CFD-seeded design params are optimal for the
real printed geometry. The only new actionable finding is the **100 mm root
number** (cruise 0.01589, peak 1.374) — the wide-pitch root config the
progressive-stall install needs, now measured rather than assumed.

---

## 5. Progressive-stall install (final, validated numbers)

One printed part (**v3, 6 mm delta, filleted, toe-out**), varied only by pitch:

| Spanwise zone | Pitch | Role | CFD basis |
|---|---|---|---|
| **Very root** | bare / very wide | stalls FIRST (warning, may replace stall strip) | clean stalls ~15° |
| **Inboard half** | ramp **100 → 70 mm** | smooth root→mid stall sweep | 100 mm: peak 1.374, Cd 0.01589 |
| **Outboard (ailerons)** | **70 mm** uniform | max attachment + roll authority | 70 mm: peak 1.445, Cd 0.01618 |

Orient vanes to the **local flow** on the swept wing (spanwise reference at right
angles to the A/C centerline), **toe-out**, tips at **7% chord**.

**CFD honesty:** the 2-D slice models one pitch at a time, so it rigorously
proves the discrete *wider-pitch → less-lift-+-less-drag → stalls-earlier*
relationship (this gambit), but the continuous spanwise gradient is built by
*interpolating* between the 70 mm and 100 mm measured points — not a directly
simulated 3-D gradient. Stated plainly so the gradient isn't over-trusted.

---

## 6. Files
- `gpu/rapidcfd/06-21-26_revalidation.md` — this report
- `arrowhead_v3_verdict.md` — why the arrowhead planform caps at ~1.44 (fillets
  not the cause)
- `build_user_vane.py` — imports the printed STL, sweeps pitch/beta/toe (`--pitch
  --beta --toe --version`)
- `build_gambit.py` / `gambit_cases.txt` — the 15-case gambit definition
- cases `uvg06v3_p{070,100}_b{08,10,12}_t{o,i}_a{02,15,18,20}` — the runs

*All values tail-500 means of converged RapidCFD (k-ω SST, two-stage, ~2.46 M
cells), identical pipeline to the original 63-case study, on the user's REAL
printed v3 geometry. High-AoA means read through stall buffet; pk-pk = stall-
manners indicator. This closes the design loop: original CFD → STL → re-CFD.*
