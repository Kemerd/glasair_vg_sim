# Arrowhead VG (6mm_deltavg_v3) — A/B Test Verdict (2026-06-19)

**The idea:** take the winning 6 mm delta, **round/fillet the front + side faces**
into an arrowhead/paper-airplane shape to cut cruise drag, while **leaving the
back edge tall & sharp** to preserve clean separation (the stall recovery). Add
a small bonding flange around the base. Tested as a counter-rotating pair at the
identical 7% c / 70 mm pitch / 10° toe-out as the champion — a clean A/B.

**Result: the arrowhead lost on BOTH axes. The plain delta stays champion.**

---

## The head-to-head numbers (tail-500 means)

| α | Arrowhead v3 Cl | Champion 6mm Cl | who's higher |
|---|---|---|---|
| 2° (cruise) | Cd **0.01618** | Cd **0.01553** | champion (lower drag) |
| 15° | 1.423 | 1.435 | ~tie (champion hair up) |
| **18° (the peak)** | **1.445** | **1.709** | **champion by a mile** |
| 20° | 1.520 | 1.516 | ~tie (both past peak) |

### Stall (peak-to-peak Clmax method, Vs₀ = 69.5 kt)
| | peak Clmax | @α | ΔVstall full | ΔVstall 40% |
|---|---|---|---|---|
| **Champion 6 mm delta** | **1.709** | 18° | **−5.6 kt** | **−2.4 kt** |
| Arrowhead v3 | 1.445 | 18° | **−0.1 kt** | ~0 kt |

### Cruise (fixed-power loss)
| | Cd tax | loss full | loss 40% |
|---|---|---|---|
| **Champion 6 mm delta** | **+6.2%** | −3.6 kt | −1.5 kt |
| Arrowhead v3 | +10.7% | −6.0 kt | −2.5 kt |

**So the arrowhead is ~0.5 kt WORSE in cruise *and* gives essentially ZERO
stall benefit (−0.1 vs −5.6 kt). It's worse on both counts.**

---

## Why it failed — the physics lesson

The hypothesis was "sharp back = keeps the vortex, smooth front = less drag."
The CFD says that's backwards about **where the vortex is born**:

- A vortex generator works by shedding a **tip/edge vortex off its
  LEADING & SIDE edges** as the flow spills over them. Those edges need to be
  **sharp** to fix the separation point and shed a tight, energetic, coherent
  vortex that survives downstream and re-energizes the boundary layer.
- The plain delta has **sharp edges all around**, so it sheds a strong vortex
  that holds the flow attached all the way to **α18 → Clmax 1.709**.
- The arrowhead **rounded/filleted exactly those leading & side edges**. A
  rounded edge lets the flow stay attached *over the vane* instead of spilling
  off as a sharp vortex — so the shed vortex is **weak and diffuse**. It loses
  its grip early: by α18 the arrowhead is already letting go (Cl sagging to
  1.445, basically near-stalled), instead of climbing to 1.709.
- The **sharp *trailing* edge didn't save it** — by the time the flow reaches
  the back of the vane, the vortex strength was already set by the (now rounded)
  leading edges. Sharp-back-only is the wrong lever.
- The **flange** added wetted area that *raised* cruise drag — so even the one
  thing it was supposed to help (cruise) went the wrong way, because the
  flange's extra surface outweighed the rounded-nose benefit at this tiny scale.

**One-line takeaway:** for a passive VG, **the leading & side edges must stay
sharp** — that's what makes the vortex. Rounding them for cruise trades away far
more stall than it saves drag. (And at 6 mm, a vane this small already hides in
the thin cruise boundary layer, so there's almost no cruise to recover by
smoothing it — the plain delta is *already* near-optimal on cruise.)

---

## Recommendation

**Stick with the plain sharp-edged 6 mm delta** (`assets/vg_6mm_delta_vane.stl`).
It remains the champion on every axis.

**If you want to keep iterating** on a low-drag shape, the productive direction
is NOT rounding the vortex-shedding edges. Options that *could* help cruise
without killing the vortex:
1. **Thinner vane / sharper trailing taper only** — keep the LE & side edges
   crisp, just thin the body. (We could CFD a 6 mm delta at ~1.0 mm thickness.)
2. **Remove/shrink the flange** — bond with a fillet of adhesive instead of a
   printed skirt; the flange was pure drag here.
3. **Slightly shorter chord (lower l/h)** — less wetted area, same height; might
   shave cruise with a smaller stall cost. Worth a sweep.
4. **A true low-drag micro-VG** (even shorter, e.g. 4–5 mm) accepts a bit less
   stall for less cruise — a different point on the same trade curve.

Any of those I can build & run as the next A/B whenever you want — just say which.

---

## Files
- `assets/user_vanes/6mm_deltavg_v3.stl` — the tested arrowhead vane (kept)
- `build_user_vane.py` — imports a custom vane STL, reorients to the wing frame,
  stamps the A/B case at 7%c / 70mm / 10° (reusable for the next iteration)
- cases `uvg06v3_a02/a15/a18/a20` — the 4 runs behind this verdict

*All values tail-500 means of converged RapidCFD (k-ω SST, two-stage, ~2.46 M
cells), identical pipeline to the 63-case main study. Stall = peak-to-peak Clmax
method. The arrowhead was placed at the exact champion config for a fair A/B.*
