# Rerun queue — cases needing a second attempt

| case | reason | recipe |
|---|---|---|
| act3_a16_h10p90 | lattice instability: healthy until step 32k, then NaN (16 deg buffet spiked past stability at u=0.10) | rerun with `u = 0.075` in the per-case config (smaller lattice speed = stability headroom; ~33% more steps for the same 0.5 s physics) |

Watch rule for the rest of Act III: ONE NaN case = isolated bad luck, rerun
solo. If a SECOND case NaNs tonight, the whole 16-20 deg envelope needs the
u = 0.075 treatment and the affected angles' results should be regenerated
for consistency (mixed-u comparisons are still valid via coefficients, but
keep u uniform within a comparison block where possible).

**RULE TRIGGERED 04:57 (a18_speck NaN = second victim, and the 18-deg
baseline at that).** Action taken: suite3 killed and relaunched with
u = 0.075 + resume logic (banked non-NaN cases skip; NaN victims rerun
automatically). Consequences for the morning analysis: (1) the six good
16-deg results are u=0.10, the a16_h10p90 rerun and ALL 18/20-deg results
are u=0.075 — same-coefficient comparisons remain valid, note the mixed-u
footnote within the 16-deg block; (2) ~18.7 min/case from here -> Act III
ends ~09:30, so the Act IV chain will likely hit its deadline and skip
(owner's machine-free-by-9:30 rule wins; Act IV is a one-command morning
run if wanted: python gpu/fluidx3d/run_suite4.py).
