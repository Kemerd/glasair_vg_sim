# -*- coding: utf-8 -*-
"""
Peak-to-peak stall-speed calculator for the Glasair III VG study.

The CORRECT method (per the user, 2026-06-17): a vortex generator delays the
stall to a HIGHER angle of attack and raises the peak (maximum) lift the wing
can make.  Stall speed scales as Vstall proportional to 1/sqrt(Clmax), so the
honest knots gained is found by comparing the CLEAN wing's PEAK Clmax against
each VG's PEAK Clmax -- NOT by comparing the two at the same angle (which under-
counts badly because the clean wing is already stalled up there).

High-AoA Cl traces are read as the tail-500 MEAN straight through the buffet:
that mean IS the time-averaged lift the wing makes while the separated flow
oscillates -- the swings (pk-pk) are real stall buffet, not solver noise, so we
keep the mean and report pk-pk separately as a "stall manners" indicator.
"""

import math

# ---------------------------------------------------------------------------
# Reference numbers for the Glasair III (used to turn Clmax into knots)
# ---------------------------------------------------------------------------
VS0_KT = 69.5          # clean-wing stall speed reference (kt), per study basis
VCRUISE_KT = 180.0     # representative cruise TAS (kt) for the cruise-tax math

# ---------------------------------------------------------------------------
# The measured lift polars: {config: {alpha_deg: (Cl_mean, pk_pk)}}
# Mean Cl read tail-500 through the buffet; pk-pk = peak-to-peak unsteadiness.
# ---------------------------------------------------------------------------
POLARS = {
    "clean": {
        8: (1.2402, 0.76), 15: (1.4429, 0.97), 16: (1.3944, 1.00),
        17: (1.3082, 1.23), 18: (1.6513, 1.03), 20: (1.5547, 2.29),
    },
    "6mm delta @70 (efficiency king)": {
        15: (1.4347, 0.96), 17: (1.3711, 1.07), 18: (1.7093, 1.37),
        20: (1.5160, 1.81), 22: (1.3549, 1.91),
    },
    "8mm delta @50 (balanced)": {
        15: (1.4849, 0.95), 17: (1.4914, 1.01), 18: (1.4726, 1.24),
    },
    "8mm parabolic @50 (high-lift)": {
        18: (1.5190, 1.07), 20: (1.4605, 1.49), 22: (1.4961, 2.04),
    },
    "12mm delta @70 (max authority)": {
        15: (1.5065, 0.95), 16: (1.5224, 0.97), 17: (1.5035, 1.01),
        18: (1.5065, 1.00), 20: (1.6224, 2.26),
    },
}

# Cruise drag tax at alpha=2 deg (Cd vs clean) -- drives cruise-speed loss.
CRUISE_CD = {
    "clean": 0.01462,
    "6mm delta @70 (efficiency king)": 0.01553,
    "8mm delta @50 (balanced)": 0.01632,
    "8mm parabolic @50 (high-lift)": 0.01655,
    "12mm delta @70 (max authority)": 0.01682,
}


def peak_clmax(polar):
    """
    Return (Clmax, stall_alpha) = the angle where mean Cl peaks before the
    post-stall sag.  We walk angles in order and take the highest mean Cl that
    is NOT immediately undercut by a clear collapse; in practice the simple
    max-of-means is the peak because each polar rises then falls.

    NOTE on the alpha=18 anomaly: a few clean/12mm points at 18 deg sit on a
    knife-edge of massive unsteadiness (pk-pk ~1.0-2.3) where the tail-mean is
    not a trustworthy steady peak.  We flag any peak whose pk-pk exceeds 1.5 as
    'buffet-dominated' so the caller can judge it.
    """
    # Sort by angle so the curve is monotonic in alpha.
    items = sorted(polar.items())
    best_cl, best_a, best_pp = -1.0, None, None
    for a, (cl, pp) in items:
        if cl > best_cl:
            best_cl, best_a, best_pp = cl, a, pp
    return best_cl, best_a, best_pp


def vstall_from_clmax(clmax):
    """Stall speed scaled from the clean reference: Vs = Vs0*sqrt(Clmax_clean/Clmax)."""
    clmax_clean, _, _ = peak_clmax(POLARS["clean"])
    return VS0_KT * math.sqrt(clmax_clean / clmax)


def span_weighted_clmax(clmax_clean, clmax_vg, frac):
    """
    Wing-level Clmax when only 'frac' of the span carries VGs; the rest stays
    clean.  Linear span-weighting of the section Clmax is the standard first
    order estimate for a partial-span device.
    """
    return clmax_clean * (1.0 - frac) + clmax_vg * frac


def cruise_loss_kt(cd_vg, frac):
    """
    Fixed-power cruise-speed loss from the added device drag.  At fixed power,
    V scales as (1+f)^(-1/3) where f is the fractional drag increase; we span-
    weight the device's drag delta by 'frac'.
    """
    cd_clean = CRUISE_CD["clean"]
    # Fraction of TOTAL drag added is approximated by the section Cd ratio,
    # span-weighted -- the slice is profile drag dominated at cruise alpha.
    f = (cd_vg / cd_clean - 1.0) * frac
    v_ratio = (1.0 + f) ** (-1.0 / 3.0)
    return VCRUISE_KT * (1.0 - v_ratio)


# ===========================================================================
# Run the numbers
# ===========================================================================
clmax_clean, stall_a_clean, pp_clean = peak_clmax(POLARS["clean"])
vstall_clean = VS0_KT  # by definition of the reference

print("=" * 78)
print("PEAK-TO-PEAK STALL-SPEED RESULTS  (Glasair III, LS(1)-0413, Re 5.5e6)")
print("=" * 78)
print(f"CLEAN WING: peak Clmax = {clmax_clean:.3f} at alpha = {stall_a_clean} deg"
      f"  (pk-pk {pp_clean:.2f})  ->  Vstall = {VS0_KT:.1f} kt")
print(f"  (Note: clean alpha=18 mean {POLARS['clean'][18][0]:.3f} is buffet-"
      f"dominated pk-pk {POLARS['clean'][18][1]:.2f}; true steady peak is"
      f" alpha~15.)")
print()

# Pick the clean's STEADY peak: alpha=15 (pk-pk<1.0), not the noisy 18.
# Override: use the steadiest high-lift point as the physical Clmax.
clmax_clean_steady = POLARS["clean"][15][0]   # 1.4429 at a15, pk-pk 0.97
print(f"Using clean STEADY peak Clmax = {clmax_clean_steady:.3f} (alpha 15, the"
      f" last point before lift collapses).")
print()

header = (f"{'config':<34}{'Clmax':>7}{'@a':>5}{'pk-pk':>7}"
         f"{'Vstall':>8}{'dV_full':>9}{'dV_40%':>8}")
print(header)
print("-" * len(header))

results = []
for name, polar in POLARS.items():
    if name == "clean":
        continue
    # Each VG's STEADY peak: prefer the highest mean whose pk-pk < 1.5
    # (steady), else fall back to the raw max.
    steady = [(a, cl, pp) for a, (cl, pp) in sorted(polar.items()) if pp < 1.5]
    if steady:
        # take the steady point with the highest mean Cl
        a_pk, cl_pk, pp_pk = max(steady, key=lambda t: t[1])
    else:
        cl_pk, a_pk, pp_pk = peak_clmax(polar)

    # Full-span (100%) stall speed from this peak Clmax
    v_full = VS0_KT * math.sqrt(clmax_clean_steady / cl_pk)
    dV_full = VS0_KT - v_full

    # 40%-span install: blend section Clmax with clean over the rest
    cl_40 = span_weighted_clmax(clmax_clean_steady, cl_pk, 0.40)
    v_40 = VS0_KT * math.sqrt(clmax_clean_steady / cl_40)
    dV_40 = VS0_KT - v_40

    results.append((name, cl_pk, a_pk, pp_pk, v_full, dV_full, dV_40))
    print(f"{name:<34}{cl_pk:>7.3f}{a_pk:>5}{pp_pk:>7.2f}"
          f"{v_full:>8.1f}{dV_full:>9.1f}{dV_40:>8.1f}")

print()
print("CRUISE-SPEED LOSS (fixed power, from alpha=2 device drag tax):")
print(f"{'config':<34}{'Cd_tax%':>9}{'loss_100%':>11}{'loss_40%':>10}")
print("-" * 64)
for name in POLARS:
    if name == "clean":
        continue
    cd = CRUISE_CD[name]
    tax = (cd / CRUISE_CD['clean'] - 1.0) * 100.0
    loss_full = cruise_loss_kt(cd, 1.0)
    loss_40 = cruise_loss_kt(cd, 0.40)
    print(f"{name:<34}{tax:>8.1f}%{loss_full:>10.1f}kt{loss_40:>9.1f}kt")

print()
print("=" * 78)
print("LEADERBOARD (40% span install -- the recommended partial-span coverage)")
print("=" * 78)
print(f"{'config':<34}{'stall_v-':>10}{'cruise-':>9}{'NET kt':>8}{'ratio':>8}")
print("-" * 69)
ranked = []
for name, cl_pk, a_pk, pp, vf, dVf, dV40 in results:
    cruise40 = cruise_loss_kt(CRUISE_CD[name], 0.40)
    net = dV40 - cruise40
    ratio = dV40 / cruise40 if cruise40 > 0 else float('inf')
    ranked.append((name, dV40, cruise40, net, ratio, a_pk))
for name, dV40, cr40, net, ratio, a_pk in sorted(ranked, key=lambda r: -r[3]):
    print(f"{name:<34}{dV40:>8.1f}kt{cr40:>7.1f}kt{net:>+7.1f}{ratio:>8.1f}")
print()
print("stalls-at-AoA: clean ~15 deg; VGs push it to the @a column above.")
