# -*- coding: utf-8 -*-
"""
Re-validation GAMBIT for the user's REAL printed VG STL.

Closes the loop: the original parametric CFD seeded the STL design (6mm delta +
bonding flange, 7%c, 70mm, beta10, toe-out); now we sweep the ACTUAL printed
geometry across spacing / incidence / counter-rotating orientation to confirm
those params are optimal for the real part and find the configs the user needs
for a progressive-stall install (wide pitch = less lift+drag at the ROOT, tight
pitch over the AILERONS).

WINNER vane (set below) = the best of the user's variants. As of 2026-06-20 the
data says v3 (filleted-everywhere) wins: lowest cruise Cd (0.01618), knuckle-
safe, stall-neutral vs crisp. If v4fsb upsets that, change WINNER and re-run.

This script only PRINTS the build+run commands per config (it does not launch);
the overnight chain script drives the actual GPU runs one at a time.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- the winning user vane variant (change if v4fsb upsets v3) ---------------
WINNER = "v3"          # filleted-everywhere: best cruise + safe + stall-neutral

# --- the gambit matrix -------------------------------------------------------
# Each entry: (pitch_mm, beta_deg, toe, [alphas]). Ordered by VALUE so the most
# important configs run first (in case the 8hr window runs short).
#
# Priority logic:
#  1. 100mm @ baseline beta/toe  -> the PROGRESSIVE-STALL root config the user
#     explicitly wants (less lift + less Cd at the root). FULL polar.
#  2. toe-IN vs toe-OUT at 70mm AND 100mm -> orientation question, at stall+cruise.
#  3. beta sweep (8/12) at 70mm -> confirm beta10 is the incidence optimum.
# (70mm/beta10/toe-out baseline already exists as v3 a02/a15/a18/a20 - reuse it.)
GAMBIT = [
    # 1. 100mm progressive-stall ROOT config -- full polar (THE deliverable)
    (100, 10, "out", [2, 15, 18, 20]),
    # 2. orientation: toe-IN at 70 and 100 (cruise + the peak a18 + a20)
    (70,  10, "in",  [2, 18, 20]),
    (100, 10, "in",  [2, 18, 20]),
    # 3. incidence sweep at 70mm (cruise + peak) -- bracket beta10
    (70,   8, "out", [2, 18]),
    (70,  12, "out", [2, 18]),
    # 4. incidence at 100mm too (peak only) -- does wider want more beta?
    (100, 12, "out", [18]),
]


def tag_for(pitch, beta, toe):
    return f"uvg06{WINNER}_p{pitch:03d}_b{beta:02d}_t{toe[0]}"


def build_all():
    """Build every gambit case; return the ordered list of case names."""
    names = []
    for pitch, beta, toe, alphas in GAMBIT:
        alpha_args = [str(a) for a in alphas]
        cmd = [sys.executable, str(HERE / "build_user_vane.py"),
               "--version", WINNER, "--pitch", str(pitch),
               "--beta", str(beta), "--toe", toe, "--alphas", *alpha_args]
        print(f"\n=== building {tag_for(pitch,beta,toe)}  alphas={alphas} ===")
        subprocess.run(cmd, check=True)
        for a in alphas:
            names.append(f"{tag_for(pitch, beta, toe)}_a{int(a):02d}")
    return names


if __name__ == "__main__":
    names = build_all()
    print("\n" + "=" * 70)
    print(f"GAMBIT built: {len(names)} cases on winner '{WINNER}'")
    print("CASES:", " ".join(names))
    # Write the ordered case list so the chain script can read it.
    (HERE / "gambit_cases.txt").write_text("\n".join(names) + "\n", newline="\n")
    print(f"case list -> {HERE / 'gambit_cases.txt'}")
