# validation_2d -- Phase-1 clean-section OpenFOAM case template

Template tree for the LS(1)-0413 2D validation series (spec Phase 1).
Nothing in `template/` runs directly: tokens of the form `@NAME@` are
placeholders, and `system/blockMeshDict` is a deliberate stub. Instantiate a
runnable case with:

```
python scripts/build_validation_case.py --aoa 4 --re 3e6 --level 0
```

which writes `cases/runs/val2d_aoa4_re3e6_lvl0/` -- a complete, dry-run
friendly case directory (no OpenFOAM needed to build it; `Allrun` executes
it later inside WSL2 Ubuntu with ESI OpenFOAM v2506).

What the builder does, in order:

1. Resamples the committed LS(1)-0413 coordinates (blunt TE kept -- the real
   0.0055 c base; rationale in the per-case README) onto the 241-point
   cosine loop via `geometry.airfoil.resample_airfoil`.
2. Sizes the first wall-normal cell from the y+ < 1 correlation chain in
   `scripts/first_cell_height.py` (imported, not duplicated) and lays out a
   5-block structured C-grid: far field 25 c, wake 25 c, growth ratio
   <= 1.2, >= 30 cells inside the estimated boundary layer, one cell thick
   in z with `empty` front/back patches. `--level {0,1,2}` applies sqrt(2)
   refinement steps for the spec section 4 mesh-independence study.
3. Fills the template tokens: freestream vector and forceCoeffs lift/drag
   directions rotated per AoA, nu from `aircraft.yaml` so Re is exact at
   c = 1 m, Mack/Langtry-Menter inlet turbulence chain, and the wall-normal
   boundary-layer sample lines on the suction surface.

Every boundary-condition choice is justified inline in the `template/0/`
dictionaries; the per-case `README.md` carries the flow-condition table, the
Re 6M Mach note, the mesh summary, and the exact pimpleFoam pseudo-transient
fallback deltas. Unit tests: `tests/test_case_gen.py`.

## Post-stall fallback (template/pimple_overrides/)

`template/pimple_overrides/` holds the COMPLETE pimpleFoam localEuler
variant dictionaries (controlDict / fvSchemes / fvSolution; deltas marked
`VARIANT:` inline). The builder copies the directory into every
instantiated case; when a post-stall point (alpha >= 14 deg by default,
`--pimple-alpha` to change) fails the steady convergence gate,
`scripts/run_validation.py` copies these files over the case's `system/`
and re-runs the solve once -- a reproducible, mechanical swap, never a hand
edit. The gate report (`validation/compare_gate.py`) reads `application`
from each case's controlDict and flags those points as pseudo-transient in
`validation/report.md`. Full delta documentation: `template/README.md`.

## Transition-gate data chain

`template/system/wallShearStressObj` dumps the airfoil-patch kinematic wall
shear (raw surface format, `onEnd`) to `postProcessing/wallCf/<time>/`;
`scripts/extract_cf.py` converts it to the suction-surface
`postProcessing/cf_upper.csv` (`Cf = |tau_w/rho| / (0.5 U_inf^2)`), which
feeds the transition detector in `validation/compare_gate.py` (gate c).
