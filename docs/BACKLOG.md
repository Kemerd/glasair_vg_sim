# Engineering notes — GPU options (DECISION PENDING, owner's call)

Analysis notes only. Nothing here is decided; scheduling and scope belong
to the project owner. Recorded so the technical comparison survives session
boundaries.

## GPU acceleration: AmgX pressure-solve offload

**Option:** benchmark NVIDIA AmgX (via PETSc4FOAM + AmgXWrapper +
FOAM2CSR) against CPU GAMG, on the workstation's RTX 5090. Payoff scales
with mesh size; largest at the ~10 M cell Phase-3 meshes.

**Why this path and not others (evaluated 2026-06-10):**
* AmgX offloads only the pressure linear solve but keeps mainline ESI
  OpenFOAM v2506 — kOmegaSSTLM, the jBAY fvOptions (M2), and the whole M1
  validation chain stay valid. Published results: ~4-8x on the pressure
  step, ~2-3x wallclock, case dependent (NVIDIA/ESI, 8M-cell lid cavity).
* AMG solves are memory-bandwidth-bound, so the RTX 5090's ~1.8 TB/s can
  win despite GeForce FP64 FLOP gimping (1/64 rate) — bandwidth, not FLOPs,
  is the binding constraint for sparse multigrid.
* RapidCFD (full-CUDA fork): REJECTED — verified its RAS model directory
  carries no transition model at all (kOmegaSSTLM mandatory per spec), 12
  year old code base, poor linear-solver performance per the gpuFOAM
  community's own assessment.
* stdpar/gpuFOAM mainline porting: WATCH — community targets merging into
  OpenFOAM mainline (~v2606). If/when released, re-evaluate; the trust-
  deltas philosophy makes re-validation on a new release tractable.
* FP64-emulation (Ozaki/cuBLAS): not applicable — accelerates dense GEMM;
  CFD kernels are sparse and bandwidth-bound.

**Not before Phase 3:** current 2D validation cases are ~25k cells and
solve in about a minute on 8 CPU ranks; GPU transfer/launch overhead
exceeds the entire solve at this size.

## Sweep throughput tuning (after M1 gate passes)

2D cases are over-decomposed at 8 ranks (~3k cells/rank). Benchmark
4 ranks/case x 4-6 cases wide against the current 8x2 on the 14900K
(memory-bandwidth-bound workload: expect the optimum well below 100% CPU
utilization; document the measured sweet spot in the README).
