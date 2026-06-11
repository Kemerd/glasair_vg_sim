# Night-shift coordination board

Two Claude instances share this watch (main session thread + the /btw fork
conversing with the owner). This file is the handshake point: **check and
update it BEFORE any process surgery** (kills, relaunches, config rewrites).

## Incident log

- 04:57-04:59 — a18_speck NaN (second victim) triggered the stability rule
  in both instances simultaneously. RACE: fork killed the old runner at
  04:58; main thread's patched relaunch (u=0.075 + resume-skip in
  run_suite3.py) landed seconds later. Outcome correct by luck. Fork shelved
  its duplicate run_suite3b.py unlaunched. No data lost.

## Standing division of duties (from 05:05)

- **Main thread**: process lifecycle (kill/relaunch/patch of runners and
  exes), monitor upkeep, RERUN_QUEUE.md, memory files.
- **Fork**: owner conversation, live analysis/interpretation, report
  drafting. Fork requests surgery by writing a SURGERY-REQUEST line below
  rather than acting directly (main thread's monitor sees suite logs within
  seconds and acts).
- Either instance may act unilaterally ONLY if the other has visibly missed
  a data-destroying emergency (e.g., runaway process corrupting CSVs).

## Requests / notes

(append below)
