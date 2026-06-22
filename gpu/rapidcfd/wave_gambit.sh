#!/bin/bash
# GAMBIT wave: re-validate the user's winning printed VG STL across spacing /
# incidence / toe orientation. Parks BEHIND the flange wave, then runs the 15
# gambit cases (priority-ordered in gambit_cases.txt) GPU-idle-gated, one at a
# time. Self-running for ~15-20h; the priority order means the most valuable
# configs (100mm progressive-stall polar, toe-in/out) finish first.
cd /mnt/l/Dev/glasair_vg_sim/gpu/rapidcfd || exit 1
LOG=/tmp/wave_gambit.log
echo "=== wave GAMBIT parked behind flange wave $(date) ===" >> "$LOG"

# Wait for the flange wave to print its done banner (non-expiring poll).
for i in $(seq 1 100000); do
    grep -q "=== wave UVG4-FLANGE done" /tmp/wave_uvg4_flange.log 2>/dev/null && break
    sleep 60
done
echo "=== flange wave done, GAMBIT running $(date) ===" >> "$LOG"

# Run each gambit case in priority order, GPU-idle-gated + idempotent.
while IFS= read -r c; do
    [ -z "$c" ] && continue
    if grep -q "^End$" ~/glasair_rapidcfd/"$c"/log.simpleFoam 2>/dev/null; then
        echo "### $c already done, skip" >> "$LOG"; continue
    fi
    while pgrep -x simpleFoam >/dev/null 2>&1; do sleep 30; done
    echo "### launching $c at $(date)" >> "$LOG"
    bash run_all.sh "$c" >> "$LOG" 2>&1
done < /mnt/l/Dev/glasair_vg_sim/gpu/rapidcfd/gambit_cases.txt
echo "=== wave GAMBIT done $(date) ===" >> "$LOG"
