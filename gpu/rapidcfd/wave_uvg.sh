#!/bin/bash
# User arrowhead-vane test wave: 4 AoA cases, GPU-idle-gated, one at a time.
cd /mnt/l/Dev/glasair_vg_sim/gpu/rapidcfd || exit 1
LOG=/tmp/wave_uvg.log
echo "=== wave UVG (user arrowhead vane) start $(date) ===" >> "$LOG"
for c in uvg06v3_a02 uvg06v3_a15 uvg06v3_a18 uvg06v3_a20; do
    # skip if already finished (idempotent restart)
    if grep -q "^End$" ~/glasair_rapidcfd/"$c"/log.simpleFoam 2>/dev/null; then
        echo "### $c already done, skip" >> "$LOG"; continue
    fi
    # wait for the GPU to be free (non-expiring poll, no collisions)
    while pgrep -x simpleFoam >/dev/null 2>&1; do sleep 30; done
    echo "### launching $c at $(date)" >> "$LOG"
    bash run_all.sh "$c" >> "$LOG" 2>&1
done
echo "=== wave UVG done $(date) ===" >> "$LOG"
