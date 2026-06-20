#!/bin/bash
# User CRISP arrowhead-vane (v4, no fillets) test wave: 4 AoA, GPU-idle-gated.
cd /mnt/l/Dev/glasair_vg_sim/gpu/rapidcfd || exit 1
LOG=/tmp/wave_uvg4.log
echo "=== wave UVG4 (crisp v4 vane) start $(date) ===" >> "$LOG"
for c in uvg06v4_a02 uvg06v4_a15 uvg06v4_a18 uvg06v4_a20; do
    if grep -q "^End$" ~/glasair_rapidcfd/"$c"/log.simpleFoam 2>/dev/null; then
        echo "### $c already done, skip" >> "$LOG"; continue
    fi
    while pgrep -x simpleFoam >/dev/null 2>&1; do sleep 30; done
    echo "### launching $c at $(date)" >> "$LOG"
    bash run_all.sh "$c" >> "$LOG" 2>&1
done
echo "=== wave UVG4 done $(date) ===" >> "$LOG"
