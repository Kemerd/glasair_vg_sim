#!/bin/bash
# Overnight wave: the two FLANGE-FILLET v4 variants (8 cases), queued BEHIND the
# running crisp-v4 wave. v4fs = flange sides filleted; v4fsb = flange sides+back
# filleted; both leave the delta fin sharp. Isolates "soften flange for cruise
# without touching the vortex-shedding delta edges that set the stall."
cd /mnt/l/Dev/glasair_vg_sim/gpu/rapidcfd || exit 1
LOG=/tmp/wave_uvg4_flange.log
echo "=== wave UVG4-FLANGE parked behind crisp-v4 $(date) ===" >> "$LOG"

# Wait until the crisp-v4 wave prints its done banner (non-expiring poll).
for i in $(seq 1 100000); do
    grep -q "=== wave UVG4 done" /tmp/wave_uvg4.log 2>/dev/null && break
    sleep 60
done
echo "=== crisp-v4 done, flange variants running $(date) ===" >> "$LOG"

# Run all 8: both variants x 4 AoA. GPU-idle-gated, one at a time, idempotent.
for c in \
    uvg06v4fs_a02 uvg06v4fs_a15 uvg06v4fs_a18 uvg06v4fs_a20 \
    uvg06v4fsb_a02 uvg06v4fsb_a15 uvg06v4fsb_a18 uvg06v4fsb_a20; do
    if grep -q "^End$" ~/glasair_rapidcfd/"$c"/log.simpleFoam 2>/dev/null; then
        echo "### $c already done, skip" >> "$LOG"; continue
    fi
    while pgrep -x simpleFoam >/dev/null 2>&1; do sleep 30; done
    echo "### launching $c at $(date)" >> "$LOG"
    bash run_all.sh "$c" >> "$LOG" 2>&1
done
echo "=== wave UVG4-FLANGE done $(date) ===" >> "$LOG"
