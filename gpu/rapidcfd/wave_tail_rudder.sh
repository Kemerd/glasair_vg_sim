#!/bin/bash
# Tail RUDDER VG spacing sweep: clean baseline + 30/50/70mm at the IMP74 station
# (v3 vane, 100mm ahead of the 25deg-deflected rudder hinge). GPU-idle-gated.
cd /mnt/l/Dev/glasair_vg_sim/gpu/rapidcfd || exit 1
LOG=/tmp/wave_tail_rudder.log
echo "=== wave TAIL-RUDDER start $(date) ===" >> "$LOG"
for c in tail_rudder_clean tail_rudder_p030 tail_rudder_p050 tail_rudder_p070; do
    if grep -q "^End$" ~/glasair_rapidcfd/"$c"/log.simpleFoam 2>/dev/null; then
        echo "### $c already done, skip" >> "$LOG"; continue
    fi
    while pgrep -x simpleFoam >/dev/null 2>&1; do sleep 30; done
    echo "### launching $c at $(date)" >> "$LOG"
    bash run_all.sh "$c" >> "$LOG" 2>&1
done
echo "=== wave TAIL-RUDDER done $(date) ===" >> "$LOG"
