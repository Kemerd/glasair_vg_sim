@echo off
rem ---------------------------------------------------------------------------
rem  act2.cmd -- Act II of the overnight VG campaign (runs after the 14-deg
rem  height ladder finishes):
rem    block 1: 16 deg x {clean, VG12, VG16}  -- riding the stall break
rem    block 2: 18 deg x {clean, VG12, VG16}  -- past clean stall (Strausak test)
rem    block 3: 14 deg x {speck}              -- quasi-2D artifact meter
rem  All at Re 2.21e6 (80 mph at the aileron chord). Sequential; each case
rem  auto-closes. Progress appends to results/suite/suite.log (same monitor).
rem ---------------------------------------------------------------------------
cd /d L:\Dev\glasair_vg_sim
python gpu\fluidx3d\run_block.py --aoa 16 --mph 80 --designs clean,vg12mm,vg16mm
python gpu\fluidx3d\run_block.py --aoa 18 --mph 80 --designs clean,vg12mm,vg16mm
python gpu\fluidx3d\run_block.py --aoa 14 --mph 80 --designs speck
echo Act II complete.
