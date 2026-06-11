@echo off
rem ---------------------------------------------------------------------------
rem  solve_case.cmd <case_dir_name> -- run one validation case's Allrun chain
rem  inside WSL2 (blockMesh / checkMesh / decomposePar / solver / reconstruct).
rem
rem  Usage:   scripts\solve_case.cmd val2d_aoa4_re3e6_lvl0
rem  The window stays open while the solver runs (it IS the WSL keep-alive --
rem  WSL reaps background processes once their last client exits, so this
rem  process babysits the run); console output also lands in the case's
rem  allrun.console. Launch minimized/hidden for unattended runs.
rem ---------------------------------------------------------------------------
if "%~1"=="" (
    echo usage: solve_case.cmd ^<case_dir_name^>
    exit /b 2
)
wsl -e bash -lc "cd /mnt/l/Dev/glasair_vg_sim/cases/validation/%~1 && ./Allrun > allrun.console 2>&1"
exit /b %errorlevel%
