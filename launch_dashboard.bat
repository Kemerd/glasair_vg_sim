@echo off
rem ---------------------------------------------------------------------------
rem  launch_dashboard.bat -- double-click to open the live CFD dashboard.
rem
rem  Starts scripts/live_dashboard.py detached from this console window using
rem  pythonw (no console), so closing the terminal never kills the dashboard.
rem  Panels: Cl convergence, residuals, per-core CPU, GPU/render status, and
rem  the GPU-rendered velocity field of the freshest snapshot.
rem  Close the dashboard window itself to stop it. Safe to run while a sweep
rem  is solving -- the dashboard only reads case data.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
start "" pythonw scripts\live_dashboard.py
echo Dashboard launching... the window appears within a few seconds.
timeout /t 3 > nul
