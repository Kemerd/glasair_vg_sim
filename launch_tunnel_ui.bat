@echo off
rem Double-click to open the Glasair GPU tunnel launcher UI (AoA, airspeed,
rem VG configuration -> auto-rotating tunnel view). No console window.
cd /d "%~dp0"
start "" pythonw gpu\fluidx3d\launcher.py
