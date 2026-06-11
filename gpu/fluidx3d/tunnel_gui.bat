@echo off
rem ---------------------------------------------------------------------------
rem  tunnel_gui.bat -- double-click to open the VG tunnel launcher GUI.
rem
rem  Geometry (wing section / wing slice / full tail assembly), control
rem  deflections, analytic VG rows (wing / elevator / rudder) and airspeed,
rem  all without editing tunnel_config.txt. Uses pythonw so no console
rem  window hangs around; the GUI generates missing tail articles itself.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
start "" pythonw tunnel_gui.py
