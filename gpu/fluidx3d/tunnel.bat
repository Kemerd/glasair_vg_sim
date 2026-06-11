@echo off
rem ---------------------------------------------------------------------------
rem  tunnel.bat [aoa_deg] [re] [mode] [stl] -- launch the Glasair GPU tunnel.
rem
rem  Examples:
rem    tunnel.bat                 (use tunnel_config.txt as-is)
rem    tunnel.bat 14              (14 deg AoA, keep the rest)
rem    tunnel.bat 14 2.2e6        (stall-speed Reynolds)
rem    tunnel.bat 12 2.9e6 slice  (high-res span-periodic strip)
rem  Generates a per-run config so tunnel_config.txt stays your editable base.
rem ---------------------------------------------------------------------------
setlocal
set BASE=%~dp0tunnel_config.txt
set RUN=%~dp0tunnel_run.txt
copy /y "%BASE%" "%RUN%" > nul
if not "%~1"=="" echo aoa_deg = %~1>> "%RUN%"
if not "%~2"=="" echo re = %~2>> "%RUN%"
if not "%~3"=="" echo mode = %~3>> "%RUN%"
if not "%~4"=="" echo stl = %~4>> "%RUN%"
if not exist "%~dp0results" mkdir "%~dp0results"
start "" /D "L:\Dev\FluidX3D" "L:\Dev\FluidX3D\bin\FluidX3D.exe" "%RUN%"
echo Tunnel launched (config: %RUN%)
endlocal
