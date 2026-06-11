@echo off
rem ---------------------------------------------------------------------------
rem  rebuild_tunnel.bat -- sync setup_glasair.cpp into the FluidX3D tree and
rem  rebuild bin\FluidX3D.exe (Release x64, VS2022 MSBuild).
rem
rem  Run this after any edit to setup_glasair.cpp. The tunnel window must be
rem  CLOSED first: a running FluidX3D.exe holds a lock on its own binary and
rem  the link step fails with LNK1104.
rem ---------------------------------------------------------------------------
tasklist /FI "IMAGENAME eq FluidX3D.exe" 2>nul | find /I "FluidX3D.exe" >nul
if not errorlevel 1 (
    echo FluidX3D is RUNNING - close the tunnel window first, then re-run this.
    pause
    exit /b 1
)
copy /y "%~dp0setup_glasair.cpp" "L:\Dev\FluidX3D\src\setup.cpp" >nul
"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe" ^
    "L:\Dev\FluidX3D\FluidX3D.sln" /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
if errorlevel 1 (
    echo BUILD FAILED - see errors above.
) else (
    echo Build OK: L:\Dev\FluidX3D\bin\FluidX3D.exe is up to date.
)
pause
