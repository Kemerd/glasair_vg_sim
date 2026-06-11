@echo off
rem ---------------------------------------------------------------------------
rem  film.bat [aoa_deg] [re] [seconds_of_airflow] [video_seconds] -- run the
rem  tunnel for a fixed amount of PHYSICAL airflow time while exporting a
rem  360-degree orbit as 60 fps PNG frames (YouTube-ready), then auto-close.
rem
rem  Examples:
rem    film.bat                 (14 deg, stall Re, 1.2 s of air, 10 s clip)
rem    film.bat 16 2.2e6 1.5 12
rem  Frames land in gpu\fluidx3d\results\frames\ -- assemble with:
rem    ffmpeg -framerate 60 -i image-%%06d.png -c:v libx264 -pix_fmt yuv420p wing.mp4
rem ---------------------------------------------------------------------------
setlocal
set BASE=%~dp0tunnel_config.txt
set RUN=%~dp0tunnel_run.txt
copy /y "%BASE%" "%RUN%" > nul
if "%~1"=="" (echo aoa_deg = 14>> "%RUN%") else (echo aoa_deg = %~1>> "%RUN%")
if "%~2"=="" (echo re = 2.2e6>> "%RUN%") else (echo re = %~2>> "%RUN%")
if "%~3"=="" (echo t_end_si = 1.2>> "%RUN%") else (echo t_end_si = %~3>> "%RUN%")
if "%~4"=="" (echo video_s = 10>> "%RUN%") else (echo video_s = %~4>> "%RUN%")
rem Film standard (owner spec): INFINITE wing -- span-periodic domain scaled
rem so the STL spans wall-to-wall and the tips are never simulated. Same
rem machinery as the science slice mode, fed the full 1.5 m wing.
echo mode = slice>> "%RUN%"
echo span_m = 1.5>> "%RUN%"
if not exist "%~dp0results\frames" mkdir "%~dp0results\frames"
start "" /D "L:\Dev\FluidX3D" "L:\Dev\FluidX3D\bin\FluidX3D.exe"
echo Filming run launched: auto-closes when the airflow time is up.
endlocal
