@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Newton AVBD/IPBF FSI -> cache -> Blender preview render
rem
rem Edit the variables in this block to switch scenes, coupling settings,
rem frame counts, output folders, or preview quality.
rem ---------------------------------------------------------------------------

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

rem If uv is not on PATH, keep this full path. Otherwise you may set UV_EXE=uv.
set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

rem Scene / method settings.
set "SCENE=three_sphere_buoys"
set "DEVICE=cuda:0"
set "COUPLING_MODE=interlinked"
set "HYDROSTATIC_VOLUME_MODE=dynamic-shape-surface-quadrature"
set "PRESSURE_REACTION=2.0"
set "PROJECTION_REACTION=0.0"
set "VELOCITY_REACTION=0.0"
set "INCLUDE_STATIC_BOUNDARY_SAMPLES=0"

rem Simulation/export settings.
set "FPS=60"
set "SIM_SUBSTEPS=6"
set "IPBF_ITERATIONS=2"
set "RIGID_ITERATIONS=2"
set "COUPLING_ITERATIONS=3"
set "START_TIME_SECONDS=0.0"
set "DURATION_SECONDS=10.0"
set "VIDEO_FPS=30"
set "RECORD_INITIAL_FRAME=1"
set "OVERWRITE_CACHE=1"
set "EXPORT_CACHE=1"

rem Output folders.
set "CACHE_DIR=.blender\cache\fsi_three_sphere_buoys"
set "RENDER_DIR=.blender\renders\fsi_three_sphere_buoys_preview"
set "BLEND_FILE=.blender\templates\fsi_three_sphere_buoys_preview.blend"

rem Blender preview settings. Use auto to import every cached frame.
set "RENDER_FRAMES=auto"
set "WATER_MAX_PARTICLES=2500"
set "WATER_RADIUS_SCALE=0.55"
set "RENDER_IMAGES=0"
set "SAVE_BLEND=1"

rem Add custom exporter/render args here without editing the command body.
rem Examples:
rem set "EXTRA_EXPORT_ARGS=--pool-dims 32 18 24 --sphere-radius 0.06"
rem set "EXTRA_RENDER_ARGS=--start-frame 0 --stride 1 --axis-conversion none"
set "EXTRA_EXPORT_ARGS="
set "EXTRA_RENDER_ARGS="

rem Keep tool caches local to TEMP so this script does not depend on global cache permissions.
set "UV_CACHE_DIR=%TEMP%\uv-cache-fsi-render"
set "UV_TOOL_DIR=%TEMP%\uv-tools-fsi-render"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-fsi-render"

rem Pause before closing the terminal window. Set to 0 if launching from an existing shell.
set "PAUSE_ON_EXIT=1"

rem ---------------------------------------------------------------------------
rem Implementation below. Usually you do not need to edit this part.
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

set "OVERWRITE_ARG="
if "%OVERWRITE_CACHE%"=="1" set "OVERWRITE_ARG=--overwrite"

set "STATIC_BOUNDARY_ARG=--no-include-static-boundary-samples"
if "%INCLUDE_STATIC_BOUNDARY_SAMPLES%"=="1" set "STATIC_BOUNDARY_ARG=--include-static-boundary-samples"

set "RECORD_INITIAL_ARG=--no-record-initial-frame"
if "%RECORD_INITIAL_FRAME%"=="1" set "RECORD_INITIAL_ARG=--record-initial-frame"

set "RENDER_FRAMES_ARG=--frames %RENDER_FRAMES%"
if /I "%RENDER_FRAMES%"=="auto" set "RENDER_FRAMES_ARG="

echo.
echo [Newton FSI] Repository: %CD%
echo [Newton FSI] Scene: %SCENE%
echo [Newton FSI] Time range: start %START_TIME_SECONDS%s, duration %DURATION_SECONDS%s, video FPS %VIDEO_FPS%
echo [Newton FSI] Cache: %CACHE_DIR%
echo [Newton FSI] Render: %RENDER_DIR%
echo.

if "%EXPORT_CACHE%"=="1" (
    echo [1/2] Exporting Newton cache...
    "%UV_EXE%" run python .blender\scripts\export_fsi_cache.py ^
        --scene "%SCENE%" ^
        --device "%DEVICE%" ^
        --output-dir "%CACHE_DIR%" ^
        --fps %FPS% ^
        --sim-substeps %SIM_SUBSTEPS% ^
        --ipbf-iterations %IPBF_ITERATIONS% ^
        --rigid-iterations %RIGID_ITERATIONS% ^
        --coupling-iterations %COUPLING_ITERATIONS% ^
        --coupling-mode "%COUPLING_MODE%" ^
        --hydrostatic-volume-mode "%HYDROSTATIC_VOLUME_MODE%" ^
        --pressure-reaction-relaxation %PRESSURE_REACTION% ^
        --projection-reaction-relaxation %PROJECTION_REACTION% ^
        --velocity-reaction-relaxation %VELOCITY_REACTION% ^
        --start-time %START_TIME_SECONDS% ^
        --duration %DURATION_SECONDS% ^
        --video-fps %VIDEO_FPS% ^
        %RECORD_INITIAL_ARG% ^
        %STATIC_BOUNDARY_ARG% ^
        %OVERWRITE_ARG% ^
        %EXTRA_EXPORT_ARGS%
    if errorlevel 1 goto error
) else (
    echo [1/2] Skipping Newton cache export. Using existing cache.
)

if not exist "%BLENDER_EXE%" (
    echo.
    echo ERROR: Blender executable was not found:
    echo "%BLENDER_EXE%"
    echo Edit BLENDER_DIR or BLENDER_EXE at the top of this bat file.
    goto error
)

if "%RENDER_IMAGES%"=="1" goto run_blender
if "%SAVE_BLEND%"=="1" goto run_blender

echo [2/2] Skipping Blender. RENDER_IMAGES=0 and SAVE_BLEND=0.
goto success

:run_blender
set "RENDER_ARG="
if "%RENDER_IMAGES%"=="1" set "RENDER_ARG=--render"

echo [2/2] Running Blender preview script...
if "%SAVE_BLEND%"=="1" (
    "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_cache.py -- ^
        --cache-dir "%CACHE_DIR%" ^
        --output-dir "%RENDER_DIR%" ^
        %RENDER_FRAMES_ARG% ^
        --water-max-particles %WATER_MAX_PARTICLES% ^
        --water-radius-scale %WATER_RADIUS_SCALE% ^
        --save-blend "%BLEND_FILE%" ^
        %RENDER_ARG% ^
        %EXTRA_RENDER_ARGS%
) else (
    "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_cache.py -- ^
        --cache-dir "%CACHE_DIR%" ^
        --output-dir "%RENDER_DIR%" ^
        %RENDER_FRAMES_ARG% ^
        --water-max-particles %WATER_MAX_PARTICLES% ^
        --water-radius-scale %WATER_RADIUS_SCALE% ^
        %RENDER_ARG% ^
        %EXTRA_RENDER_ARGS%
)
if errorlevel 1 goto error

:success
echo.
echo Done.
echo Cache metadata:
echo %CD%\%CACHE_DIR%\metadata.json
echo Render output:
echo %CD%\%RENDER_DIR%
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:error
echo.
echo Failed. Check the error messages above.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
