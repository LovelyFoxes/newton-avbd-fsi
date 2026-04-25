@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Newton cache -> splashsurf particle PLY -> splashsurf surface mesh sequence
rem
rem Edit the variable block below for your current case.
rem ---------------------------------------------------------------------------

set "PY_SPLASHSURF_EXE=pysplashsurf"

rem If uv is not on PATH, keep this full path. Otherwise you may set UV_EXE=uv.
set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

rem Input/output folders.
set "CACHE_DIR=.blender\cache\fsi_three_sphere_buoys"
set "PARTICLES_DIR=%CACHE_DIR%\particles"
set "SURFACE_DIR=%CACHE_DIR%\surface_obj"

rem Export selection from cached Newton frames.
set "SOURCE_START_FRAME=0"
set "SOURCE_END_FRAME=auto"
set "SOURCE_STRIDE=1"
set "AXIS_CONVERSION=newton-y-up-to-blender-z-up"

rem Splashsurf execution toggles.
set "EXPORT_PARTICLES=1"
set "RUN_SPLASHSURF=1"
set "OVERWRITE_OUTPUTS=1"

rem Override automatic reconstruct parameters only if needed.
rem Leave these empty to use splashsurf_particles_metadata.json recommendations.
set "PARTICLE_RADIUS_OVERRIDE="
set "REST_DENSITY_OVERRIDE="
set "SMOOTHING_LENGTH_OVERRIDE="
set "CUBE_SIZE_OVERRIDE="
set "SURFACE_THRESHOLD_OVERRIDE="
set "START_INDEX_OVERRIDE="
set "END_INDEX_OVERRIDE="
set "MESH_SMOOTHING_ITERS_OVERRIDE="
set "NORMALS_SMOOTHING_ITERS_OVERRIDE="

rem Binary switches for reconstruct helper.
set "MESH_SMOOTHING_WEIGHTS=1"
set "MESH_CLEANUP=1"
set "NORMALS=1"
set "MT_FILES=off"
set "MT_PARTICLES=on"
set "NUM_THREADS="

rem Pause before closing the terminal window.
set "PAUSE_ON_EXIT=1"

rem ---------------------------------------------------------------------------
rem Implementation below. Usually you do not need to edit this part.
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

set "UV_CACHE_DIR=%TEMP%\uv-cache-fsi-render"
set "UV_TOOL_DIR=%TEMP%\uv-tools-fsi-render"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-fsi-render"

set "SOURCE_END_ARG="
if /I not "%SOURCE_END_FRAME%"=="auto" set "SOURCE_END_ARG=--end-frame %SOURCE_END_FRAME%"

set "OVERWRITE_ARG="
if "%OVERWRITE_OUTPUTS%"=="1" set "OVERWRITE_ARG=--overwrite"

set "MESH_SMOOTHING_WEIGHTS_ARG=--no-mesh-smoothing-weights"
if "%MESH_SMOOTHING_WEIGHTS%"=="1" set "MESH_SMOOTHING_WEIGHTS_ARG=--mesh-smoothing-weights"

set "MESH_CLEANUP_ARG=--no-mesh-cleanup"
if "%MESH_CLEANUP%"=="1" set "MESH_CLEANUP_ARG=--mesh-cleanup"

set "NORMALS_ARG=--no-normals"
if "%NORMALS%"=="1" set "NORMALS_ARG=--normals"

set "PARTICLE_RADIUS_ARG="
if not "%PARTICLE_RADIUS_OVERRIDE%"=="" set "PARTICLE_RADIUS_ARG=--particle-radius %PARTICLE_RADIUS_OVERRIDE%"

set "REST_DENSITY_ARG="
if not "%REST_DENSITY_OVERRIDE%"=="" set "REST_DENSITY_ARG=--rest-density %REST_DENSITY_OVERRIDE%"

set "SMOOTHING_LENGTH_ARG="
if not "%SMOOTHING_LENGTH_OVERRIDE%"=="" set "SMOOTHING_LENGTH_ARG=--smoothing-length %SMOOTHING_LENGTH_OVERRIDE%"

set "CUBE_SIZE_ARG="
if not "%CUBE_SIZE_OVERRIDE%"=="" set "CUBE_SIZE_ARG=--cube-size %CUBE_SIZE_OVERRIDE%"

set "SURFACE_THRESHOLD_ARG="
if not "%SURFACE_THRESHOLD_OVERRIDE%"=="" set "SURFACE_THRESHOLD_ARG=--surface-threshold %SURFACE_THRESHOLD_OVERRIDE%"

set "START_INDEX_ARG="
if not "%START_INDEX_OVERRIDE%"=="" set "START_INDEX_ARG=--start-index %START_INDEX_OVERRIDE%"

set "END_INDEX_ARG="
if not "%END_INDEX_OVERRIDE%"=="" set "END_INDEX_ARG=--end-index %END_INDEX_OVERRIDE%"

set "MESH_SMOOTHING_ITERS_ARG="
if not "%MESH_SMOOTHING_ITERS_OVERRIDE%"=="" set "MESH_SMOOTHING_ITERS_ARG=--mesh-smoothing-iters %MESH_SMOOTHING_ITERS_OVERRIDE%"

set "NORMALS_SMOOTHING_ITERS_ARG="
if not "%NORMALS_SMOOTHING_ITERS_OVERRIDE%"=="" set "NORMALS_SMOOTHING_ITERS_ARG=--normals-smoothing-iters %NORMALS_SMOOTHING_ITERS_OVERRIDE%"

set "NUM_THREADS_ARG="
if not "%NUM_THREADS%"=="" set "NUM_THREADS_ARG=--num-threads %NUM_THREADS%"

echo.
echo [Splashsurf] Repository: %CD%
echo [Splashsurf] Cache: %CACHE_DIR%
echo [Splashsurf] Particles: %PARTICLES_DIR%
echo [Splashsurf] Surface: %SURFACE_DIR%
echo.

if "%EXPORT_PARTICLES%"=="1" (
    echo [1/2] Exporting splashsurf particle PLY sequence...
    "%UV_EXE%" run python .blender\scripts\export_splashsurf_particles.py ^
        --cache-dir "%CACHE_DIR%" ^
        --output-dir "%PARTICLES_DIR%" ^
        --start-frame %SOURCE_START_FRAME% ^
        %SOURCE_END_ARG% ^
        --stride %SOURCE_STRIDE% ^
        --axis-conversion "%AXIS_CONVERSION%" ^
        %OVERWRITE_ARG%
    if errorlevel 1 goto error
) else (
    echo [1/2] Skipping particle export. Using existing PLY sequence.
)

if "%RUN_SPLASHSURF%"=="0" (
    echo [2/2] Skipping splashsurf reconstruction.
    goto success
)

echo [2/2] Running splashsurf reconstruct...
"%UV_EXE%" run python .blender\scripts\run_splashsurf_reconstruct.py ^
    --particles-dir "%PARTICLES_DIR%" ^
    --output-dir "%SURFACE_DIR%" ^
    --pysplashsurf-exe "%PY_SPLASHSURF_EXE%" ^
    %PARTICLE_RADIUS_ARG% ^
    %REST_DENSITY_ARG% ^
    %SMOOTHING_LENGTH_ARG% ^
    %CUBE_SIZE_ARG% ^
    %SURFACE_THRESHOLD_ARG% ^
    %START_INDEX_ARG% ^
    %END_INDEX_ARG% ^
    %MESH_SMOOTHING_ITERS_ARG% ^
    %NORMALS_SMOOTHING_ITERS_ARG% ^
    %MESH_SMOOTHING_WEIGHTS_ARG% ^
    %MESH_CLEANUP_ARG% ^
    %NORMALS_ARG% ^
    --mt-files %MT_FILES% ^
    --mt-particles %MT_PARTICLES% ^
    %NUM_THREADS_ARG% ^
    %OVERWRITE_ARG%
if errorlevel 1 goto error

:success
echo.
echo Done.
echo Particle metadata:
echo %CD%\%PARTICLES_DIR%\splashsurf_particles_metadata.json
echo Surface directory:
echo %CD%\%SURFACE_DIR%
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:error
echo.
echo Failed. Check the error messages above.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
