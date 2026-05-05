@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Single FSI scene: box_wall_break -> cache -> particle .blend / mesh .blend
rem ---------------------------------------------------------------------------

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

set "PY_SPLASHSURF_EXE=pysplashsurf"
set "CONFIG_PATH=.blender\config\fsi_experiment_scenes.json"
set "DEVICE=cuda:0"
set "CASE_KEY=box_wall_break"
set "CASE_NAME=box_wall_break"

set "CACHE_ROOT=.blender\cache\fsi_experiment_scenes"
set "CACHE_DIR=%CACHE_ROOT%\%CASE_KEY%"
set "PARTICLE_RENDER_DIR=.blender\renders\fsi_experiment_particles\%CASE_NAME%"
set "SURFACE_RENDER_DIR=.blender\renders\fsi_experiment_surfaces\%CASE_NAME%"
set "METRICS_DIR=.blender\renders\fsi_experiment_metrics"
set "PARTICLE_BLEND_FILE=.blender\templates\fsi_experiment_particles\%CASE_NAME%.blend"
set "SURFACE_BLEND_FILE=.blender\templates\fsi_experiment_surfaces\%CASE_NAME%.blend"

rem Use auto for every cached frame, or comma-separated indices such as 0,60,120,180.
set "FRAME_INDICES=auto"

rem Turn EXPORT_CACHE on for the first run or whenever the simulation settings changed.
set "EXPORT_CACHE=1"
set "EXPORT_SPLASHSURF_PARTICLES=1"
set "RUN_SPLASHSURF=1"
set "SAVE_PARTICLE_BLEND=1"
set "SAVE_SURFACE_BLEND=1"
set "RENDER_IMAGES=0"
set "PLOT_METRICS=1"
set "OVERWRITE_OUTPUTS=1"
set "TEST_EXAMPLE_CONFIG=0"

rem Leave these empty to use .blender\config\fsi_experiment_scenes.json.
set "RECORD_FRAMES=300"
set "RENDER_FPS=30"

set "SOURCE_START_FRAME=0"
set "SOURCE_END_FRAME=auto"
set "SOURCE_STRIDE=1"
set "AXIS_CONVERSION=newton-y-up-to-blender-z-up"

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
set "MESH_SMOOTHING_WEIGHTS=1"
set "MESH_CLEANUP=1"
set "NORMALS=1"
set "MT_FILES=off"
set "MT_PARTICLES=on"
set "NUM_THREADS="

set "EXTRA_EXPORT_ARGS="
set "EXTRA_PARTICLE_RENDER_ARGS="
set "EXTRA_SURFACE_RENDER_ARGS="
set "EXTRA_METRIC_ARGS=--case-key %CASE_KEY%"

set "UV_CACHE_DIR=%TEMP%\uv-cache-fsi-experiment-%CASE_NAME%"
set "UV_TOOL_DIR=%TEMP%\uv-tools-fsi-experiment-%CASE_NAME%"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-fsi-experiment-%CASE_NAME%"

set "PAUSE_ON_EXIT=1"

cd /d "%~dp0.."

set "OVERWRITE_ARG="
if "%OVERWRITE_OUTPUTS%"=="1" set "OVERWRITE_ARG=--overwrite"
set "TEST_ARG="
if "%TEST_EXAMPLE_CONFIG%"=="1" set "TEST_ARG=--test-example-config"
set "RENDER_ARG="
if "%RENDER_IMAGES%"=="1" set "RENDER_ARG=--render"
set "SOURCE_END_ARG="
if /I not "%SOURCE_END_FRAME%"=="auto" set "SOURCE_END_ARG=--end-frame %SOURCE_END_FRAME%"
set "RECORD_FRAMES_ARG="
if not "%RECORD_FRAMES%"=="" set "RECORD_FRAMES_ARG=--record-frames %RECORD_FRAMES%"
set "EXPORT_RENDER_FPS_ARG="
if not "%RENDER_FPS%"=="" set "EXPORT_RENDER_FPS_ARG=--render-fps %RENDER_FPS%"
set "BLENDER_FPS_ARG="
if not "%RENDER_FPS%"=="" set "BLENDER_FPS_ARG=--fps %RENDER_FPS%"
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
set "MESH_SMOOTHING_WEIGHTS_ARG=--no-mesh-smoothing-weights"
if "%MESH_SMOOTHING_WEIGHTS%"=="1" set "MESH_SMOOTHING_WEIGHTS_ARG=--mesh-smoothing-weights"
set "MESH_CLEANUP_ARG=--no-mesh-cleanup"
if "%MESH_CLEANUP%"=="1" set "MESH_CLEANUP_ARG=--mesh-cleanup"
set "NORMALS_ARG=--no-normals"
if "%NORMALS%"=="1" set "NORMALS_ARG=--normals"
set "NUM_THREADS_ARG="
if not "%NUM_THREADS%"=="" set "NUM_THREADS_ARG=--num-threads %NUM_THREADS%"

echo.
echo [FSI Experiment] Case: %CASE_KEY%
echo [FSI Experiment] Cache: %CACHE_DIR%
echo [FSI Experiment] Particle blend: %PARTICLE_BLEND_FILE%
echo [FSI Experiment] Surface blend: %SURFACE_BLEND_FILE%
echo.

if "%EXPORT_CACHE%"=="1" (
    "%UV_EXE%" run python .blender\scripts\export_fsi_example_cache.py ^
        --config "%CONFIG_PATH%" ^
        --output-root "%CACHE_ROOT%" ^
        --case-key "%CASE_KEY%" ^
        --device "%DEVICE%" ^
        %RECORD_FRAMES_ARG% ^
        %EXPORT_RENDER_FPS_ARG% ^
        %OVERWRITE_ARG% ^
        %TEST_ARG% ^
        %EXTRA_EXPORT_ARGS%
    if errorlevel 1 goto error
) else (
    echo [1/6] Skipping cache export. Using existing cache.
)

if not exist "%CACHE_DIR%\metadata.json" (
    echo.
    echo Missing cache metadata:
    echo %CD%\%CACHE_DIR%\metadata.json
    echo Set EXPORT_CACHE=1 once to regenerate this cache with scene metadata.
    goto error
)

if "%SAVE_PARTICLE_BLEND%"=="1" (
    if not exist "%BLENDER_EXE%" (
        echo Blender not found: "%BLENDER_EXE%"
        goto error
    )
    "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_example_blend.py -- ^
        --config "%CONFIG_PATH%" ^
        --cache-dir "%CACHE_DIR%" ^
        --output-dir "%PARTICLE_RENDER_DIR%" ^
        --mode particles ^
        --frame-indices "%FRAME_INDICES%" ^
        %BLENDER_FPS_ARG% ^
        --save-blend "%PARTICLE_BLEND_FILE%" ^
        %RENDER_ARG% ^
        %EXTRA_PARTICLE_RENDER_ARGS%
    if errorlevel 1 goto error
) else (
    echo [2/6] Skipping particle blend generation.
)

if "%EXPORT_SPLASHSURF_PARTICLES%"=="1" (
    "%UV_EXE%" run python .blender\scripts\export_splashsurf_particles.py ^
        --cache-dir "%CACHE_DIR%" ^
        --output-dir "%CACHE_DIR%\particles" ^
        --start-frame %SOURCE_START_FRAME% ^
        %SOURCE_END_ARG% ^
        --stride %SOURCE_STRIDE% ^
        --axis-conversion "%AXIS_CONVERSION%" ^
        %OVERWRITE_ARG%
    if errorlevel 1 goto error
) else (
    echo [3/6] Skipping splashsurf particle PLY export.
)

if "%RUN_SPLASHSURF%"=="1" (
    "%UV_EXE%" run python .blender\scripts\run_splashsurf_reconstruct.py ^
        --particles-dir "%CACHE_DIR%\particles" ^
        --output-dir "%CACHE_DIR%\surface_obj" ^
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
) else (
    echo [4/6] Skipping splashsurf reconstruction.
)

if "%SAVE_SURFACE_BLEND%"=="1" (
    if not exist "%BLENDER_EXE%" (
        echo Blender not found: "%BLENDER_EXE%"
        goto error
    )
    "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_example_blend.py -- ^
        --config "%CONFIG_PATH%" ^
        --cache-dir "%CACHE_DIR%" ^
        --surface-dir "%CACHE_DIR%\surface_obj" ^
        --output-dir "%SURFACE_RENDER_DIR%" ^
        --mode surface ^
        --frame-indices "%FRAME_INDICES%" ^
        %BLENDER_FPS_ARG% ^
        --save-blend "%SURFACE_BLEND_FILE%" ^
        %RENDER_ARG% ^
        %EXTRA_SURFACE_RENDER_ARGS%
    if errorlevel 1 goto error
) else (
    echo [5/6] Skipping surface blend generation.
)

if "%PLOT_METRICS%"=="1" (
    "%UV_EXE%" run python .blender\scripts\plot_fsi_example_metrics.py ^
        --cache-root "%CACHE_ROOT%" ^
        --output-dir "%METRICS_DIR%" ^
        %EXTRA_METRIC_ARGS%
    if errorlevel 1 goto error
) else (
    echo [6/6] Skipping metrics.
)

echo.
echo Done.
echo Particle blend:
echo %CD%\%PARTICLE_BLEND_FILE%
echo Surface blend:
echo %CD%\%SURFACE_BLEND_FILE%
echo Metrics:
echo %CD%\%METRICS_DIR%
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:error
echo.
echo Failed. Check the error messages above.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
