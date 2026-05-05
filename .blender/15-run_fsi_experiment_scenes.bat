@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem FSI thesis experiment scenes -> cache -> particle .blend / splashsurf .blend
rem
rem This script follows the verified 01/02/03 split, but loops over the five
rem chapter-5 FSI scenes. It does not render PNGs by default.
rem ---------------------------------------------------------------------------

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

set "PY_SPLASHSURF_EXE=pysplashsurf"
set "CONFIG_PATH=.blender\config\fsi_experiment_scenes.json"
set "DEVICE=cuda:0"

set "CASE_KEYS=three_sphere_buoys box_wall_break moving_wall_float_box cloth_cascade cloth_payload_rain"
set "CACHE_ROOT=.blender\cache\fsi_experiment_scenes"
set "PARTICLE_RENDER_ROOT=.blender\renders\fsi_experiment_particles"
set "SURFACE_RENDER_ROOT=.blender\renders\fsi_experiment_surfaces"
set "METRICS_ROOT=.blender\renders\fsi_experiment_metrics"
set "PARTICLE_BLEND_ROOT=.blender\templates\fsi_experiment_particles"
set "SURFACE_BLEND_ROOT=.blender\templates\fsi_experiment_surfaces"

rem Turn EXPORT_CACHE on for the first run or whenever you changed simulation settings.
set "EXPORT_CACHE=0"
set "EXPORT_SPLASHSURF_PARTICLES=1"
set "RUN_SPLASHSURF=1"
set "SAVE_PARTICLE_BLEND=1"
set "SAVE_SURFACE_BLEND=1"
set "RENDER_IMAGES=0"
set "PLOT_METRICS=1"
set "OVERWRITE_OUTPUTS=1"
set "TEST_EXAMPLE_CONFIG=0"

set "SOURCE_START_FRAME=0"
set "SOURCE_END_FRAME=auto"
set "SOURCE_STRIDE=1"
set "AXIS_CONVERSION=newton-y-up-to-blender-z-up"

set "EXTRA_EXPORT_ARGS="
set "EXTRA_PARTICLE_RENDER_ARGS="
set "EXTRA_SURFACE_RENDER_ARGS="
set "EXTRA_METRIC_ARGS="

set "UV_CACHE_DIR=%TEMP%\uv-cache-fsi-experiment-scenes"
set "UV_TOOL_DIR=%TEMP%\uv-tools-fsi-experiment-scenes"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-fsi-experiment-scenes"

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

echo.
echo [FSI Experiment Scenes] Config: %CONFIG_PATH%
echo [FSI Experiment Scenes] Cases: %CASE_KEYS%
echo.

if "%EXPORT_CACHE%"=="1" (
    for %%C in (%CASE_KEYS%) do (
        echo.
        echo [1/6] Exporting cache for %%C
        "%UV_EXE%" run python .blender\scripts\export_fsi_example_cache.py ^
            --config "%CONFIG_PATH%" ^
            --output-root "%CACHE_ROOT%" ^
            --case-key %%C ^
            --device "%DEVICE%" ^
            %OVERWRITE_ARG% ^
            %TEST_ARG% ^
            %EXTRA_EXPORT_ARGS%
        if errorlevel 1 goto error
    )
) else (
    echo [1/6] Skipping cache export. Using existing %CACHE_ROOT%.
)

if "%SAVE_PARTICLE_BLEND%"=="1" (
    if not exist "%BLENDER_EXE%" (
        echo Blender not found: "%BLENDER_EXE%"
        goto error
    )
    for %%C in (%CASE_KEYS%) do (
        echo.
        echo [2/6] Building particle blend for %%C
        "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_example_blend.py -- ^
            --config "%CONFIG_PATH%" ^
            --cache-dir "%CACHE_ROOT%\%%C" ^
            --output-dir "%PARTICLE_RENDER_ROOT%\%%C" ^
            --mode particles ^
            --save-blend "%PARTICLE_BLEND_ROOT%\%%C.blend" ^
            %RENDER_ARG% ^
            %EXTRA_PARTICLE_RENDER_ARGS%
        if errorlevel 1 goto error
    )
) else (
    echo [2/6] Skipping particle blend generation.
)

if "%EXPORT_SPLASHSURF_PARTICLES%"=="1" (
    for %%C in (%CASE_KEYS%) do (
        echo.
        echo [3/6] Exporting splashsurf particle PLY for %%C
        "%UV_EXE%" run python .blender\scripts\export_splashsurf_particles.py ^
            --cache-dir "%CACHE_ROOT%\%%C" ^
            --output-dir "%CACHE_ROOT%\%%C\particles" ^
            --start-frame %SOURCE_START_FRAME% ^
            %SOURCE_END_ARG% ^
            --stride %SOURCE_STRIDE% ^
            --axis-conversion "%AXIS_CONVERSION%" ^
            %OVERWRITE_ARG%
        if errorlevel 1 goto error
    )
) else (
    echo [3/6] Skipping splashsurf particle PLY export.
)

if "%RUN_SPLASHSURF%"=="1" (
    for %%C in (%CASE_KEYS%) do (
        echo.
        echo [4/6] Running splashsurf for %%C
        "%UV_EXE%" run python .blender\scripts\run_splashsurf_reconstruct.py ^
            --particles-dir "%CACHE_ROOT%\%%C\particles" ^
            --output-dir "%CACHE_ROOT%\%%C\surface_obj" ^
            --pysplashsurf-exe "%PY_SPLASHSURF_EXE%" ^
            --mt-files off ^
            --mt-particles on ^
            %OVERWRITE_ARG%
        if errorlevel 1 goto error
    )
) else (
    echo [4/6] Skipping splashsurf reconstruction.
)

if "%SAVE_SURFACE_BLEND%"=="1" (
    if not exist "%BLENDER_EXE%" (
        echo Blender not found: "%BLENDER_EXE%"
        goto error
    )
    for %%C in (%CASE_KEYS%) do (
        echo.
        echo [5/6] Building surface blend for %%C
        "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_example_blend.py -- ^
            --config "%CONFIG_PATH%" ^
            --cache-dir "%CACHE_ROOT%\%%C" ^
            --surface-dir "%CACHE_ROOT%\%%C\surface_obj" ^
            --output-dir "%SURFACE_RENDER_ROOT%\%%C" ^
            --mode surface ^
            --save-blend "%SURFACE_BLEND_ROOT%\%%C.blend" ^
            %RENDER_ARG% ^
            %EXTRA_SURFACE_RENDER_ARGS%
        if errorlevel 1 goto error
    )
) else (
    echo [5/6] Skipping surface blend generation.
)

if "%PLOT_METRICS%"=="1" (
    echo.
    echo [6/6] Plotting metrics
    "%UV_EXE%" run python .blender\scripts\plot_fsi_example_metrics.py ^
        --cache-root "%CACHE_ROOT%" ^
        --output-dir "%METRICS_ROOT%" ^
        %EXTRA_METRIC_ARGS%
    if errorlevel 1 goto error
) else (
    echo [6/6] Skipping metrics.
)

echo.
echo Done.
echo Particle blends: %CD%\%PARTICLE_BLEND_ROOT%
echo Surface blends: %CD%\%SURFACE_BLEND_ROOT%
echo Metrics: %CD%\%METRICS_ROOT%
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:error
echo.
echo Failed. Check the error messages above.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
