@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Single IPBF double-dam-break scene -> cache -> particle .blend / stills
rem ---------------------------------------------------------------------------

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

set "CONFIG_PATH=.blender\config\ipbf_particle_examples.json"
set "DEVICE=cuda:0"
set "CASE_KEY=ipbf_double_dam_break"
set "CASE_NAME=ipbf_double_dam_break"

set "CACHE_ROOT=.blender\cache\ipbf_particle_examples"
set "CACHE_DIR=%CACHE_ROOT%\%CASE_KEY%"
set "RENDER_DIR=.blender\renders\ipbf_particle_examples\%CASE_NAME%"
set "METRICS_DIR=.blender\renders\ipbf_particle_metrics"
set "BLEND_FILE=.blender\templates\ipbf_particle_examples\%CASE_NAME%.blend"

rem Use auto for every cached frame, or comma-separated indices such as 0,45,90,150.
set "FRAME_INDICES=auto"

rem Turn EXPORT_CACHE on only when the simulation cache needs to be regenerated.
set "EXPORT_CACHE=0"
set "SAVE_BLEND=1"
set "RENDER_IMAGES=0"
set "PLOT_METRICS=1"
set "OVERWRITE_CACHE=1"
set "TEST_EXAMPLE_CONFIG=0"

set "EXTRA_EXPORT_ARGS="
set "EXTRA_RENDER_ARGS="
set "EXTRA_METRIC_ARGS=--case-key %CASE_KEY%"

set "UV_CACHE_DIR=%TEMP%\uv-cache-ipbf-particle-%CASE_NAME%"
set "UV_TOOL_DIR=%TEMP%\uv-tools-ipbf-particle-%CASE_NAME%"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-ipbf-particle-%CASE_NAME%"

set "PAUSE_ON_EXIT=1"

cd /d "%~dp0.."

set "OVERWRITE_ARG="
if "%OVERWRITE_CACHE%"=="1" set "OVERWRITE_ARG=--overwrite"
set "TEST_ARG="
if "%TEST_EXAMPLE_CONFIG%"=="1" set "TEST_ARG=--test-example-config"
set "RENDER_ARG="
if "%RENDER_IMAGES%"=="1" set "RENDER_ARG=--render"

echo.
echo [IPBF Particles] Case: %CASE_KEY%
echo [IPBF Particles] Cache: %CACHE_DIR%
echo [IPBF Particles] Blend: %BLEND_FILE%
echo.

if "%EXPORT_CACHE%"=="1" (
    "%UV_EXE%" run python .blender\scripts\export_ipbf_particle_examples.py ^
        --config "%CONFIG_PATH%" ^
        --output-root "%CACHE_ROOT%" ^
        --case-key "%CASE_KEY%" ^
        --device "%DEVICE%" ^
        %OVERWRITE_ARG% ^
        %TEST_ARG% ^
        %EXTRA_EXPORT_ARGS%
    if errorlevel 1 goto error
) else (
    echo [1/3] Skipping cache export. Using existing cache.
)

if "%SAVE_BLEND%"=="1" goto run_blender
if "%RENDER_IMAGES%"=="1" goto run_blender
goto skip_blender

:run_blender
if not exist "%BLENDER_EXE%" (
    echo Blender not found: "%BLENDER_EXE%"
    goto error
)
"%BLENDER_EXE%" --background --python .blender\scripts\render_ipbf_particle_frames.py -- ^
    --config "%CONFIG_PATH%" ^
    --cache-root "%CACHE_ROOT%" ^
    --output-dir "%RENDER_DIR%" ^
    --case-key "%CASE_KEY%" ^
    --frame-indices "%FRAME_INDICES%" ^
    --save-blend "%BLEND_FILE%" ^
    %RENDER_ARG% ^
    %EXTRA_RENDER_ARGS%
if errorlevel 1 goto error

:skip_blender

if "%PLOT_METRICS%"=="1" (
    "%UV_EXE%" run python .blender\scripts\plot_ipbf_particle_metrics.py ^
        --cache-root "%CACHE_ROOT%" ^
        --output-dir "%METRICS_DIR%" ^
        %EXTRA_METRIC_ARGS%
    if errorlevel 1 goto error
)

echo.
echo Done.
echo Blend file:
echo %CD%\%BLEND_FILE%
echo Metrics:
echo %CD%\%METRICS_DIR%
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:error
echo.
echo Failed. Check the error messages above.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
