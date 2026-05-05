@echo off
setlocal EnableExtensions

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

set "CONFIG_PATH=.blender\config\ipbf_particle_examples.json"
set "DEVICE=cuda:0"
set "CACHE_ROOT=.blender\cache\ipbf_particle_examples"
set "RENDER_DIR=.blender\renders\ipbf_particle_examples"
set "METRICS_DIR=.blender\renders\ipbf_particle_metrics"
set "BLEND_DIR=.blender\templates\ipbf_particle_examples"
set "EXPORT_CACHE=0"
set "SAVE_BLEND=1"
set "RENDER_IMAGES=0"
set "PLOT_METRICS=1"
set "OVERWRITE_CACHE=1"
set "TEST_EXAMPLE_CONFIG=0"
set "EXTRA_EXPORT_ARGS="
set "EXTRA_RENDER_ARGS="
set "EXTRA_METRIC_ARGS="

set "UV_CACHE_DIR=%TEMP%\uv-cache-ipbf-particle-examples"
set "UV_TOOL_DIR=%TEMP%\uv-tools-ipbf-particle-examples"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-ipbf-particle-examples"

cd /d "%~dp0.."

set "OVERWRITE_ARG="
if "%OVERWRITE_CACHE%"=="1" set "OVERWRITE_ARG=--overwrite"
set "TEST_ARG="
if "%TEST_EXAMPLE_CONFIG%"=="1" set "TEST_ARG=--test-example-config"

echo.
echo [IPBF Particles] Config: %CONFIG_PATH%
echo.

if "%EXPORT_CACHE%"=="1" (
    "%UV_EXE%" run python .blender\scripts\export_ipbf_particle_examples.py ^
        --config "%CONFIG_PATH%" ^
        --output-root "%CACHE_ROOT%" ^
        --device "%DEVICE%" ^
        %OVERWRITE_ARG% ^
        %TEST_ARG% ^
        %EXTRA_EXPORT_ARGS%
    if errorlevel 1 goto error
)

if "%SAVE_BLEND%"=="1" goto run_blender
if "%RENDER_IMAGES%"=="1" goto run_blender
goto skip_blender

:run_blender
    if not exist "%BLENDER_EXE%" (
        echo Blender not found: "%BLENDER_EXE%"
        goto error
    )
    set "RENDER_ARG="
    if "%RENDER_IMAGES%"=="1" set "RENDER_ARG=--render"
    if "%SAVE_BLEND%"=="1" (
        "%BLENDER_EXE%" --background --python .blender\scripts\render_ipbf_particle_frames.py -- ^
            --config "%CONFIG_PATH%" ^
            --cache-root "%CACHE_ROOT%" ^
            --output-dir "%RENDER_DIR%" ^
            --save-blend "%BLEND_DIR%" ^
            %RENDER_ARG% ^
            %EXTRA_RENDER_ARGS%
    ) else (
        "%BLENDER_EXE%" --background --python .blender\scripts\render_ipbf_particle_frames.py -- ^
            --config "%CONFIG_PATH%" ^
            --cache-root "%CACHE_ROOT%" ^
            --output-dir "%RENDER_DIR%" ^
            %RENDER_ARG% ^
            %EXTRA_RENDER_ARGS%
    )
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
pause
exit /b 0

:error
echo.
echo Failed.
pause
exit /b 1
