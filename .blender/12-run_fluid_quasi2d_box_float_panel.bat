@echo off
setlocal EnableExtensions

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

set "UV_EXE=E:\ProgEnv\uv\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

set "CONFIG_PATH=.blender\config\fluid_quasi2d_box_float_panel.json"
set "DEVICE=cuda:0"
set "RECORD_FRAMES=180"
set "EXPORT_CACHE=1"
set "RENDER_IMAGES=0"
set "SAVE_BLEND=1"
set "OVERWRITE_CACHE=1"
set "EXTRA_EXPORT_ARGS="
set "EXTRA_RENDER_ARGS="

set "UV_CACHE_DIR=%TEMP%\uv-cache-fluid-q2d-box-float"
set "UV_TOOL_DIR=%TEMP%\uv-tools-fluid-q2d-box-float"
set "WARP_CACHE_PATH=%TEMP%\warp-cache-fluid-q2d-box-float"

cd /d "%~dp0.."

set "OVERWRITE_ARG="
if "%OVERWRITE_CACHE%"=="1" set "OVERWRITE_ARG=--overwrite"

echo.
echo [Fluid Compare] Config: %CONFIG_PATH%
echo [Fluid Compare] Record frames: %RECORD_FRAMES%
echo.

if "%EXPORT_CACHE%"=="1" (
    "%UV_EXE%" run python .blender\scripts\export_fluid_compare_cache.py ^
        --config "%CONFIG_PATH%" ^
        --device "%DEVICE%" ^
        --record-frames %RECORD_FRAMES% ^
        %OVERWRITE_ARG% ^
        %EXTRA_EXPORT_ARGS%
    if errorlevel 1 goto error
)

if not exist "%BLENDER_EXE%" (
    echo Blender not found: "%BLENDER_EXE%"
    goto error
)

set "RENDER_ARG="
if "%RENDER_IMAGES%"=="1" set "RENDER_ARG=--render"
set "SAVE_BLEND_ARG="
if "%SAVE_BLEND%"=="0" set "SAVE_BLEND_ARG=--skip-save-blend"

"%BLENDER_EXE%" --background --python .blender\scripts\render_fluid_compare_panel.py -- ^
    --config "%CONFIG_PATH%" ^
    --frames %RECORD_FRAMES% ^
    %SAVE_BLEND_ARG% ^
    %RENDER_ARG% ^
    %EXTRA_RENDER_ARGS%
if errorlevel 1 goto error

echo.
echo Done.
pause
exit /b 0

:error
echo.
echo Failed.
pause
exit /b 1
