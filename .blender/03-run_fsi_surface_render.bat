@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Build a Blender scene from splashsurf OBJ surface sequence + Newton rigid cache
rem ---------------------------------------------------------------------------

set "BLENDER_DIR=E:\Softwares\3D Animation\Blender Foundation\Blender 4.5"
set "BLENDER_EXE=%BLENDER_DIR%\blender.exe"

set "CACHE_DIR=.blender\cache\fsi_three_sphere_buoys"
set "SURFACE_DIR=%CACHE_DIR%\surface_obj"
set "PARTICLES_DIR=%CACHE_DIR%\particles"
set "RENDER_DIR=.blender\renders\fsi_three_sphere_buoys_surface"
set "BLEND_FILE=.blender\templates\fsi_three_sphere_buoys_surface.blend"

set "START_FRAME=0"
set "STRIDE=1"
set "FRAMES=auto"

set "RENDER_IMAGES=0"
set "SAVE_BLEND=1"
set "PAUSE_ON_EXIT=1"

set "EXTRA_RENDER_ARGS="

cd /d "%~dp0.."

if not exist "%BLENDER_EXE%" (
    echo.
    echo ERROR: Blender executable was not found:
    echo "%BLENDER_EXE%"
    goto error
)

set "FRAMES_ARG=--frames %FRAMES%"
if /I "%FRAMES%"=="auto" set "FRAMES_ARG="

set "RENDER_ARG="
if "%RENDER_IMAGES%"=="1" set "RENDER_ARG=--render"

echo.
echo [FSI Surface] Cache: %CACHE_DIR%
echo [FSI Surface] Surface: %SURFACE_DIR%
echo [FSI Surface] Render: %RENDER_DIR%
echo.

if "%SAVE_BLEND%"=="1" (
    "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_surface_sequence.py -- ^
        --cache-dir "%CACHE_DIR%" ^
        --surface-dir "%SURFACE_DIR%" ^
        --particles-dir "%PARTICLES_DIR%" ^
        --output-dir "%RENDER_DIR%" ^
        --start-frame %START_FRAME% ^
        --stride %STRIDE% ^
        %FRAMES_ARG% ^
        --save-blend "%BLEND_FILE%" ^
        %RENDER_ARG% ^
        %EXTRA_RENDER_ARGS%
) else (
    "%BLENDER_EXE%" --background --python .blender\scripts\render_fsi_surface_sequence.py -- ^
        --cache-dir "%CACHE_DIR%" ^
        --surface-dir "%SURFACE_DIR%" ^
        --particles-dir "%PARTICLES_DIR%" ^
        --output-dir "%RENDER_DIR%" ^
        --start-frame %START_FRAME% ^
        --stride %STRIDE% ^
        %FRAMES_ARG% ^
        %RENDER_ARG% ^
        %EXTRA_RENDER_ARGS%
)
if errorlevel 1 goto error

echo.
echo Done.
echo Blend file:
echo %CD%\%BLEND_FILE%
echo Render output:
echo %CD%\%RENDER_DIR%
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:error
echo.
echo Failed. Check the error messages above.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
