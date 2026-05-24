@echo off
setlocal

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=v3"

set "CONFIG=configs/default.yaml"
if not "%~2"=="" set "CONFIG=%~2"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo [Aurora] mode=%MODE%
echo [Aurora] config=%CONFIG%
echo [Aurora] python=%PYTHON%

if /I "%MODE%"=="prepare" goto prepare
if /I "%MODE%"=="prepare-refresh" goto prepare_refresh
if /I "%MODE%"=="v3" goto v3
if /I "%MODE%"=="v3-refresh" goto v3_refresh
if /I "%MODE%"=="v4" goto v4
if /I "%MODE%"=="v4-refresh" goto v4_refresh
if /I "%MODE%"=="test" goto test
goto usage

:prepare
"%PYTHON%" scripts\prepare_data.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:prepare_refresh
"%PYTHON%" scripts\prepare_data.py --config "%CONFIG%" --refresh
if errorlevel 1 goto fail
goto done

:v3
"%PYTHON%" scripts\train_v3_robust.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:v3_refresh
"%PYTHON%" scripts\train_v3_robust.py --config "%CONFIG%" --refresh
if errorlevel 1 goto fail
goto done

:v4
"%PYTHON%" scripts\train_v4_migration_compare.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:v4_refresh
"%PYTHON%" scripts\train_v4_migration_compare.py --config "%CONFIG%" --refresh
if errorlevel 1 goto fail
goto done

:test
"%PYTHON%" -m pytest -q
if errorlevel 1 goto fail
goto done

:usage
echo.
echo Usage: run_aurora.bat [mode] [config_path]
echo.
echo Modes:
echo   prepare         prepare data
echo   prepare-refresh prepare data with --refresh
echo   v3              rolling validation + threshold tuning + error analysis
echo   v3-refresh      same as v3 with data refresh
echo   v4              SQL feature migration compare (pandas vs sql)
echo   v4-refresh      same as v4 with data refresh
echo   test            run pytest
exit /b 1

:fail
echo [Aurora] failed in mode=%MODE%
exit /b 1

:done
echo [Aurora] completed mode=%MODE%
exit /b 0
