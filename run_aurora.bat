@echo off
setlocal

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

set "CONFIG=configs/default.yaml"
if not "%~2"=="" set "CONFIG=%~2"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo [Aurora] mode=%MODE%
echo [Aurora] config=%CONFIG%
echo [Aurora] python=%PYTHON%

if /I "%MODE%"=="full" goto full
if /I "%MODE%"=="full-refresh" goto full_refresh
if /I "%MODE%"=="prepare" goto prepare
if /I "%MODE%"=="prepare-refresh" goto prepare_refresh
if /I "%MODE%"=="train" goto train
if /I "%MODE%"=="predict" goto predict
if /I "%MODE%"=="metrics" goto metrics
if /I "%MODE%"=="v2" goto v2
if /I "%MODE%"=="v2-refresh" goto v2_refresh
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

:train
"%PYTHON%" scripts\train_lstm.py --config "%CONFIG%"
if errorlevel 1 goto fail
"%PYTHON%" scripts\show_metrics.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:predict
"%PYTHON%" scripts\predict.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:metrics
"%PYTHON%" scripts\show_metrics.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:v2
"%PYTHON%" scripts\train_v2_compare.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:v2_refresh
"%PYTHON%" scripts\train_v2_compare.py --config "%CONFIG%" --refresh
if errorlevel 1 goto fail
goto done

:test
"%PYTHON%" -m pytest -q
if errorlevel 1 goto fail
goto done

:full
"%PYTHON%" scripts\prepare_data.py --config "%CONFIG%"
if errorlevel 1 goto fail
"%PYTHON%" scripts\train_lstm.py --config "%CONFIG%"
if errorlevel 1 goto fail
"%PYTHON%" scripts\predict.py --config "%CONFIG%"
if errorlevel 1 goto fail
"%PYTHON%" scripts\show_metrics.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:full_refresh
"%PYTHON%" scripts\prepare_data.py --config "%CONFIG%" --refresh
if errorlevel 1 goto fail
"%PYTHON%" scripts\train_lstm.py --config "%CONFIG%"
if errorlevel 1 goto fail
"%PYTHON%" scripts\predict.py --config "%CONFIG%"
if errorlevel 1 goto fail
"%PYTHON%" scripts\show_metrics.py --config "%CONFIG%"
if errorlevel 1 goto fail
goto done

:usage
echo.
echo Usage: run_aurora.bat [mode] [config_path]
echo.
echo Modes:
echo   full            prepare + train + predict
echo   full-refresh    prepare --refresh + train + predict
echo   prepare         prepare data
echo   prepare-refresh prepare data with --refresh
echo   train           train and evaluate
echo   predict         run latest-window inference
echo   metrics         print metrics/predictions summary
echo   v2              train LSTM + LogReg + MLP and make charts
echo   v2-refresh      same as v2 with data refresh
echo   test            run pytest
exit /b 1

:fail
echo [Aurora] failed in mode=%MODE%
exit /b 1

:done
echo [Aurora] completed mode=%MODE%
exit /b 0
