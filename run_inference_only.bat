@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM INFERENCIA SOLO (sin entrenamiento)
REM Asume modelos ya entrenados en models\
REM Uso: run_inference_only.bat [--tag TAG]
REM ============================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"
set "TAG=infer1"

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--tag" (set "TAG=%~2" & shift & shift & goto parse_args)
shift
goto parse_args
:end_parse

if exist "%ROOT_DIR%.venv\Scripts\activate.bat" call "%ROOT_DIR%.venv\Scripts\activate.bat"

set "PY_EXE=python"
%PY_EXE% --version >nul 2>&1
if errorlevel 1 (echo [ERROR] Python no encontrado. & pause & exit /b 1)
python -c "import sys; print('PYTHON:', sys.executable)"

set "PYTHONPATH=%ROOT_DIR%"
set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"

set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"

if not exist "%MODEL_MICRO%" (
    echo [ERROR] Modelo no encontrado: %MODEL_MICRO%
    echo Ejecuta run_pipeline.bat primero para entrenar.
    pause & exit /b 1
)
if not exist "%PORTFOLIO%" (
    echo [ERROR] Portfolio no encontrado: %PORTFOLIO%
    pause & exit /b 1
)

echo =================================================================
echo  INFERENCIA SOLO  -  TAG: %TAG%
echo  Modelo    : %MODEL_MICRO%
echo  Portfolio : %PORTFOLIO%
echo =================================================================

REM ============================================================
REM 1. INFERENCIA COORDINADA - 3 POSTURAS
REM ============================================================
echo.
echo [1/5] Inferencia coordinada - 3 posturas...

for %%P in (prudencial balanceado desinversion) do (
    echo.
    echo    [%%P]
    %PY_EXE% -m agent.coordinator_inference ^
        --model-micro %MODEL_MICRO% ^
        --portfolio %PORTFOLIO% ^
        --risk-posture %%P ^
        --vn-micro %VN_MICRO% ^
        --n-steps 5 ^
        --top-k 5 ^
        --tag %TAG%
    if errorlevel 1 (echo [ERROR] Inferencia %%P fallida. & exit /b 1)
)

REM ============================================================
REM 2. STRESS ENGINE (BCE/EBA - PC9/PC10)
REM ============================================================
echo.
echo [2/5] Stress engine (multi-escenario)...
%PY_EXE% -m engines.stress_engine ^
    --tag %TAG% ^
    --portfolio %PORTFOLIO% ^
    --scenarios configs/stress_scenarios.yaml ^
    --postures prudencial balanceado desinversion
if errorlevel 1 (echo [ERROR] Stress engine fallido. & exit /b 1)
echo [OK] reports\stress_summary_%TAG%.csv

REM ============================================================
REM 3. BACKTESTING LIGHT
REM ============================================================
echo.
echo [3/5] Backtesting light...
%PY_EXE% -m reports.backtesting_light ^
    --tag %TAG% ^
    --stress-scenarios configs/stress_scenarios.yaml
if errorlevel 1 (echo [WARNING] Backtesting light fallo (no bloqueante).)

REM ============================================================
REM 4. COMPARACION DE POSTURAS
REM ============================================================
echo.
echo [4/5] Comparacion de posturas...
%PY_EXE% -m reports.compare_postures --tag %TAG%
if errorlevel 1 (echo [WARNING] compare_postures fallo (no bloqueante).)

REM ============================================================
REM 5. KPI REPORT + RESUMEN
REM ============================================================
echo.
echo [5/5] KPI report y resumen final...
%PY_EXE% -m reports.posture_kpi_report --tag %TAG%
if errorlevel 1 (echo [WARNING] posture_kpi_report fallo (no bloqueante).)
%PY_EXE% reports/results_summary.py
if errorlevel 1 (echo [WARNING] results_summary.py fallo (no bloqueante).)

echo.
echo =================================================================
echo  INFERENCIA FINALIZADA  -  TAG: %TAG%
echo  Resultados en: reports\
echo =================================================================
pause
