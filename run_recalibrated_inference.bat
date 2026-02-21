@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM INFERENCIA RECALIBRADA
REM Parametros recalibrados (mandatos selectivos ~20-30%)
REM Genera evidencia completa: KPIs, stress, evaluacion, pack
REM Uso: run_recalibrated_inference.bat [--tag TAG]
REM ============================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set TAG=RECALIBRATED_V2
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--tag" (set "TAG=%~2" & shift & shift & goto parse_args)
shift
goto parse_args
:end_parse

if exist "%ROOT_DIR%.venv\Scripts\activate.bat" call "%ROOT_DIR%.venv\Scripts\activate.bat"

set "PY_EXE=python"
set "PYTHONPATH=%ROOT_DIR%"
set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"
set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"
set "N_STEPS=3"
set "TOP_K=5"

python -c "import sys; print('PYTHON:', sys.executable)"

echo ======================================================================
echo  INFERENCIA RECALIBRADA  -  TAG: %TAG%
echo  Portfolio: %PORTFOLIO%
echo  Parametros recalibrados:
echo    DESINV : mandatos ~1%% cartera (recovery^<3%%, age^>=18m)
echo    PRUD   : sale_floor=25%%, loss_cap=85%%, recovery_min=15%%
echo    BAL    : sale_floor=18%%, loss_cap=90%%, recovery_min=10%%
echo ======================================================================

REM ============================================================
REM 1-3. INFERENCIA COORDINADA - 3 POSTURAS
REM ============================================================
echo.
echo [1/6] Inferencia coordinada - 3 posturas...

for %%P in (prudencial balanceado desinversion) do (
    echo.
    echo    [%%P]
    %PY_EXE% -m agent.coordinator_inference ^
        --model-micro %MODEL_MICRO% ^
        --portfolio %PORTFOLIO% ^
        --risk-posture %%P ^
        --vn-micro %VN_MICRO% ^
        --n-steps %N_STEPS% ^
        --top-k %TOP_K% ^
        --tag %TAG%
    if errorlevel 1 (echo [ERROR] Inferencia %%P fallida. & exit /b 1)
    echo    [OK] %%P completado
)

REM ============================================================
REM 2. STRESS ENGINE (PC9/PC10)
REM ============================================================
echo.
echo [2/6] Stress engine (multi-escenario BCE/EBA)...
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
echo [3/6] Backtesting light...
%PY_EXE% -m reports.backtesting_light ^
    --tag %TAG% ^
    --stress-scenarios configs/stress_scenarios.yaml
if errorlevel 1 (echo [WARNING] Backtesting light fallo (no bloqueante).)

REM ============================================================
REM 4. COMPARACION DE POSTURAS + KPI REPORT
REM ============================================================
echo.
echo [4/6] Comparacion de posturas y KPI report...
%PY_EXE% -m reports.compare_postures --tag %TAG%
if errorlevel 1 (echo [WARNING] compare_postures fallo (no bloqueante).)

%PY_EXE% -m reports.posture_kpi_report --tag %TAG%
if errorlevel 1 (echo [WARNING] posture_kpi_report fallo (no bloqueante).)

REM ============================================================
REM 5. EVALUACION VS BASELINES
REM ============================================================
echo.
echo [5/6] Evaluacion vs baselines...
%PY_EXE% reports/evaluate_against_baselines_sim.py ^
    --tag %TAG% ^
    --select latest ^
    --out reports/evaluation_%TAG%.csv ^
    --report reports/evaluation_%TAG%_report.md
if errorlevel 1 (echo [WARNING] Evaluacion vs baselines fallo (no bloqueante).)

REM ============================================================
REM 6. COMMITTEE PACK
REM ============================================================
echo.
echo [6/6] Generando committee pack...
%PY_EXE% -m reports.make_committee_pack --tag %TAG%
if errorlevel 1 (echo [WARNING] make_committee_pack fallo (no bloqueante).)

echo.
echo ======================================================================
echo  INFERENCIA RECALIBRADA COMPLETADA  -  TAG: %TAG%
echo ======================================================================
echo.
echo  Outputs principales:
echo    reports\stress_summary_%TAG%.csv
echo    reports\backtesting_light_%TAG%.csv
echo    reports\evaluation_%TAG%.csv
echo    reports\evaluation_%TAG%_report.md
echo    reports\committee_pack_%TAG%_*\
echo.
echo  Verificacion KPIs:
echo    python -m reports.posture_kpi_report --tag %TAG%
echo    python -m pytest tests\test_stress_summary_pricing_kpis.py -v
echo.
pause
