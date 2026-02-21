@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM OPTIMIZADOR DE CARTERAS - PIPELINE COMPLETO
REM Uso: run_pipeline.bat [--tag TAG]
REM Pasos: datos -> entrenamiento -> inferencia(x3) -> stress
REM        -> backtesting -> compare -> kpi_report -> pack
REM ============================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"
set "TAG=run1"

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--tag" (set "TAG=%~2" & shift & shift & goto parse_args)
shift
goto parse_args
:end_parse

REM --- Activar venv ---
if exist "%ROOT_DIR%.venv\Scripts\activate.bat" call "%ROOT_DIR%.venv\Scripts\activate.bat"

set "PY_EXE=python"
%PY_EXE% --version >nul 2>&1
if errorlevel 1 (echo [ERROR] Python no encontrado. & pause & exit /b 1)
python -c "import sys; print('PYTHON:', sys.executable)"

set "PYTHONPATH=%ROOT_DIR%"
set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"

REM --- Portfolio: snapshot real > sintetico ---
set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"

REM --- Directorios ---
if not exist "%ROOT_DIR%logs" mkdir "%ROOT_DIR%logs"
if not exist "%ROOT_DIR%models" mkdir "%ROOT_DIR%models"
if not exist "%ROOT_DIR%reports" mkdir "%ROOT_DIR%reports"
if not exist "%ROOT_DIR%logs\tensorboard" mkdir "%ROOT_DIR%logs\tensorboard"
set "SB3_TENSORBOARD_LOG=%ROOT_DIR%logs\tensorboard"

echo =================================================================
echo  PIPELINE COMPLETO  -  TAG: %TAG%
echo  Portfolio : %PORTFOLIO%
echo  Modelo    : %MODEL_MICRO%
echo =================================================================

REM ============================================================
REM 1. DATOS - generacion o ingesta
REM ============================================================
echo.
echo [1/8] Datos (generacion / ingesta)...
if exist "data\portfolio_snapshot.xlsx" (
    echo [INFO] Ingiriendo portfolio real: data\portfolio_snapshot.xlsx
    %PY_EXE% -m data.ingest_portfolio ^
        --input data\portfolio_snapshot.xlsx ^
        --output data\portfolio_synth.xlsx
) else (
    echo [INFO] Generando portfolio sintetico...
    %PY_EXE% data/generate_portfolio.py
)
if errorlevel 1 (echo [ERROR] Paso 1 (datos) fallido. & exit /b 1)

REM ============================================================
REM 2. ENTRENAMIENTO (Micro RL + Macro RL)
REM ============================================================
echo.
echo [2/8] Entrenamiento Micro + Macro (RL)...
%PY_EXE% agent/train_agent.py --total-steps 20000 --portfolio %PORTFOLIO%
if errorlevel 1 (echo [ERROR] Entrenamiento fallido. & exit /b 1)

REM ============================================================
REM 3. INFERENCIA COORDINADA - 3 POSTURAS
REM ============================================================
echo.
echo [3/8] Inferencia coordinada - 3 posturas (TAG: %TAG%)...

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
REM 4. MOTOR DE ESTRES (BCE/EBA multi-escenario - PC9/PC10)
REM ============================================================
echo.
echo [4/8] Stress engine (multi-escenario)...
%PY_EXE% -m engines.stress_engine ^
    --tag %TAG% ^
    --portfolio %PORTFOLIO% ^
    --scenarios configs/stress_scenarios.yaml ^
    --postures prudencial balanceado desinversion
if errorlevel 1 (echo [ERROR] Stress engine fallido. & exit /b 1)
if not exist "reports\stress_summary_%TAG%.csv" (
    echo [ERROR] stress_summary_%TAG%.csv no generado.
    exit /b 1
)
echo [OK] reports\stress_summary_%TAG%.csv

REM --- Verificar KPIs PC10 en stress summary ---
powershell -Command "$f='reports\stress_summary_%TAG%.csv'; try { $cols=(Import-Csv $f)[0].PSObject.Properties.Name; $kpis='sale_pnl_total','avg_sale_pnl','avg_bid_pct_ead','sell_blocked_count'; $miss=$kpis|Where-Object{$_ -notin $cols}; if($miss){Write-Output \"[WARNING] PC10 KPI cols ausentes: $($miss -join ', ')\"}else{Write-Output '[OK] PC10 KPI cols presentes'} } catch { Write-Output '[SKIP] No se pudo verificar KPIs PC10' }"

REM ============================================================
REM 5. BACKTESTING LIGHT
REM ============================================================
echo.
echo [5/8] Backtesting light...
%PY_EXE% -m reports.backtesting_light ^
    --tag %TAG% ^
    --stress-scenarios configs/stress_scenarios.yaml
if errorlevel 1 (echo [WARNING] Backtesting light fallo (no bloqueante).)

REM ============================================================
REM 6. COMPARACION DE POSTURAS
REM ============================================================
echo.
echo [6/8] Comparacion de posturas...
%PY_EXE% -m reports.compare_postures --tag %TAG%
if errorlevel 1 (echo [WARNING] compare_postures fallo (no bloqueante).)

REM ============================================================
REM 7. KPI REPORT POR POSTURA (pricing KPIs PC10)
REM ============================================================
echo.
echo [7/8] KPI report por postura...
%PY_EXE% -m reports.posture_kpi_report --tag %TAG%
if errorlevel 1 (echo [WARNING] posture_kpi_report fallo (no bloqueante).)
%PY_EXE% reports/results_summary.py
if errorlevel 1 (echo [WARNING] results_summary.py fallo (no bloqueante).)

REM ============================================================
REM 8. COMMITTEE PACK
REM ============================================================
echo.
echo [8/8] Generando committee pack...
%PY_EXE% -m reports.make_committee_pack --tag %TAG%
if errorlevel 1 (echo [WARNING] make_committee_pack fallo (no bloqueante).)

echo.
echo =================================================================
echo  PIPELINE FINALIZADO  -  TAG: %TAG%
echo  Resultados en: reports\
echo =================================================================
pause
