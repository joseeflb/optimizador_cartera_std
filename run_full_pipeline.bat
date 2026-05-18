@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM  OPTIMIZADOR DE CARTERAS NPL — Pipeline Completo
REM  Basilea III Metodo Estandar · PPO RL · Banco L1.5
REM ============================================================
REM  USO:
REM    run_full_pipeline.bat                          (cartera sintetica default)
REM    run_full_pipeline.bat --portfolio MiCartera.xlsx  (cartera real)
REM    run_full_pipeline.bat --skip-train              (solo inferencia)
REM    run_full_pipeline.bat --n 1000 --steps 200000   (custom)
REM ============================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM --- Defaults ---
set "PORTFOLIO=data\portfolio_synth.xlsx"
set "N_LOANS=500"
set "STEPS_LOAN=500000"
set "STEPS_PORTFOLIO=200000"
set "TAG=FINAL_V4"
set "SKIP_TRAIN=0"
set "SKIP_GENERATE=0"
set "TOP_K=5"
set "N_STEPS_MACRO=3"

REM --- Parse args ---
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--portfolio"    (set "PORTFOLIO=%~2" & set "SKIP_GENERATE=1" & shift & shift & goto parse_args)
if /i "%~1"=="--n"            (set "N_LOANS=%~2" & shift & shift & goto parse_args)
if /i "%~1"=="--steps"        (set "STEPS_LOAN=%~2" & set "STEPS_PORTFOLIO=%~2" & shift & shift & goto parse_args)
if /i "%~1"=="--tag"          (set "TAG=%~2" & shift & shift & goto parse_args)
if /i "%~1"=="--skip-train"   (set "SKIP_TRAIN=1" & shift & goto parse_args)
if /i "%~1"=="--skip-generate" (set "SKIP_GENERATE=1" & shift & goto parse_args)
shift
goto parse_args
:end_parse

REM --- Activate venv ---
if exist "%ROOT_DIR%.venv\Scripts\activate.bat" call "%ROOT_DIR%.venv\Scripts\activate.bat"
set "PY_EXE=python"
set "PYTHONPATH=%ROOT_DIR%"

REM --- Model paths ---
set "MODEL_LOAN=models\best_model_loan.zip"
set "VN_LOAN=models\vecnormalize_loan.pkl"
set "MODEL_PORTFOLIO=models\best_model_portfolio.zip"
set "VN_PORTFOLIO=models\vecnormalize_portfolio.pkl"

echo.
echo ==================================================================
echo   OPTIMIZADOR DE CARTERAS NPL - Pipeline Completo
echo ==================================================================
echo   Portfolio : %PORTFOLIO%
echo   Tag       : %TAG%
echo   Loans     : %N_LOANS%
echo   Steps     : Loan=%STEPS_LOAN% / Portfolio=%STEPS_PORTFOLIO%
echo   Skip Gen  : %SKIP_GENERATE%
echo   Skip Train: %SKIP_TRAIN%
echo ==================================================================
echo.

REM ======================================
REM  PASO 1: Generar cartera sintetica
REM ======================================
if "%SKIP_GENERATE%"=="1" (
    echo [1/5] Generacion de cartera: OMITIDA (portfolio=%PORTFOLIO%)
    goto step2
)
echo [1/5] Generando cartera sintetica (%N_LOANS% prestamos)...
%PY_EXE% -m data.generate_portfolio --n %N_LOANS% --out %PORTFOLIO%
if errorlevel 1 (echo [ERROR] Generacion fallida. & exit /b 1)
echo [OK] Cartera generada: %PORTFOLIO%

REM ======================================
REM  PASO 2: Entrenamiento RL
REM ======================================
:step2
if "%SKIP_TRAIN%"=="1" (
    echo.
    echo [2/5] Entrenamiento: OMITIDO (--skip-train)
    if not exist "%MODEL_LOAN%" (
        echo [ERROR] Modelo loan no encontrado: %MODEL_LOAN%. Entrena primero.
        exit /b 1
    )
    goto step3
)
echo.
echo [2/5] Entrenando agentes RL...
echo   [2a] Agente LOAN (micro)...
%PY_EXE% -m agent.train_subagents --agent loan --portfolio %PORTFOLIO% --total-steps-loan %STEPS_LOAN% --patience 15
if errorlevel 1 (echo [ERROR] Entrenamiento LOAN fallido. & exit /b 1)
echo   [OK] Agente Loan entrenado.
echo.
echo   [2b] Agente PORTFOLIO (macro)...
%PY_EXE% -m agent.train_subagents --agent portfolio --portfolio %PORTFOLIO% --total-steps-portfolio %STEPS_PORTFOLIO% --eval-freq-portfolio 4096 --patience 25
if errorlevel 1 (echo [ERROR] Entrenamiento PORTFOLIO fallido. & exit /b 1)
echo   [OK] Agente Portfolio entrenado.

REM ======================================
REM  PASO 3: Inferencia coordinada (3 posturas)
REM ======================================
:step3
echo.
echo [3/5] Inferencia coordinada (3 posturas)...
for %%P in (prudencial balanceado desinversion) do (
    echo.
    echo   [%%P] Ejecutando...
    %PY_EXE% -m agent.coordinator_inference ^
        --model-micro %MODEL_LOAN% ^
        --portfolio %PORTFOLIO% ^
        --risk-posture %%P ^
        --vn-micro %VN_LOAN% ^
        --model-macro %MODEL_PORTFOLIO% ^
        --vn-macro %VN_PORTFOLIO% ^
        --n-steps %N_STEPS_MACRO% ^
        --top-k %TOP_K% ^
        --tag %TAG%
    if errorlevel 1 (echo [WARN] Inferencia %%P con warnings -no bloqueante-.)
    echo   [OK] %%P completado
)

REM ======================================
REM  PASO 4: Stress testing
REM ======================================
echo.
echo [4/5] Stress testing (multi-escenario)...
%PY_EXE% -m engines.stress_engine ^
    --tag %TAG% ^
    --portfolio %PORTFOLIO% ^
    --scenarios configs/stress_scenarios.yaml ^
    --postures prudencial balanceado desinversion
if errorlevel 1 (echo [WARN] Stress engine con warnings -no bloqueante-.)

REM ======================================
REM  PASO 5: Reports & Committee Pack
REM ======================================
echo.
echo [5/5] Generando reportes...

%PY_EXE% -m reports.compare_postures --tag %TAG% 2>nul
%PY_EXE% -m reports.posture_kpi_report --tag %TAG% 2>nul
%PY_EXE% -m reports.backtesting_light --tag %TAG% --stress-scenarios configs/stress_scenarios.yaml 2>nul
%PY_EXE% -m reports.make_committee_pack --tag %TAG% 2>nul

echo.
echo ==================================================================
echo   PIPELINE COMPLETADO - TAG: %TAG%
echo ==================================================================
echo.
echo   Modelos:  %MODEL_LOAN%  /  %MODEL_PORTFOLIO%
echo   Reports:  reports\coordinated_inference_%TAG%_*
echo   Stress:   reports\stress_summary_%TAG%.csv
echo.
echo   Para re-ejecutar solo inferencia:
echo     run_full_pipeline.bat --skip-train --portfolio %PORTFOLIO%
echo.
pause
