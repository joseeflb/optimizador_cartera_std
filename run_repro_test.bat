@echo off
setlocal EnableExtensions

REM ============================
REM Test de Reproducibilidad
REM ============================
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PY_EXE=py"
set "PYTHONPATH=%ROOT_DIR%"

echo ===============================================
echo TEST DE REPRODUCIBILIDAD - 3 POSTURAS
echo ===============================================
echo.

REM ============================
REM Parámetros idénticos a run original
REM ============================
set "MODEL_MICRO=%ROOT_DIR%models\best_model_loan.zip"
set "VN_MICRO=%ROOT_DIR%models\vecnormalize_loan.pkl"
set "PORTFOLIO=%ROOT_DIR%data\portfolio_synth.xlsx"
set "N_STEPS=3"
set "TOP_K=5"
set "TAG=repro_test"

echo Modelo micro: %MODEL_MICRO%
echo VecNormalize: %VN_MICRO%
echo Portfolio: %PORTFOLIO%
echo n-steps: %N_STEPS%
echo top-k: %TOP_K%
echo Tag: %TAG%
echo.

REM ============================
REM PRUDENCIAL
REM ============================
echo [1/3] Ejecutando PRUDENCIAL...
"%PY_EXE%" -m agent.coordinator_inference ^
  --model-micro "%MODEL_MICRO%" ^
  --portfolio "%PORTFOLIO%" ^
  --risk-posture prudencial ^
  --vn-micro "%VN_MICRO%" ^
  --n-steps %N_STEPS% ^
  --top-k %TOP_K% ^
  --tag %TAG%_pru

if errorlevel 1 (
  echo ERROR: PRUDENCIAL fallo
  pause
  exit /b 1
)
echo OK: PRUDENCIAL completado
echo.

REM ============================
REM BALANCEADO
REM ============================
echo [2/3] Ejecutando BALANCEADO...
"%PY_EXE%" -m agent.coordinator_inference ^
  --model-micro "%MODEL_MICRO%" ^
  --portfolio "%PORTFOLIO%" ^
  --risk-posture balanceado ^
  --vn-micro "%VN_MICRO%" ^
  --n-steps %N_STEPS% ^
  --top-k %TOP_K% ^
  --tag %TAG%_bal

if errorlevel 1 (
  echo ERROR: BALANCEADO fallo
  pause
  exit /b 1
)
echo OK: BALANCEADO completado
echo.

REM ============================
REM DESINVERSION
REM ============================
echo [3/3] Ejecutando DESINVERSION...
"%PY_EXE%" -m agent.coordinator_inference ^
  --model-micro "%MODEL_MICRO%" ^
  --portfolio "%PORTFOLIO%" ^
  --risk-posture desinversion ^
  --vn-micro "%VN_MICRO%" ^
  --n-steps %N_STEPS% ^
  --top-k %TOP_K% ^
  --tag %TAG%_des

if errorlevel 1 (
  echo ERROR: DESINVERSION fallo
  pause
  exit /b 1
)
echo OK: DESINVERSION completado
echo.

echo ===============================================
echo 3 INFERENCIAS COMPLETADAS
echo Carpetas generadas en reports\
echo ===============================================
echo.
pause
exit /b 0
