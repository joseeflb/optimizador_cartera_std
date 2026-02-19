@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================
REM ROOT DEL PROYECTO
REM ============================
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM Intentar activar entorno virtual si existe
if exist "%ROOT_DIR%\.venv\Scripts\activate.bat" (
    call "%ROOT_DIR%\.venv\Scripts\activate.bat"
)


REM ============================
REM PYTHON
REM ============================
set "PY_EXE=python"
set "PY_ARGS="

%PY_EXE% %PY_ARGS% --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: No se encuentra Python.
  pause
  exit /b 1
)

python -c "import sys; print('USANDO PYTHON:', sys.executable)"

echo ===============================================
echo EJECUCION SOLO INFERENCIA (MICRO + MACRO)
echo ===============================================
echo ROOT: %ROOT_DIR%
echo PY:   %PY_EXE%

REM Comprobar si existen modelos antes de correr
if not exist "models\best_model.zip" (
  echo ERROR: No existe models\best_model.zip.
  echo Ejecuta run_pipeline.bat primero para entrenar.
  pause
  exit /b 1
)

echo.
echo OK: artefactos verificados.
echo.

REM --- Postura PRUDENCIAL ---
echo.
echo [1/3] Inferencia PRUDENCIAL...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture prudencial --vn-micro models\vecnormalize_loan.pkl --n-steps 5 --top-k 5 --tag run1
if errorlevel 1 exit /b 1

REM --- Postura BALANCEADO ---
echo.
echo [2/3] Inferencia BALANCEADO...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 5 --top-k 5 --tag run1
if errorlevel 1 exit /b 1

REM --- Postura DESINVERSION ---
echo.
echo [3/3] Inferencia DESINVERSION...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture desinversion --vn-micro models\vecnormalize_loan.pkl --n-steps 5 --top-k 5 --tag run1
if errorlevel 1 exit /b 1


echo.
echo ===============================================
echo FIN INFERENCIA.
echo ===============================================
pause
