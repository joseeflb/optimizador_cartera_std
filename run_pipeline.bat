@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM ROOT DEL PROYECTO & ENTORNO VIRTUAL
REM ============================================================
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM Intentar activar entorno virtual si existe
if exist "%ROOT_DIR%\.venv\Scripts\activate.bat" (
    call "%ROOT_DIR%\.venv\Scripts\activate.bat"
)

REM ============================================================
REM PYTHON (Forzar uso de python del path, que debería ser el del venv)
REM ============================================================
set "PY_EXE=python"
%PY_EXE% --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: No se encuentra Python.
  echo Asegurate de activar el entorno virtual o instalar Python.
  pause
  exit /b 1
)

python -c "import sys; print('USANDO PYTHON:', sys.executable)"

REM ============================================================
REM PYTHONPATH (imports robustos)
REM ============================================================
set "PYTHONPATH=%ROOT_DIR%"

REM ============================================================
REM Directorios
REM ============================================================
set "DATA_DIR=%ROOT_DIR%data"
set "MODELS_DIR=%ROOT_DIR%models"
set "LOGS_DIR=%ROOT_DIR%logs"
set "REPORTS_DIR=%ROOT_DIR%reports"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
if not exist "%REPORTS_DIR%" mkdir "%REPORTS_DIR%"

set "TB_LOGDIR=%LOGS_DIR%\tensorboard"
if not exist "%TB_LOGDIR%" mkdir "%TB_LOGDIR%"
set "SB3_TENSORBOARD_LOG=%TB_LOGDIR%"

echo ===============================================================
echo OPTIMIZADOR DE CARTERAS NPL - PIPELINE COMPLETO (DELIVERABLE)
echo (Generate -> Train -> Infer (3 posturas) -> Summary)
echo ===============================================================
echo ROOT:   %ROOT_DIR%
echo PYTHON: %PY_EXE%

REM ============================================================
REM 1. GENERACION DE DATOS (Portfolio + Loans + Market)
REM ============================================================
echo.
echo [1/4] Generando datos sinteticos...
%PY_EXE% data/generate_portfolio.py
if errorlevel 1 (
  echo ERROR en generate_portfolio.py
  exit /b 1
)

REM ============================================================
REM 2. ENTRENAMIENTO (Reinforcement Learning)
REM ============================================================
echo.
echo [2/4] Entrenando Agentes (Micro + Macro)...

REM Entrena micro (loan level) y macro (portfolio level)
REM Ajusta steps si quieres algo rápido (ej. --total-timesteps 5000)
%PY_EXE% agent/train_agent.py --total-steps 20000 --portfolio data/portfolio_synth.xlsx
if errorlevel 1 (
  echo ERROR en train_agent.py
  exit /b 1
)

REM ============================================================
REM 3. INFERENCIA COORDINADA (3 POSTURAS)
REM ============================================================
echo.
echo [3/4] Ejecutando Inferencia (Coordinador Micro-Macro)...
echo Se generaran 3 sets de resultados: Prudencial / Balanceado / Desinversion

REM --- Postura PRUDENCIAL ---
echo.
echo ... Ejecutando Postura PRUDENCIAL ...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture prudencial --vn-micro models\vecnormalize_loan.pkl --n-steps 5 --top-k 5 --tag run1
if errorlevel 1 (
  echo ERROR en inferencia PRUDENCIAL
  exit /b 1
)

REM --- Postura BALANCEADO ---
echo.
echo ... Ejecutando Postura BALANCEADO ...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 5 --top-k 5 --tag run1
if errorlevel 1 (
  echo ERROR en inferencia BALANCEADO
  exit /b 1
)

REM --- Postura DESINVERSION ---
echo.
echo ... Ejecutando Postura DESINVERSION ...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture desinversion --vn-micro models\vecnormalize_loan.pkl --n-steps 5 --top-k 5 --tag run1
if errorlevel 1 (
  echo ERROR en inferencia DESINVERSION
  exit /b 1
)

REM ============================================================
REM 4. GENERACION DE REPORTES FINALES
REM ============================================================
echo.
echo [4/4] Generando reportes y resumenes...

REM Ejecuta el script de reporte resumen
REM Ajustar paths si los outputs cambian de nombre/fecha
REM Por defecto coordinator_inference guarda en reports/inference_YYYYMMDD_...

REM (Opcional) Un script que busque el ultimo run y genere un dashboard
%PY_EXE% reports/results_summary.py
if errorlevel 1 (
    echo AVISO: No se pudo generar el resumen global o no existe el script results_summary.py
)

echo.
echo ===========================================================
echo PROCESO FINALIZADO EXITOSAMENTE.
echo Revisa la carpeta reports/ para ver los resultados.
echo ===========================================================
pause
