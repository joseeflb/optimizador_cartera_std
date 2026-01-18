@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM ROOT DEL PROYECTO
REM ============================================================
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM ============================================================
REM PYTHON FIJO (evita PATH y WindowsApps)
REM ============================================================
set "PY_EXE=C:\Users\josef\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if not exist "%PY_EXE%" (
  echo ERROR: No se encuentra Python en:
  echo   %PY_EXE%
  pause
  exit /b 1
)

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
echo TB:     %TB_LOGDIR%
echo.

REM ============================================================
REM Verifica main.py
REM ============================================================
if not exist "%ROOT_DIR%main.py" (
  echo ERROR: No se encuentra main.py en %ROOT_DIR%
  pause
  exit /b 1
)

REM ============================================================
REM 1) Verifica dependencias base
REM ============================================================
echo Verificando dependencias (numpy, pandas, openpyxl, torch, gymnasium, stable_baselines3)...
"%PY_EXE%" -c "import numpy,pandas,openpyxl,torch,gymnasium,stable_baselines3; print('OK')"
if errorlevel 1 (
  echo ERROR: Faltan dependencias en este Python.
  echo Ejecuta: install_requirements_smart.py con el mismo Python.
  pause
  exit /b 1
)
echo OK: entorno verificado.
echo.

REM ============================================================
REM 2) Generación de cartera
REM ============================================================
echo [1/4] Generando cartera sintetica (500 prestamos)...
set "START_TIME=%time%"
"%PY_EXE%" "%ROOT_DIR%main.py" generate --n 500 --out "%DATA_DIR%\portfolio_synth.xlsx"
if errorlevel 1 (
  echo ERROR: fallo en generate.
  pause
  exit /b 1
)
echo OK: cartera generada: %DATA_DIR%\portfolio_synth.xlsx
echo Time: %START_TIME% -> %time%
echo.

REM ============================================================
REM 3) Entrenamiento (micro + macro)
REM ============================================================
set "TOTAL_STEPS=500000"
echo [2/4] Entrenando RL (agent both) total_steps=%TOTAL_STEPS% ...
set "START_TIME=%time%"

"%PY_EXE%" "%ROOT_DIR%main.py" train --agent both --portfolio "%DATA_DIR%\portfolio_synth.xlsx" --total-steps %TOTAL_STEPS%
if errorlevel 1 (
  echo ERROR: fallo en train.
  pause
  exit /b 1
)

REM ---- Verifica artefactos reales (segun tu pipeline actual)
set "MODEL_MICRO=%MODELS_DIR%\best_model_loan.zip"
set "MODEL_MACRO=%MODELS_DIR%\best_model_portfolio.zip"
set "VN_MICRO=%MODELS_DIR%\vecnormalize_loan.pkl"
set "VN_MACRO=%MODELS_DIR%\vecnormalize_portfolio.pkl"
set "VN_LOAN=%MODELS_DIR%\vecnormalize_loan.pkl"

if not exist "%MODEL_MICRO%" (
  echo ERROR: no se genero %MODEL_MICRO%
  pause
  exit /b 1
)
if not exist "%MODEL_MACRO%" (
  echo ERROR: no se genero %MODEL_MACRO%
  pause
  exit /b 1
)
if not exist "%VN_MICRO%" (
  echo ERROR: no se genero %VN_MICRO%
  pause
  exit /b 1
)
if not exist "%VN_MACRO%" (
  echo ERROR: no se genero %VN_MACRO%
  pause
  exit /b 1
)

echo OK: entrenamiento completado.
echo Modelos detectados:
echo   MICRO: %MODEL_MICRO%
echo   MACRO: %MODEL_MACRO%
echo   VN_MICRO: %VN_MICRO%
echo   VN_MACRO: %VN_MACRO%
echo   VN_LOAN : %VN_LOAN%
echo Time: %START_TIME% -> %time%
echo.

REM ============================================================
REM 4) Inferencia COORDINADA multi-postura (UNA carpeta DELIVERABLE)
REM ============================================================
echo [3/4] Inferencia COORDINADA multi-postura (prudencial/balanceado/desinversion)...
set "START_TIME=%time%"

REM knobs recomendados para smoke; ajusta si quieres
set "N_STEPS=1"
set "TOP_K=3"
set "TAG=deliverable_run"

echo Ejecutando:
echo   "%PY_EXE%" "%ROOT_DIR%main.py" infer --all-postures --model-micro "%MODEL_MICRO%" --model-macro "%MODEL_MACRO%" --portfolio "%DATA_DIR%\portfolio_synth.xlsx" --tag "%TAG%" --n-steps %N_STEPS% --top-k %TOP_K% --deterministic --vn-micro "%VN_MICRO%" --vn-macro "%VN_MACRO%" --vn-loan "%VN_LOAN%"
echo.

"%PY_EXE%" "%ROOT_DIR%main.py" infer ^
  --all-postures ^
  --model-micro "%MODEL_MICRO%" ^
  --model-macro "%MODEL_MACRO%" ^
  --portfolio "%DATA_DIR%\portfolio_synth.xlsx" ^
  --tag "%TAG%" ^
  --n-steps %N_STEPS% ^
  --top-k %TOP_K% ^
  --deterministic ^
  --vn-micro "%VN_MICRO%" ^
  --vn-macro "%VN_MACRO%" ^
  --vn-loan "%VN_LOAN%"

if errorlevel 1 (
  echo ERROR: fallo en infer (coordinated).
  pause
  exit /b 1
)

echo OK: inferencia completada.
echo Time: %START_TIME% -> %time%
echo.

REM ============================================================
REM 5) Summary (consolidado run-level)
REM ============================================================
echo [4/4] Generando resumen ejecutivo (si aplica)...
"%PY_EXE%" "%ROOT_DIR%main.py" summary
if errorlevel 1 (
  echo ERROR: fallo en summary.
  pause
  exit /b 1
)

echo.
echo ===============================================================
echo PIPELINE COMPLETADO OK
echo Reportes: %REPORTS_DIR%
echo Deberias ver una carpeta:
echo   reports\DELIVERABLE_%TAG%_YYYYMMDD_HHMMSS
echo Con 3 Excels:
echo   decisiones_finales_prudencial.xlsx
echo   decisiones_finales_balanceado.xlsx
echo   decisiones_finales_desinversion.xlsx
echo ===============================================================
echo.
pause
exit /b 0
