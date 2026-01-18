@echo off
setlocal EnableExtensions

REM ============================
REM ROOT DEL PROYECTO
REM ============================
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM ============================
REM PYTHON (launcher)
REM ============================
set "PY_EXE=py"
set "PY_ARGS=-3.11"

%PY_EXE% %PY_ARGS% --version >nul 2>&1
if errorlevel 1 (
  set "PY_ARGS="
  %PY_EXE% %PY_ARGS% --version >nul 2>&1
)

if errorlevel 1 (
  set "PY_EXE=python"
  set "PY_ARGS="
  %PY_EXE% %PY_ARGS% --version >nul 2>&1
)

if errorlevel 1 (
  echo ERROR: Python no disponible.
  pause
  exit /b 1
)

REM ============================
REM PYTHONPATH
REM ============================
set "PYTHONPATH=%ROOT_DIR%"

REM ============================
REM Directorios
REM ============================
set "DATA_DIR=%ROOT_DIR%data"
set "MODELS_DIR=%ROOT_DIR%models"
set "REPORTS_DIR=%ROOT_DIR%reports"

echo ===============================================
echo EJECUCION SOLO INFERENCIA (MICRO + MACRO)
echo ===============================================
echo ROOT: %ROOT_DIR%
echo PY:   %PY_EXE% %PY_ARGS%
echo.

REM ============================
REM Validaciones
REM ============================
if not exist "%MODELS_DIR%\best_model.zip" (
  echo ERROR: Falta models\best_model.zip
  pause
  exit /b 1
)

if not exist "%MODELS_DIR%\vecnormalize_final.pkl" (
  echo ERROR: Falta models\vecnormalize_final.pkl
  pause
  exit /b 1
)

if not exist "%MODELS_DIR%\best_model_portfolio.zip" (
  echo ERROR: Falta models\best_model_portfolio.zip
  pause
  exit /b 1
)

if not exist "%DATA_DIR%\portfolio_synth.xlsx" (
  echo ERROR: Falta data\portfolio_synth.xlsx
  pause
  exit /b 1
)

echo OK: artefactos verificados.
echo.

REM ============================
REM Inferencia coordinada
REM ============================
%PY_EXE% %PY_ARGS% "%ROOT_DIR%agent\coordinator_inference.py" ^
  --model-micro "%MODELS_DIR%\best_model.zip" ^
  --vecnorm "%MODELS_DIR%\vecnormalize_final.pkl" ^
  --portfolio "%DATA_DIR%\portfolio_synth.xlsx" ^
  --model-macro "%MODELS_DIR%\best_model_portfolio.zip" ^
  --n-steps 5 ^
  --top-k 5 ^
  --tag run1 ^
  --all-postures

if errorlevel 1 (
  echo ERROR: fallo en coordinator_inference.py
  pause
  exit /b 1
)

echo OK: inferencia completada.
echo Target final:
echo   reports\final\decisiones_finales_prudencial.xlsx
echo   reports\final\decisiones_finales_balanceado.xlsx
echo   reports\final\decisiones_finales_desinversion.xlsx
echo.
pause
exit /b 0
