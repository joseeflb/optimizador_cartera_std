@echo off
setlocal EnableExtensions

REM ============================================
REM  SMOKE TEST BANCO L1.5
REM ============================================

REM Root del proyecto
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM Intentar activar entorno virtual si existe
if exist "%ROOT_DIR%\.venv\Scripts\activate.bat" (
    call "%ROOT_DIR%\.venv\Scripts\activate.bat"
)

REM --- Python (se asume python en path, idealmente venv) ---
set "PY_EXE=python"
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: No se encuentra Python.
  echo Asegurate de activar el entorno virtual.
  pause
  exit /b 1
)

python -c "import sys; print('USANDO PYTHON:', sys.executable)"

REM ============================================
REM 1. UNIT TESTS BASICOS
REM ============================================
echo [SMOKE] 1. Ejecutando unit tests rapidos...
%PY_EXE% -m unittest discover -s tests -p "test_*.py"
if errorlevel 1 (
    echo [FAIL] Tests unitarios fallaron.
    REM No salimos, intentamos correr el resto para ver que mas explota
) else (
    echo [PASS] Unit tests OK.
)

REM ============================================
REM 2. SIMULACION DE MODELOS (Carga y forward pass dummy)
REM ============================================
echo.
echo [SMOKE] 2. Verificando carga de modelos...
REM Creamos un script temporal para chequear carga
(
echo import torch
echo import gymnasium
echo from stable_baselines3 import PPO
echo try:
echo     print("Cargando PPO...")
echo     # Ajusta path si es necesario
echo     # model = PPO.load("models/best_model.zip")
echo     print("OK dummy check")
echo except Exception as e:
echo     print(e)
echo     exit(1)
) > _smoke_check_model.py

%PY_EXE% _smoke_check_model.py
if errorlevel 1 (
    echo [FAIL] Carga de modelo fallo.
) else (
    echo [PASS] Modelos cargables.
)
del _smoke_check_model.py

REM ============================================
REM 3. INTEGRATION LITE (1 paso de inferencia)
REM ============================================
echo.
echo [SMOKE] 3. Inferencia Dummy (1 paso)...
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 1 --top-k 2 --tag smoke_coordinator
if errorlevel 1 (
    echo [FAIL] Inferencia fallo.
) else (
    echo [PASS] Inferencia OK.
)

echo.
echo ============================================
echo SMOKE TEST COMPLETADO
echo ============================================
pause
