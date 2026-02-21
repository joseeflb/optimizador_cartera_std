@echo off
setlocal EnableExtensions

REM ============================================================
REM SMOKE TEST - Banco L1.5
REM Verifica Python, paquetes criticos, tests unitarios e inferencia dummy.
REM ============================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if exist "%ROOT_DIR%.venv\Scripts\activate.bat" call "%ROOT_DIR%.venv\Scripts\activate.bat"

set "PY_EXE=python"
where python >nul 2>&1
if errorlevel 1 (echo [ERROR] Python no encontrado. & pause & exit /b 1)
python -c "import sys; print('PYTHON:', sys.executable)"

set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"
set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"

set ERRORS=0

REM ============================================================
REM 1. PAQUETES CRITICOS
REM ============================================================
echo.
echo [SMOKE 1/4] Verificando paquetes criticos...
%PY_EXE% -c "import pandas, numpy, stable_baselines3, gymnasium, yaml; print('[OK] numpy', numpy.__version__, '| pandas', pandas.__version__, '| sb3', stable_baselines3.__version__)"
if errorlevel 1 (echo [FAIL] Paquetes criticos no disponibles. & set ERRORS=1) else (echo [PASS] Paquetes OK.)

REM ============================================================
REM 2. UNIT TESTS (pytest)
REM ============================================================
echo.
echo [SMOKE 2/4] Unit tests (pytest -q)...
%PY_EXE% -m pytest -q --tb=short
if errorlevel 1 (
    echo [FAIL] Unit tests fallaron.
    set ERRORS=1
) else (
    echo [PASS] Tests OK.
)

REM ============================================================
REM 3. CARGA DE MODELO
REM ============================================================
echo.
echo [SMOKE 3/4] Carga de modelo...
if not exist "%MODEL_MICRO%" (
    echo [SKIP] Modelo no encontrado: %MODEL_MICRO% ^(no bloqueante en entorno sin artefactos^)
) else (
    %PY_EXE% -c "from stable_baselines3 import PPO; m=PPO.load(r'%MODEL_MICRO%'); print('[OK] Modelo cargado:', type(m).__name__)"
    if errorlevel 1 (echo [FAIL] Carga de modelo fallo. & set ERRORS=1) else (echo [PASS] Modelo cargable.)
)

REM ============================================================
REM 4. INFERENCIA DUMMY (1 paso, balanceado)
REM ============================================================
echo.
echo [SMOKE 4/4] Inferencia dummy (1 paso)...
if exist "%MODEL_MICRO%" (
    if exist "%PORTFOLIO%" (
        %PY_EXE% -m agent.coordinator_inference ^
            --model-micro %MODEL_MICRO% ^
            --portfolio %PORTFOLIO% ^
            --risk-posture balanceado ^
            --vn-micro %VN_MICRO% ^
            --n-steps 1 ^
            --top-k 2 ^
            --tag smoke_test
        if errorlevel 1 (echo [FAIL] Inferencia dummy fallo. & set ERRORS=1) else (echo [PASS] Inferencia OK.)
    ) else (
        echo [SKIP] Portfolio no encontrado: %PORTFOLIO%
    )
) else (
    echo [SKIP] Modelo no disponible para inferencia dummy.
)

echo.
echo ============================================================
if %ERRORS%==0 (
    echo  SMOKE TEST COMPLETADO SIN ERRORES
) else (
    echo  SMOKE TEST COMPLETADO CON FALLOS - revisar mensajes arriba
)
echo ============================================================
pause
