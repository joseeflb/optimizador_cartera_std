@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM EJECUTABILIDAD 3 POSTURAS
REM Script de validacion rapida: verifica que las 3 posturas
REM corran sin errores (sin stress engine ni reportes).
REM Uso: run_3_postures_executability.bat [--tag TAG]
REM ============================================================

set "TAG_BASE=executability_v1"

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--tag" (set "TAG_BASE=%~2" & shift & shift & goto parse_args)
shift
goto parse_args
:end_parse

cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"

set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"
set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"

python -c "import sys; print('PYTHON:', sys.executable)"

if not exist "%MODEL_MICRO%" (
    echo [ERROR] Modelo no encontrado: %MODEL_MICRO%
    echo Entrena primero con run_pipeline.bat
    exit /b 1
)
if not exist "%PORTFOLIO%" (
    echo [ERROR] Portfolio no encontrado: %PORTFOLIO%
    exit /b 1
)

echo =================================================================
echo  EJECUTABILIDAD 3 POSTURAS  -  TAG BASE: %TAG_BASE%
echo  Modelo    : %MODEL_MICRO%
echo  Portfolio : %PORTFOLIO%
echo =================================================================

for %%P in (prudencial balanceado desinversion) do (
    echo.
    echo ========================================
    echo  Postura: %%P  (TAG: %TAG_BASE%_%%P)
    echo ========================================
    python -m agent.coordinator_inference ^
        --model-micro %MODEL_MICRO% ^
        --portfolio %PORTFOLIO% ^
        --risk-posture %%P ^
        --vn-micro %VN_MICRO% ^
        --n-steps 3 ^
        --top-k 5 ^
        --tag %TAG_BASE%_%%P
    if errorlevel 1 (echo [ERROR] Postura %%P fallo. & exit /b 1)
    echo [OK] %%P
)

echo.
echo =================================================================
echo  EJECUTABILIDAD OK - las 3 posturas corren sin errores.
echo =================================================================
