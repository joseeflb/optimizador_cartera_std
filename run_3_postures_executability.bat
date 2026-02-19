@echo off
setlocal
cd /d "%~dp0"

REM Parse arguments (supports --tag <value>)
set "TAG_BASE=executability_v1"

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--tag" (
    set "TAG_BASE=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

REM Intentar activar entorno virtual si existe
if exist "%~dp0\.venv\Scripts\activate.bat" (
    call "%~dp0\.venv\Scripts\activate.bat"
)

python -c "import sys; print('USANDO PYTHON:', sys.executable)"

echo ========================================
echo Ejecutando inferencia para PRUDENCIAL (TAG: %TAG_BASE%_prudencial)
echo ========================================
python -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture prudencial --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag %TAG_BASE%_prudencial

echo.
echo ========================================
echo Ejecutando inferencia para BALANCEADO (TAG: %TAG_BASE%_balanceado)
echo ========================================
python -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag %TAG_BASE%_balanceado

echo.
echo ========================================
echo Ejecutando inferencia para DESINVERSION (TAG: %TAG_BASE%_desinversion)
echo ========================================
python -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture desinversion --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag %TAG_BASE%_desinversion

echo.
REM pause
