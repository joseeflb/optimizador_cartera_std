@echo off
setlocal
cd /d "%~dp0"

REM Intentar activar entorno virtual si existe
if exist "%~dp0\.venv\Scripts\activate.bat" (
    call "%~dp0\.venv\Scripts\activate.bat"
)

python -c "import sys; print('USANDO PYTHON:', sys.executable)"

echo ========================================
echo Ejecutando inferencia para PRUDENCIAL
echo ========================================
python -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture prudencial --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag executability_v1_pru

echo.
echo ========================================
echo Ejecutando inferencia para BALANCEADO
echo ========================================
python -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag executability_v1_bal

echo.
echo ========================================
echo Ejecutando inferencia para DESINVERSION
echo ========================================
python -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture desinversion --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag executability_v1_des

echo.
pause
