@echo off
REM Ejecutar inferencia coordinada para 3 posturas con nueva lógica de ejecutabilidad
echo ========================================
echo Ejecutando inferencia para PRUDENCIAL
echo ========================================
py -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture prudencial --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag executability_v1_pru

echo.
echo ========================================
echo Ejecutando inferencia para BALANCEADO
echo ========================================
py -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag executability_v1_bal

echo.
echo ========================================
echo Ejecutando inferencia para DESINVERSION
echo ========================================
py -m agent.coordinator_inference --model-micro models\best_model_loan.zip --portfolio data\portfolio_synth.xlsx --risk-posture desinversion --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag executability_v1_des

echo.
echo ========================================
echo COMPLETADO: 3 posturas ejecutadas
echo ========================================
pause
