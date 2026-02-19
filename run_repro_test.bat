@echo off
setlocal EnableExtensions

REM ============================
REM Test de Reproducibilidad
REM ============================
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM Intentar activar entorno virtual si existe
if exist "%ROOT_DIR%\.venv\Scripts\activate.bat" (
    call "%ROOT_DIR%\.venv\Scripts\activate.bat"
)

set "PY_EXE=python"
set "PYTHONPATH=%ROOT_DIR%"

python -c "import sys; print('USANDO PYTHON:', sys.executable)"

echo ===============================================
echo TEST DE REPRODUCIBILIDAD - 3 POSTURAS
echo ===============================================
echo.

REM ============================
REM Parámetros idénticos a run original
REM ============================
REM NOTA: Usamos seeds fijas si el código lo permite, o confiamos en 
REM la carga determinista de modelos. 

echo [1/3] Run A (original) ...
REM Simulamos generando un tag único
set "TAG_A=repro_A"
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag %TAG_A%
if errorlevel 1 exit /b 1

echo.
echo [2/3] Run B (repetición) ...
set "TAG_B=repro_B"
%PY_EXE% -m agent.coordinator_inference --model-micro models\best_model.zip --portfolio data\portfolio_synth.xlsx --risk-posture balanceado --vn-micro models\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag %TAG_B%
if errorlevel 1 exit /b 1

echo.
echo [3/3] Comparación de resultados ...
REM Aquí podríamos llamar a un script de diff, 
REM por ahora visual: revisa si los logs/excels son idénticos.

echo.
echo ===============================================
echo VERIFICAR CARPETAS reports/inference_*_%TAG_A%_* y ...%TAG_B%_*
echo Deben ser idénticas en decisiones si el modelo es determinista.
echo ===============================================
pause
