@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM TEST DE REPRODUCIBILIDAD - 3 POSTURAS
REM Corre inferencia 2 veces con el mismo modelo y compara.
REM Uso: run_repro_test.bat [--tag TAG]
REM ============================================================

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if exist "%ROOT_DIR%.venv\Scripts\activate.bat" call "%ROOT_DIR%.venv\Scripts\activate.bat"

set "PY_EXE=python"
set "PYTHONPATH=%ROOT_DIR%"
set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"
set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"

python -c "import sys; print('PYTHON:', sys.executable)"

echo =================================================================
echo  TEST REPRODUCIBILIDAD - Modelo: %MODEL_MICRO%
echo =================================================================

REM ============================================================
REM RUN A
REM ============================================================
echo.
echo [1/3] Run A (primera ejecucion)...
set "TAG_A=repro_A"
for %%P in (prudencial balanceado desinversion) do (
    %PY_EXE% -m agent.coordinator_inference ^
        --model-micro %MODEL_MICRO% ^
        --portfolio %PORTFOLIO% ^
        --risk-posture %%P ^
        --vn-micro %VN_MICRO% ^
        --n-steps 3 ^
        --top-k 5 ^
        --tag %TAG_A%
    if errorlevel 1 (echo [ERROR] Run A - postura %%P fallo. & exit /b 1)
)
echo [OK] Run A completado.

REM ============================================================
REM RUN B
REM ============================================================
echo.
echo [2/3] Run B (repeticion)...
set "TAG_B=repro_B"
for %%P in (prudencial balanceado desinversion) do (
    %PY_EXE% -m agent.coordinator_inference ^
        --model-micro %MODEL_MICRO% ^
        --portfolio %PORTFOLIO% ^
        --risk-posture %%P ^
        --vn-micro %VN_MICRO% ^
        --n-steps 3 ^
        --top-k 5 ^
        --tag %TAG_B%
    if errorlevel 1 (echo [ERROR] Run B - postura %%P fallo. & exit /b 1)
)
echo [OK] Run B completado.

REM ============================================================
REM COMPARACION
REM ============================================================
echo.
echo [3/3] Comparando resultados A vs B...
%PY_EXE% -m reports.compare_postures --tag %TAG_A%
%PY_EXE% -m reports.compare_postures --tag %TAG_B%

echo.
echo =================================================================
echo  REPRODUCIBILIDAD: verificar que los outputs de TAG_A y TAG_B
echo  sean identicos en decisiones si el modelo es determinista.
echo  - reports\decisiones_%TAG_A%_*.xlsx
echo  - reports\decisiones_%TAG_B%_*.xlsx
echo =================================================================
pause
