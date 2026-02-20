@echo off
setlocal enabledelayedexpansion
:: ==========================================
:: CI LOCAL (Continuous Integration Substitute)
:: ==========================================
:: Ejecuta el pipeline completo de validación y empaquetado.
:: Uso: .\ci_local.bat [TAG]
::
:: Ejemplo: .\ci_local.bat pc8_prep
:: ==========================================

:: 1. Configuración
if "%1"=="" (
    set TAG=pc5_postures_validation
) else (
    set TAG=%1
)

:: Get Timestamp via PowerShell
for /f "usebackq tokens=*" %%a in (`powershell -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do (
    set TS=%%a
)

set LOGFILE=logs\ci_local_%TAG%_%TS%.log

echo ========================================
echo  CI LOCAL STARTED: %date% %time%
echo  TAG: %TAG%
echo  LOG: %LOGFILE%
echo ========================================

:: Redirigir todo a log y consola (usando powershell Tee-Object si se quiere, o simple >>)

echo [%time%] 1. Check Python Environment...
call .venv\Scripts\activate.bat > %LOGFILE% 2>&1
python --version >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python environment check failed.
    exit /b 1
)

echo [%time%] 2. Running Smoke Test...
echo [INFO] Running smoke_test_venv.bat >> %LOGFILE%
call .\smoke_test_venv.bat >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Smoke test failed. See %LOGFILE%
    exit /b 1
)

echo [%time%] 3. Running 3 Postures Inference...
echo [INFO] Running run_3_postures_executability_venv.bat --tag %TAG% >> %LOGFILE%
call .\run_3_postures_executability_venv.bat --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Postures inference failed. See %LOGFILE%
    exit /b 1
)

echo [%time%] 4. Running Comparison...
echo [INFO] Running python -m reports.compare_postures --tag %TAG% >> %LOGFILE%
python -m reports.compare_postures --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Comparison failed. See %LOGFILE%
    exit /b 1
)

echo [%time%] 5. Running Evaluation vs Baselines...
echo [INFO] Running evaluate_against_baselines_sim.py >> %LOGFILE%
python reports/evaluate_against_baselines_sim.py --tag %TAG% --select latest --out reports/evaluation_pc6.csv --report reports/evaluation_report.md >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation failed. See %LOGFILE%
    exit /b 1
)

echo [%time%] 6. Generating Committee Pack (with MEMO)...
echo [INFO] Running make_committee_pack.py >> %LOGFILE%
python -m reports.make_committee_pack --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Pack generation failed. See %LOGFILE%
    exit /b 1
)

echo [%time%] 7. Running Final Test Suite (pytest)...
echo [INFO] Running pytest -q >> %LOGFILE%
python -m pytest -q >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Pytest failed. See %LOGFILE%
    exit /b 1
)

echo ========================================
echo  CI LOCAL FINISHED SUCCESSFULLY AND LOG SAVED
echo ========================================
echo [%time%] SUCCESS. >> %LOGFILE%

endlocal
exit /b 0
