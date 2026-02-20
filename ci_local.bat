@echo off
setlocal enabledelayedexpansion
:: ==========================================
:: CI LOCAL (Continuous Integration Substitute)
:: ==========================================
:: Ejecuta el pipeline completo de validaci�n y empaquetado.
:: Uso: .\ci_local.bat [TAG]
::
:: Ejemplo: .\ci_local.bat pc8_prep
:: ==========================================

:: 1. Configuraci�n
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

echo [%time%] 3.5. Running Stress and Sensitivities (PC9)...
echo [INFO] Running stress engine... >> %LOGFILE%
:: Using portfolio_snapshot.xlsx for final/ci run, or verify existence
if exist data\portfolio_snapshot.xlsx (
    set PORTFOLIO=data\portfolio_snapshot.xlsx
) else (
    set PORTFOLIO=data\portfolio_synth.xlsx
)
python -m engines.stress_engine --tag %TAG% --portfolio !PORTFOLIO! --scenarios configs/stress_scenarios.yaml --postures prudencial balanceado desinversion >> %LOGFILE% 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Stress Engine failed. See %LOGFILE%
    exit /b 1
)
:: Verify stress summary artifact exists
if not exist "reports\stress_summary_%TAG%.csv" (
    echo [ERROR] Stress summary not generated: reports\stress_summary_%TAG%.csv
    exit /b 1
)
echo [OK] Stress summary: reports\stress_summary_%TAG%.csv >> %LOGFILE%
:: Warn if pricing_crunch rows are identical to baseline (would indicate no-op pricing shock)
powershell -Command "try { $df=Import-Csv 'reports\stress_summary_%TAG%.csv'; $bl=($df | Where-Object {$_.scenario -eq 'baseline'} | Measure-Object -Property n_sales -Sum).Sum; $pc=($df | Where-Object {$_.scenario -eq 'pricing_crunch'} | Measure-Object -Property n_sales -Sum).Sum; if($bl -eq $pc){Write-Output '[WARNING] pricing_crunch n_sales identical to baseline - BID_HAIRCUT_GLOBAL may not be applied'}else{Write-Output '[OK] pricing_crunch differs from baseline (pricing shock operative)'} } catch { Write-Output '[SKIP] Could not compare pricing_crunch vs baseline' }" >> %LOGFILE% 2>&1

echo [INFO] Running backtesting light... >> %LOGFILE% 
python -m reports.backtesting_light --tag %TAG% --stress-scenarios configs/stress_scenarios.yaml >> %LOGFILE% 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Backtesting Light failed. See %LOGFILE%
    exit /b 1
)
:: Verify backtesting artifact exists
if not exist "reports\backtesting_light_%TAG%.csv" (
    echo [ERROR] Backtesting light CSV not generated: reports\backtesting_light_%TAG%.csv
    exit /b 1
)
echo [OK] Backtesting light: reports\backtesting_light_%TAG%.csv >> %LOGFILE%

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
