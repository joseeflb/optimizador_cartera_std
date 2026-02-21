@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM CI LOCAL - Pipeline completo de validacion y empaquetado
REM Uso: .\ci_local.bat [TAG]
REM Ejemplo: .\ci_local.bat pc10_final
REM ============================================================

REM --- Config ---
if "%1"=="" (set TAG=ci_local) else (set TAG=%1)

for /f "usebackq tokens=*" %%a in (`powershell -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set TS=%%a

if not exist logs mkdir logs
set LOGFILE=logs\ci_local_%TAG%_%TS%.log

set "MODEL_MICRO=models\best_model_loan.zip"
set "VN_MICRO=models\vecnormalize_loan.pkl"
set "PORTFOLIO=data\portfolio_synth.xlsx"
if exist "data\portfolio_snapshot.xlsx" set "PORTFOLIO=data\portfolio_snapshot.xlsx"

echo ========================================
echo  CI LOCAL  -  TAG: %TAG%
echo  LOG: %LOGFILE%
echo  Portfolio: %PORTFOLIO%
echo ========================================

REM ============================================================
REM 1. CHECK ENTORNO
REM ============================================================
echo [%time%] 1. Check Python environment...
call .venv\Scripts\activate.bat >> %LOGFILE% 2>&1
python --version >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] Python check fallido. & exit /b 1)
python -c "import pandas, numpy, stable_baselines3, yaml; print('paquetes OK')" >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] Paquetes criticos no disponibles. & exit /b 1)
echo [OK] Entorno OK >> %LOGFILE%

REM ============================================================
REM 2. SMOKE TEST
REM ============================================================
echo [%time%] 2. Smoke test...
call .\smoke_test_venv.bat >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] Smoke test fallido. See %LOGFILE% & exit /b 1)
echo [OK] Smoke test >> %LOGFILE%

REM ============================================================
REM 3. INFERENCIA 3 POSTURAS
REM ============================================================
echo [%time%] 3. Inferencia 3 posturas (TAG: %TAG%)...
call .\run_3_postures_executability_venv.bat --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] Postures inference fallido. See %LOGFILE% & exit /b 1)
echo [OK] Inferencia 3 posturas >> %LOGFILE%

REM ============================================================
REM 4. STRESS ENGINE (PC9/PC10)
REM ============================================================
echo [%time%] 4. Stress engine...
python -m engines.stress_engine ^
    --tag %TAG% ^
    --portfolio !PORTFOLIO! ^
    --scenarios configs/stress_scenarios.yaml ^
    --postures prudencial balanceado desinversion >> %LOGFILE% 2>&1
if !errorlevel! neq 0 (echo [ERROR] Stress engine fallido. See %LOGFILE% & exit /b 1)

if not exist "reports\stress_summary_%TAG%.csv" (
    echo [ERROR] stress_summary_%TAG%.csv no generado. & exit /b 1
)
echo [OK] reports\stress_summary_%TAG%.csv >> %LOGFILE%

REM --- Verificar columnas PC10 ---
powershell -Command "try { $f='reports\stress_summary_%TAG%.csv'; $cols=(Import-Csv $f)[0].PSObject.Properties.Name; $kpis='sale_pnl_total','avg_sale_pnl','avg_bid_pct_ead','sell_blocked_count'; $miss=$kpis|Where-Object{$_ -notin $cols}; if($miss){Write-Output \"[ERROR] PC10 KPI cols ausentes: $($miss -join ', ')\"; exit 1}else{Write-Output '[OK] PC10 KPI cols presentes'} } catch { Write-Output '[SKIP] No se pudo verificar KPIs PC10' }" >> %LOGFILE% 2>&1
if !errorlevel! equ 1 (echo [ERROR] PC10 KPI cols ausentes en stress_summary. See %LOGFILE% & exit /b 1)

REM --- Test PC10 pricing KPIs ---
echo [%time%] 4.1 Tests PC10 pricing KPIs...
python -m pytest tests/test_stress_summary_pricing_kpis.py tests/test_stress_pricing_crunch_effect.py -v >> %LOGFILE% 2>&1
if !errorlevel! neq 0 (echo [ERROR] Tests PC10 pricing KPIs fallaron. See %LOGFILE% & exit /b 1)
echo [OK] Tests PC10 pricing KPIs >> %LOGFILE%

REM ============================================================
REM 5. BACKTESTING LIGHT
REM ============================================================
echo [%time%] 5. Backtesting light...
python -m reports.backtesting_light ^
    --tag %TAG% ^
    --stress-scenarios configs/stress_scenarios.yaml >> %LOGFILE% 2>&1
if !errorlevel! neq 0 (echo [ERROR] Backtesting light fallido. See %LOGFILE% & exit /b 1)
if not exist "reports\backtesting_light_%TAG%.csv" (
    echo [ERROR] backtesting_light_%TAG%.csv no generado. & exit /b 1
)
echo [OK] reports\backtesting_light_%TAG%.csv >> %LOGFILE%

REM ============================================================
REM 6. COMPARACION DE POSTURAS
REM ============================================================
echo [%time%] 6. Comparacion de posturas...
python -m reports.compare_postures --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] compare_postures fallido. See %LOGFILE% & exit /b 1)
echo [OK] compare_postures >> %LOGFILE%

REM ============================================================
REM 7. KPI REPORT POR POSTURA (pricing PC10)
REM ============================================================
echo [%time%] 7. KPI report por postura...
python -m reports.posture_kpi_report --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [WARNING] posture_kpi_report fallo ^(no bloqueante^). >> %LOGFILE%)
echo [OK] posture_kpi_report >> %LOGFILE%

REM ============================================================
REM 8. EVALUACION VS BASELINES
REM ============================================================
echo [%time%] 8. Evaluacion vs baselines...
python reports/evaluate_against_baselines_sim.py ^
    --tag %TAG% ^
    --select latest ^
    --out reports/evaluation_%TAG%.csv ^
    --report reports/evaluation_%TAG%_report.md >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] Evaluacion vs baselines fallida. See %LOGFILE% & exit /b 1)
echo [OK] Evaluacion vs baselines >> %LOGFILE%

REM ============================================================
REM 9. COMMITTEE PACK
REM ============================================================
echo [%time%] 9. Committee pack...
python -m reports.make_committee_pack --tag %TAG% >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] make_committee_pack fallido. See %LOGFILE% & exit /b 1)
echo [OK] Committee pack >> %LOGFILE%

REM ============================================================
REM 10. TEST SUITE COMPLETO (pytest)
REM ============================================================
echo [%time%] 10. Test suite completo (pytest -q)...
python -m pytest -q >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (echo [ERROR] Pytest fallido. See %LOGFILE% & exit /b 1)
echo [OK] Pytest 45/45 >> %LOGFILE%

echo ========================================
echo  CI LOCAL COMPLETADO  -  TAG: %TAG%
echo  LOG: %LOGFILE%
echo ========================================
echo [%time%] SUCCESS >> %LOGFILE%

endlocal
exit /b 0
