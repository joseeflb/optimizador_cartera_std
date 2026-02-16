@echo off
REM ============================================================
REM run_recalibrated_inference.bat
REM Ejecuta inferencia con parámetros RECALIBRADOS (mandatos selectivos ~20-30%)
REM y genera evidencia completa para comité
REM ============================================================
setlocal enabledelayedexpansion

set TAG=RECALIBRATED_V2
set MODEL_MICRO=models\best_model_loan.zip
set VN_MICRO=models\vecnormalize_loan.pkl
set PORTFOLIO=data\portfolio_synth.xlsx
set N_STEPS=3
set TOP_K=5

echo ======================================================================
echo INFERENCIA RECALIBRADA (Mandatos Selectivos + PRUD vs BAL Diferenciadas)
echo ======================================================================
echo.
echo TAG: %TAG%
echo Parametros RECALIBRADOS:
echo   - DESINV: mandatos ~1%% cartera (recovery^<3%%, age^>=18m)
echo   - PRUD:  sale_floor=25%%, loss_cap=85%%, recovery_min=15%%
echo   - BAL:   sale_floor=18%%, loss_cap=90%%, recovery_min=10%%
echo.
echo ======================================================================

REM ============================================================
REM PASO 1: Inferencia PRUDENCIAL
REM ============================================================
echo.
echo [1/5] Ejecutando PRUDENCIAL...
echo ======================================================================
python -m agent.coordinator_inference ^
    --model-micro %MODEL_MICRO% ^
    --portfolio %PORTFOLIO% ^
    --risk-posture prudencial ^
    --vn-micro %VN_MICRO% ^
    --n-steps %N_STEPS% ^
    --top-k %TOP_K% ^
    --tag %TAG%_pru

if errorlevel 1 (
    echo ❌ ERROR en PRUDENCIAL
    exit /b 1
)
echo ✅ PRUDENCIAL completado

REM ============================================================
REM PASO 2: Inferencia BALANCEADO
REM ============================================================
echo.
echo [2/5] Ejecutando BALANCEADO...
echo ======================================================================
python -m agent.coordinator_inference ^
    --model-micro %MODEL_MICRO% ^
    --portfolio %PORTFOLIO% ^
    --risk-posture balanceado ^
    --vn-micro %VN_MICRO% ^
    --n-steps %N_STEPS% ^
    --top-k %TOP_K% ^
    --tag %TAG%_bal

if errorlevel 1 (
    echo ❌ ERROR en BALANCEADO
    exit /b 1
)
echo ✅ BALANCEADO completado

REM ============================================================
REM PASO 3: Inferencia DESINVERSION
REM ============================================================
echo.
echo [3/5] Ejecutando DESINVERSION...
echo ======================================================================
python -m agent.coordinator_inference ^
    --model-micro %MODEL_MICRO% ^
    --portfolio %PORTFOLIO% ^
    --risk-posture desinversion ^
    --vn-micro %VN_MICRO% ^
    --n-steps %N_STEPS% ^
    --top-k %TOP_K% ^
    --tag %TAG%_des

if errorlevel 1 (
    echo ❌ ERROR en DESINVERSION
    exit /b 1
)
echo ✅ DESINVERSION completado

REM ============================================================
REM PASO 4: Análisis de Calibración
REM ============================================================
echo.
echo [4/5] Analizando calibración (mixes, KPIs, mandatos, casos frontera)...
echo ======================================================================
python _tmp\analyze_calibration_evidence.py ^
    --pru "reports\*%TAG%_pru*.xlsx" ^
    --bal "reports\*%TAG%_bal*.xlsx" ^
    --des "reports\*%TAG%_des*.xlsx" ^
    --output-dir reports\calibration

if errorlevel 1 (
    echo ⚠️ WARNING: Análisis de calibración falló (puede faltar archivo)
) else (
    echo ✅ Análisis de calibración completado
)

REM ============================================================
REM PASO 5: Export CSV Canónico + MANIFEST
REM ============================================================
echo.
echo [5/5] Exportando CSV canónico + MANIFEST (reproducibilidad)...
echo ======================================================================

REM PRUDENCIAL
for /f "delims=" %%f in ('dir /b /od reports\*%TAG%_pru*.xlsx 2^>nul') do set XLSX_PRU=%%f
if defined XLSX_PRU (
    python _tmp\export_canonical_csv.py ^
        --xlsx "reports\!XLSX_PRU!" ^
        --output-dir reports\canonical ^
        --tag %TAG% ^
        --posture prudencial
    echo ✅ PRUDENCIAL CSV + MANIFEST exportado
) else (
    echo ⚠️ WARNING: XLSX PRUDENCIAL no encontrado
)

REM BALANCEADO
for /f "delims=" %%f in ('dir /b /od reports\*%TAG%_bal*.xlsx 2^>nul') do set XLSX_BAL=%%f
if defined XLSX_BAL (
    python _tmp\export_canonical_csv.py ^
        --xlsx "reports\!XLSX_BAL!" ^
        --output-dir reports\canonical ^
        --tag %TAG% ^
        --posture balanceado
    echo ✅ BALANCEADO CSV + MANIFEST exportado
) else (
    echo ⚠️ WARNING: XLSX BALANCEADO no encontrado
)

REM DESINVERSION
for /f "delims=" %%f in ('dir /b /od reports\*%TAG%_des*.xlsx 2^>nul') do set XLSX_DES=%%f
if defined XLSX_DES (
    python _tmp\export_canonical_csv.py ^
        --xlsx "reports\!XLSX_DES!" ^
        --output-dir reports\canonical ^
        --tag %TAG% ^
        --posture desinversion
    echo ✅ DESINVERSION CSV + MANIFEST exportado
) else (
    echo ⚠️ WARNING: XLSX DESINVERSION no encontrado
)

REM ============================================================
REM RESUMEN FINAL
REM ============================================================
echo.
echo ======================================================================
echo ✅ INFERENCIA RECALIBRADA COMPLETADA
echo ======================================================================
echo.
echo 📁 Outputs generados:
echo    - reports\decisiones_%TAG%_pru_*.xlsx
echo    - reports\decisiones_%TAG%_bal_*.xlsx
echo    - reports\decisiones_%TAG%_des_*.xlsx
echo    - reports\calibration\calibration_mixes_*.csv
echo    - reports\calibration\calibration_kpis_*.csv
echo    - reports\calibration\calibration_mandates_*.csv
echo    - reports\calibration\calibration_frontier_cases_*.csv
echo    - reports\canonical\decisiones_pru_canonical.csv + MANIFEST_pru.json
echo    - reports\canonical\decisiones_bal_canonical.csv + MANIFEST_bal.json
echo    - reports\canonical\decisiones_des_canonical.csv + MANIFEST_des.json
echo.
echo 🔐 Verificar integridad:
echo    python _tmp\verify_manifest.py --manifest reports\canonical\MANIFEST_pru.json
echo    python _tmp\verify_manifest.py --manifest reports\canonical\MANIFEST_bal.json
echo    python _tmp\verify_manifest.py --manifest reports\canonical\MANIFEST_des.json
echo.
echo 📋 PRÓXIMOS PASOS:
echo    1. Revisar calibration_mixes: ¿DESINV mandatos ~20-30%%?
echo    2. Revisar calibration_mandates: ¿mandatos vs voluntarias balanceados?
echo    3. Revisar calibration_frontier_cases: ¿PRUD ≠ BAL diferenciados?
echo    4. Si OK: preparar deck para comité (evidencia completa)
echo.
pause
