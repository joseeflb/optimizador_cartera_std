@echo off
setlocal EnableExtensions

REM ============================================
REM  SMOKE TEST BANCO L1.5
REM ============================================

REM --- Python fijo ---
set "PY_EXE=C:\Users\josef\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY_EXE%" (
  echo ERROR: No se encuentra Python en:
  echo   %PY_EXE%
  pause
  exit /b 1
)

REM --- Root del proyecto ---
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

REM --- PYTHONPATH ---
set "PYTHONPATH=%ROOT_DIR%"

set "LOG_MAIN=%ROOT_DIR%logs\main.log"
set "LOG_COORD=%ROOT_DIR%logs\coordinator_inference.log"

echo =======================================================
echo   SMOKE TEST BANCO L1.5
echo   Root:   %ROOT_DIR%
echo   Python: %PY_EXE%
echo =======================================================
echo.

REM --- Verifica main.py ---
if not exist "%ROOT_DIR%main.py" (
  echo ERROR: No se encuentra main.py en %ROOT_DIR%
  goto error
)

REM --- Verifica dependencias base ---
echo Verificando dependencias (numpy, pandas, openpyxl, torch, gymnasium, stable_baselines3)...
"%PY_EXE%" -c "import numpy,pandas,openpyxl,torch,gymnasium,stable_baselines3; print('OK')"
if errorlevel 1 goto error
echo OK: entorno verificado.
echo.

REM ------------------------------------------------
REM [1/4] Generar cartera sintética pequeña
REM ------------------------------------------------
set "PORTFOLIO=%ROOT_DIR%data\portfolio_synth_smoke.xlsx"
echo [1/4] Generando cartera sintetica de 200 prestamos...
"%PY_EXE%" "%ROOT_DIR%main.py" generate --n 200 --out "%PORTFOLIO%"
if errorlevel 1 goto error

REM ------------------------------------------------
REM [2/4] Entrenamiento smoke
REM ------------------------------------------------
echo [2/4] Entrenamiento RL smoke (both, 10k steps)...
"%PY_EXE%" "%ROOT_DIR%main.py" train --agent both --portfolio "%PORTFOLIO%" --total-steps 10000 --top-k 3 --scenario baseline --device cpu
if errorlevel 1 goto error

REM ------------------------------------------------
REM Resolver modelos + VecNormalize
REM ------------------------------------------------
set "MODEL_MICRO="
set "MODEL_MACRO="
set "VN_MICRO="
set "VN_MACRO="
set "VN_LOAN="

REM Micro model (preferente)
if exist "%ROOT_DIR%models\best_model_loan.zip" set "MODEL_MICRO=%ROOT_DIR%models\best_model_loan.zip"
if not defined MODEL_MICRO if exist "%ROOT_DIR%models\best_model.zip" set "MODEL_MICRO=%ROOT_DIR%models\best_model.zip"

REM Macro model (preferente)
if exist "%ROOT_DIR%models\best_model_portfolio.zip" set "MODEL_MACRO=%ROOT_DIR%models\best_model_portfolio.zip"
if not defined MODEL_MACRO if exist "%ROOT_DIR%models\best_model_macro.zip" set "MODEL_MACRO=%ROOT_DIR%models\best_model_macro.zip"

REM VecNormalize micro
if exist "%ROOT_DIR%models\vecnormalize_loan.pkl" set "VN_MICRO=%ROOT_DIR%models\vecnormalize_loan.pkl"
if not defined VN_MICRO if exist "%ROOT_DIR%models\vecnormalize_final.pkl" set "VN_MICRO=%ROOT_DIR%models\vecnormalize_final.pkl"

REM VecNormalize macro
if exist "%ROOT_DIR%models\vecnormalize_portfolio.pkl" set "VN_MACRO=%ROOT_DIR%models\vecnormalize_portfolio.pkl"
if not defined VN_MACRO if exist "%ROOT_DIR%models\vecnormalize_final.pkl" set "VN_MACRO=%ROOT_DIR%models\vecnormalize_final.pkl"

REM VN_LOAN (re-ranking)
if defined VN_MICRO set "VN_LOAN=%VN_MICRO%"

REM Validación mínima
if not defined MODEL_MICRO (
  echo ERROR: No se encuentra modelo MICRO. Buscados:
  echo   models\best_model_loan.zip  o  models\best_model.zip
  goto error
)
if not defined MODEL_MACRO (
  echo ERROR: No se encuentra modelo MACRO. Buscados:
  echo   models\best_model_portfolio.zip  o  models\best_model_macro.zip
  goto error
)

echo Modelos detectados:
echo   MICRO: %MODEL_MICRO%
echo   MACRO: %MODEL_MACRO%
echo   VN_MICRO: %VN_MICRO%
echo   VN_MACRO: %VN_MACRO%
echo   VN_LOAN : %VN_LOAN%
echo.

REM ------------------------------------------------
REM [3/4] Inferencia coordinada MULTI-POSTURA + AUDIT
REM ------------------------------------------------
echo [3/4] Inferencia COORDINADA multi-postura + AUDIT CSV + KEEP-ALL...
echo Ejecutando infer con: --n-steps 10 --top-k 5
echo.

REM CLAVE: pasar VN directamente y SIEMPRE entre comillas (aunque esten vacias).
"%PY_EXE%" "%ROOT_DIR%main.py" infer ^
  --all-postures ^
  --model-micro "%MODEL_MICRO%" ^
  --model-macro "%MODEL_MACRO%" ^
  --portfolio "%PORTFOLIO%" ^
  --tag "smoke_coordinator" ^
  --n-steps 10 ^
  --top-k 5 ^
  --deterministic ^
  --keep-all ^
  --export-audit-csv ^
  --device cpu ^
  --vn-micro "%VN_MICRO%" ^
  --vn-macro "%VN_MACRO%" ^
  --vn-loan "%VN_LOAN%"

if errorlevel 1 goto error

REM ------------------------------------------------
REM [4/4] Summary
REM ------------------------------------------------
echo.
echo [4/4] Generando resumen ejecutivo (si aplica)...
"%PY_EXE%" "%ROOT_DIR%main.py" summary --source "%ROOT_DIR%reports\runs"
if errorlevel 1 goto error

echo.
echo Smoke test completado OK.
echo.
echo IMPORTANTE:
echo - EXCELs finales:  reports\runs\YYYYMMDD_HHMMSS_smoke_coordinator_DELIVERABLE\
echo - AUDIT CSV:      reports\runs\YYYYMMDD_HHMMSS_smoke_coordinator_MULTI\{postura}\
echo - Logs:           logs\
goto end

:error
echo.
echo ERROR: Ha ocurrido un fallo en alguno de los pasos.
echo Revisa logs en: %ROOT_DIR%logs
echo.

REM Tail rapido de logs (si existen)
if exist "%LOG_MAIN%" (
  echo -------- Tail main.log --------
  powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG_MAIN%' -Tail 80"
)
if exist "%LOG_COORD%" (
  echo -------- Tail coordinator_inference.log --------
  powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG_COORD%' -Tail 120"
)

goto end

:end
echo.
echo Pulsa una tecla para cerrar...
pause >nul
endlocal
exit /b 0
