@echo off
REM Reentrenar SOLO MACRO PRUDENTE y BALANCEADO (no toca MICRO ni DESINVERSION).
REM Uso: train_macro_pru_bal.bat [TS_PORTFOLIO] [PORTFOLIO_FILE]
setlocal ENABLEDELAYEDEXPANSION

set TS_PORTFOLIO=%1
if "%TS_PORTFOLIO%"=="" set TS_PORTFOLIO=200000
set PORTFOLIO_FILE=%2
if "%PORTFOLIO_FILE%"=="" set PORTFOLIO_FILE=data\portfolio_synth.xlsx

if not exist "logs" mkdir "logs"

if defined PYTHON_EXE (
  set PYBIN=%PYTHON_EXE%
) else if exist ".venv\Scripts\python.exe" (
  set PYBIN=.venv\Scripts\python.exe
) else (
  set PYBIN=python
)
echo Usando Python: %PYBIN%

set START_TIME=%TIME%
echo ============================================================
echo  Reentrenamiento MACRO PRU+BAL (family bonuses)
echo  TS_PORTFOLIO=%TS_PORTFOLIO%
echo  Inicio: %DATE% %TIME%
echo ============================================================

for %%P in (prudente balanceado) do (
  echo.
  echo ====== MACRO %%P ======
  echo [%TIME%] portfolio %%P
  "%PYBIN%" -m agent.train_subagents --agent portfolio ^
    --portfolio "%PORTFOLIO_FILE%" ^
    --total-steps-portfolio %TS_PORTFOLIO% ^
    --posture %%P --no-legacy
  if errorlevel 1 (
    echo [ERR] Fallo training MACRO posture=%%P
    exit /b 1
  )
)

echo ============================================================
echo  Reentrenamiento MACRO PRU+BAL completado
echo  Inicio : %START_TIME%
echo  Fin    : %TIME%
echo ============================================================
endlocal
