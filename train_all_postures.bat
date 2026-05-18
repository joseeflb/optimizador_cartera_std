@echo off
REM ============================================================
REM  Entrena los 6 modelos: 3 posturas x (loan + portfolio)
REM  Uso:  train_all_postures.bat [TS_LOAN] [TS_PORTFOLIO] [PORTFOLIO_FILE]
REM  Defaults: 500000 / 200000 / data\portfolio_synth.xlsx
REM ============================================================
setlocal ENABLEDELAYEDEXPANSION

set TS_LOAN=%1
if "%TS_LOAN%"=="" set TS_LOAN=500000

set TS_PORTFOLIO=%2
if "%TS_PORTFOLIO%"=="" set TS_PORTFOLIO=200000

set PORTFOLIO_FILE=%3
if "%PORTFOLIO_FILE%"=="" set PORTFOLIO_FILE=data\portfolio_synth.xlsx

set LOG_DIR=logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Resolver Python: priorizar venv local, sino PYTHON_EXE env, sino python en PATH
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
echo  Entrenamiento full-postures
echo  TS_LOAN=%TS_LOAN%  TS_PORTFOLIO=%TS_PORTFOLIO%
echo  PORTFOLIO_FILE=%PORTFOLIO_FILE%
echo  Inicio: %DATE% %TIME%
echo ============================================================

for %%P in (prudente balanceado desinversion) do (
  echo.
  echo ====== POSTURA %%P ======
  echo [%TIME%] MICRO loan %%P
  "%PYBIN%" -m agent.train_subagents --agent loan ^
    --portfolio "%PORTFOLIO_FILE%" ^
    --total-steps-loan %TS_LOAN% ^
    --posture %%P --no-legacy
  if errorlevel 1 (
    echo [ERR] Fallo training MICRO posture=%%P
    exit /b 1
  )

  echo [%TIME%] MACRO portfolio %%P
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
echo  Entrenamiento full-postures completado
echo  Inicio : %START_TIME%
echo  Fin    : %TIME%
echo ============================================================
endlocal
