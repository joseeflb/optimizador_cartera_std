@echo off
echo ══════════════════════════════════════════════════════
echo   NPL Portfolio Optimizer — Lovable Matrix Dashboard
echo ══════════════════════════════════════════════════════
echo.
echo Arrancando servidor en http://localhost:3000 ...
echo.
cd /d "%~dp0.."
py dashboard\api_server.py
pause
