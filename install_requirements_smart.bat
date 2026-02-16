@echo off
set PYTHON=py
echo ========================================
echo 🚀 Verificando dependencias del proyecto
echo ========================================
%PYTHON% "%~dp0install_requirements_smart.py"
echo.
pause
