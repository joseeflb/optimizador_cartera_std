@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call run_inference_only.bat %*
