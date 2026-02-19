@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
(echo. | echo.) | call run_3_postures_executability.bat %*
