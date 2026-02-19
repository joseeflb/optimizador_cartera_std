@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
(echo.) | call smoke_test.bat %*
