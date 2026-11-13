@echo off
REM DocAudit web UI (Windows). Double-click or run: start_ui.bat
REM Uses the Python on PATH, or set PYTHON=C:\venv\da\Scripts\python.exe first.
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" -m src.ui.app %*
pause
