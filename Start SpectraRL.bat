@echo off
cd /d "%~dp0"
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8000"
.venv\Scripts\python.exe run.py
pause