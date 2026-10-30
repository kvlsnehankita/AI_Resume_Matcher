@echo off
echo ========================================================
echo Starting AI Career Studio - FastAPI Backend (Port 8000)
echo ========================================================
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
) else if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
