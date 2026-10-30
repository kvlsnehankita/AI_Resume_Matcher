@echo off
echo ========================================================
echo Starting AI Career Studio (Backend + Frontend)
echo ========================================================
start "AI Resume Backend (FastAPI)" cmd /k "%~dp0run_backend.bat"
start "AI Resume Frontend (React)" cmd /k "%~dp0run_frontend.bat"
echo Applications launched!
echo - Backend API:  http://127.0.0.1:8000
echo - Swagger Docs: http://127.0.0.1:8000/docs
echo - Web Frontend: http://localhost:5173
pause
