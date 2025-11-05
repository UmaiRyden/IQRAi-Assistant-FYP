@echo off
REM IQRAi Backend Startup Script for Windows

echo 🚀 Starting IQRAi FastAPI Backend...

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from env.example...
    copy env.example .env
    echo 📝 Please edit .env and add your API keys before continuing.
    exit /b 1
)

REM Start the server
echo ✅ Starting server on port 8000...
uvicorn app.main:app --reload --port 8000

