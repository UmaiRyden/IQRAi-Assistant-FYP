#!/bin/bash

# IQRAi Backend Startup Script

echo "🚀 Starting IQRAi FastAPI Backend..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from env.example..."
    cp env.example .env
    echo "📝 Please edit .env and add your API keys before continuing."
    exit 1
fi

# Start the server
echo "✅ Starting server on port 8000..."
uvicorn app.main:app --reload --port 8000

