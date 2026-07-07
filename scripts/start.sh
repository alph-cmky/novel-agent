#!/bin/bash
# Start novel-agent in production mode
# Build frontend and serve via FastAPI + uvicorn

set -e

echo "Building frontend..."
cd "$(dirname "$0")/../frontend"
npm install --silent
npm run build

echo "Starting server..."
cd ..
uv run uvicorn novel_agent.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
