#!/bin/bash

# Kill existing
pkill -f "uvicorn|npm" && sleep 1

# Start backend with absolute paths
nohup /Users/tinosingh/.pyenv/versions/whisper/bin/uvicorn backend.main:app --port 9001 --reload >> backend.log 2>&1 &

# Start frontend 
nohup bash -c "cd frontend && npm run dev" >> frontend.log 2>&1 &
disown -a

echo "Services started detached. Logs: backend.log and frontend.log"
