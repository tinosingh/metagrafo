#!/bin/bash

# Kill existing process
pkill -f "uvicorn.*9001" && sleep 1

# Activate virtual environment
source whisper/bin/activate

# Run the server
cd backend
python -m uvicorn main:app --reload
