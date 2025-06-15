#!/bin/bash

# Kill frontend (port 9000)
lsof -ti :9000 | xargs kill -9

# Kill backend (port 9001)
lsof -ti :9001 | xargs kill -9

# Cleanup GPU cache if needed
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --gpu-reset
fi

echo "Services stopped gracefully"
