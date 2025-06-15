#!/bin/bash

# --- Start Script ---
# This script starts the backend and frontend services and verifies their health.

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
PIDS_DIR="$PROJECT_ROOT/pids"

# Create log and pid directories if they don't exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"

# Clean previous logs
> "$LOGS_DIR/backend.log"
> "$LOGS_DIR/frontend.log"

# Activate the correct pyenv virtual environment
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
fi

if [ -f "$PROJECT_ROOT/whisper/bin/activate" ]; then
    source "$PROJECT_ROOT/whisper/bin/activate"
    echo "✅ Activated 'whisper' virtual environment."
else
    echo "❌ ERROR: Could not find 'whisper' virtual environment at $PROJECT_ROOT/whisper/bin/activate."
    echo "Please ensure it is created and named 'whisper' and located correctly relative to the script."
    exit 1
fi


# --- Health Check Function ---
check_service_health() {
    local service_name=$1
    local url=$2
    local timeout=$3
    
    echo -n "Verifying $service_name health at $url"
    
    end_time=$((SECONDS + timeout))
    is_healthy=false
    
    while [ $SECONDS -lt $end_time ]; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
        if [ "$status_code" -eq 200 ]; then
            is_healthy=true
            break
        fi
        echo -n "."
        sleep 2
    done
    
    echo # Newline
    if $is_healthy; then
        echo "✅ $service_name is up and running."
    else
        echo "❌ ERROR: $service_name failed to start within $timeout seconds. Check logs for details."
        # Optionally, exit with an error code if a service fails to start
        # exit 1
    fi
}

echo "🚀 Starting services..."

# --- Start Backend ---
echo "Starting backend service..."
(cd "$PROJECT_ROOT" && uvicorn backend.main:app --host 0.0.0.0 --port 9001 --reload > "$LOGS_DIR/backend.log" 2>&1 &)
BACKEND_PID=$!
echo $BACKEND_PID > "$PIDS_DIR/backend.pid"
echo "Backend service started with PID: $BACKEND_PID. Log: $LOGS_DIR/backend.log"

# --- Start Frontend ---
echo "Starting frontend service..."
# Corrected: Removed duplicate --port 9000
(cd "$PROJECT_ROOT/frontend" && npm run dev --port 9000 > "$LOGS_DIR/frontend.log" 2>&1 &)
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PIDS_DIR/frontend.pid"
echo "Frontend service started with PID: $FRONTEND_PID. Log: $LOGS_DIR/frontend.log"

# --- Verify Services ---
echo -e "\n--- Verifying Service Health ---"
sleep 5 
check_service_health "Backend" "http://localhost:9001/health" 30
check_service_health "Frontend" "http://localhost:9000" 30

echo -e "\n✅ Service startup sequence complete."
