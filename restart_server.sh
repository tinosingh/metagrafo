#!/bin/bash

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PIDS_DIR="$PROJECT_ROOT/pids"
MAX_LOG_FILES=5
VENV_NAME="whisper"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"

# --- Setup Logging ---
mkdir -p "$LOG_DIR"
mkdir -p "$PIDS_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

# --- Cleanup old logs (macOS compatible) ---
log_cleanup() {
  # List files by modification time, newest first, then skip first MAX_LOG_FILES files
  find "$LOG_DIR" -maxdepth 1 -name "server_*.log" -type f -exec stat -f "%m %N" {} \; | \
    sort -rn | cut -d' ' -f2- | tail -n +$((MAX_LOG_FILES + 1)) | xargs -I{} rm -f {}
}

log() {
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# --- Check if command exists ---
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# --- Activate virtual environment ---
if [ -f "$PROJECT_ROOT/$VENV_NAME/bin/activate" ]; then
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    log "${GREEN}✅ Activated '$VENV_NAME' virtual environment.${NC}"
else
    log "${RED}❌ ERROR: Could not activate the '$VENV_NAME' virtual environment.${NC}"
    log "${RED}Please ensure it is created and named '$VENV_NAME'.${NC}"
    exit 1
fi

# --- Install/Update backend dependencies ---
log "${YELLOW}Installing/Updating backend dependencies...${NC}"
cd "$PROJECT_ROOT/$BACKEND_DIR" || { log "${RED}Failed to enter backend directory${NC}"; exit 1; }
if [ -f "requirements.txt" ]; then
    if command_exists uv; then
        uv pip install -r requirements.txt || { log "${RED}Failed to install backend dependencies with uv.${NC}"; exit 1; }
    else
        pip install -r requirements.txt || { log "${RED}Failed to install backend dependencies with pip.${NC}"; exit 1; }
    fi
else
    log "${YELLOW}No requirements.txt found in backend directory. Skipping backend pip install.${NC}"
fi
cd "$PROJECT_ROOT" || exit 1

# --- Function to get current timestamp ---
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

# --- Function to stop processes on a given port ---
stop_port() {
  local port="$1"
  log "${YELLOW}Stopping processes on port $port...${NC}"
  lsof -ti :"$port" | xargs -r kill -9 2>/dev/null
}

# --- Start backend server ---
start_backend() {
  stop_port 9001
  
  log "${YELLOW}Starting MLX-Whisper backend on port 9001...${NC}"
  cd "$PROJECT_ROOT/$BACKEND_DIR" || return 1
  BACKEND_LOG="$LOG_DIR/backend_$(timestamp).log"
  nohup uvicorn main:app --host 0.0.0.0 --port 9001 --reload > "$BACKEND_LOG" 2>&1 &
  BACKEND_PID=$!
  echo $BACKEND_PID > "$PIDS_DIR/backend.pid"
  log "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
  log "${YELLOW}Backend log: $BACKEND_LOG${NC}"
  cd - >/dev/null || return 1
}

# --- Start frontend server ---
start_frontend() {
  if [ ! -d "$PROJECT_ROOT/$FRONTEND_DIR" ]; then
    log "${YELLOW}Frontend directory not found, skipping frontend start.${NC}"
    return 0
  fi
  
  stop_port 9000
  
  log "${YELLOW}Starting frontend server...${NC}"
  cd "$FRONTEND_DIR"
  FRONTEND_LOG="$LOG_DIR/frontend_$(timestamp).log"
  nohup npm run dev -- --port 9000 > "../$FRONTEND_LOG" 2>&1 &
  FRONTEND_PID=$!
  echo $FRONTEND_PID > "$PIDS_DIR/frontend.pid"
  log "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
  log "${YELLOW}Frontend log: $FRONTEND_LOG${NC}"
  cd - >/dev/null || return 1
}

# --- Cleanup function ---
cleanup() {
  log "${YELLOW}Initiating cleanup...${NC}"
  
  # Use stop script if it exists
  if [ -f "$PROJECT_ROOT/stop_server.sh" ]; then
    "$PROJECT_ROOT/stop_server.sh"
  fi
  
  # Ensure processes are killed
  for port in 9000 9001; do
    if lsof -ti :$port >/dev/null 2>&1; then
      log "${YELLOW}Killing processes on port $port...${NC}"
      lsof -ti :$port | xargs -r kill -9 2>/dev/null
    fi
  done
  
  # Clean up PID files
  rm -f "$PIDS_DIR/backend.pid" "$PIDS_DIR/frontend.pid"
  
  log "${GREEN}Cleanup complete.${NC}"
  exit 0
}

# --- Main execution ---
log_cleanup

# Set up trap for clean exit
trap cleanup SIGINT SIGTERM EXIT

log "${GREEN}Restarting servers...${NC}"

# Stop any running instances
stop_port 9000
stop_port 9001
sleep 2

# Start services
start_backend
sleep 2  # Give backend a moment to start
start_frontend

log "${GREEN}Servers started successfully!${NC}"
log "${YELLOW}Backend: http://localhost:9001${NC}"
log "${YELLOW}Frontend: http://localhost:9000${NC}"
log "${YELLOW}Logs: $LOG_FILE${NC}"

# Keep script running
wait
