#!/bin/bash

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$PROJECT_ROOT/pids"
LOG_DIR="$PROJECT_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Log function
log() {
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/stop_server.log"
}

# Kill process by PID file
kill_by_pid_file() {
  local service=$1
  local pid_file="$PIDS_DIR/${service}.pid"
  
  if [ ! -f "$pid_file" ]; then
    log "${YELLOW}No PID file found for $service${NC}"
    return 1
  fi
  
  local pid=$(cat "$pid_file")
  if [ -z "$pid" ]; then
    log "${YELLOW}Empty PID file for $service${NC}"
    return 1
  fi
  
  if ps -p "$pid" > /dev/null 2>&1; then
    log "${YELLOW}Stopping $service (PID: $pid)...${NC}"
    if kill "$pid" 2>/dev/null; then
      log "${GREEN}Successfully stopped $service (PID: $pid)${NC}"
    else
      log "${RED}Failed to stop $service (PID: $pid)${NC}"
      return 1
    fi
  else
    log "${YELLOW}Process $service (PID: $pid) not running${NC}"
  fi
  
  # Remove PID file
  rm -f "$pid_file"
}

# Kill processes by port
kill_by_port() {
  local port=$1
  local pids
  
  if ! command -v lsof >/dev/null 2>&1; then
    log "${YELLOW}lsof not found, cannot check port $port${NC}"
    return 1
  fi
  
  pids=$(lsof -ti ":$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    log "${YELLOW}Killing processes on port $port...${NC}"
    echo "$pids" | xargs -r kill -9 2>/dev/null
    log "${GREEN}Stopped processes on port $port${NC}"
  fi
}

# Main execution
log "${GREEN}Stopping services...${NC}"

# Stop by PID files
kill_by_pid_file "backend"
kill_by_pid_file "frontend"

# Force kill any remaining processes on our ports
kill_by_port 9000
kill_by_port 9001

# Clean up any remaining PID files
rm -f "$PIDS_DIR/"*.pid

log "${GREEN}All services stopped${NC}"
