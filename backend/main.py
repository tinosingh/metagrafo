"""Main module for the FastAPI backend."""

# Standard library imports
import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler

# Set up logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_file = '../logs/backend.log'

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10MB per file, keep 5 backups

# Set levels
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add to handlers
formatter = logging.Formatter(log_format)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Use logger instead of print
logger.info('Starting backend server...')

# Third-party imports
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
import mlx.core as mx  # For setting MLX device (important for Apple Silicon)

# Initialize FastAPI app (THIS IS THE 'app' ATTRIBUTE UVICORN LOOKS FOR)
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set MLX to use GPU (Moved from app.py, essential for Apple Silicon)
# This ensures MLX computations leverage the Apple Silicon GPU.
mx.set_default_device(mx.Device(mx.DeviceType.gpu))

# Suppress excessive logging from mlx_whisper (Moved from app.py)
# This prevents overly verbose output from the MLX Whisper library.
logging.getLogger("mlx_whisper").setLevel(logging.WARNING)

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns:
        dict: Status of the API and WebSocket connections.
    """
    # Provides a simple status check for both the API and WebSocket connections.
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "details": {
            "api": "running",
        },
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Simplified transcription endpoint"""
    contents = await file.read()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(contents)
        temp_path = temp.name
    
    try:
        # Transcribe
        result = pipe(temp_path)
        return {"text": result["text"]}
    finally:
        # Clean up
        os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application using Uvicorn.
    # Make sure this host and port match your frontend's API_URL configuration.
    uvicorn.run(app, host="0.0.0.0", port=9001)
