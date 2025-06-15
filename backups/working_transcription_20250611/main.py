"""Main module for the FastAPI backend."""

# Standard library imports
import logging
import os
import tempfile

# Third-party imports
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    File,
    Form,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
import mlx.core as mx  # For setting MLX device (important for Apple Silicon)

# Absolute imports from backend package
from backend.transcription import (
    transcribe_audio,
    ALL_WHISPER_MODELS,  # Correctly import ALL_WHISPER_MODELS
)
from backend.websocket_manager import manager as ws_manager

# Initialize FastAPI app (THIS IS THE 'app' ATTRIBUTE UVICORN LOOKS FOR)
app = FastAPI()

# Set MLX to use GPU (Moved from app.py, essential for Apple Silicon)
# This ensures MLX computations leverage the Apple Silicon GPU.
mx.set_default_device(mx.Device(mx.DeviceType.gpu))

# Suppress excessive logging from mlx_whisper (Moved from app.py)
# This prevents overly verbose output from the MLX Whisper library.
logging.getLogger("mlx_whisper").setLevel(logging.WARNING)


# CORS Configuration
# Allows requests from the frontend origin (http://localhost:9000) to access this backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Ensure this matches your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers in requests
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns:
        dict: Status of the API and WebSocket connections.
    """
    # Provides a simple status check for both the API and WebSocket connections.
    return {
        "status": "healthy",
        "details": {
            "api": "running",
            "websocket": "active"
            if ws_manager.active_connections_count
            else "inactive",
        },
    }


@app.get("/models")
async def get_models():
    """
    Returns available Whisper models to the frontend.
    The model names are retrieved from the centralized ALL_WHISPER_MODELS dictionary
    defined in the transcription module.
    """
    # The ALL_WHISPER_MODELS dictionary now contains simple names like "tiny": "tiny".
    # We return the keys as model identifiers and values as display names (which are the same for now).
    return {"models": {key: ALL_WHISPER_MODELS[key] for key in ALL_WHISPER_MODELS}}


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    client_id: str = Form(None),
):
    """Simplified transcription endpoint"""
    try:
        # Validate file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(400, "Invalid file type")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Transcribe
        transcription = await transcribe_audio(client_id, tmp_path)
        return {"transcription": transcription}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            message = await websocket.receive_text()
            await ws_manager.handle_message(client_id, message)  # This line

    except WebSocketDisconnect:
        logging.info(f"Client {client_id} disconnected from WebSocket.")
    except Exception as e:
        logging.error(f"WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        await ws_manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application using Uvicorn.
    # Make sure this host and port match your frontend's API_URL configuration.
    uvicorn.run(app, host="0.0.0.0", port=9001)
