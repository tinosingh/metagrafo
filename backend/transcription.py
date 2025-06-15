"""Audio transcription service using MLX Whisper."""

import os
import asyncio
import logging
from typing import Optional
from unittest.mock import patch
import tempfile

import mlx.core as mx
import mlx_whisper
import whisper  # For model loading

from .websocket_manager import WebSocketManager, TqdmProgressWrapper

logger = logging.getLogger(__name__)
manager = WebSocketManager()

# Supported model sizes
WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
}

# Model cache
_loaded_models = {}


# Clean transcription text
def clean_transcription(text: str) -> str:
    """Clean and normalize transcription text."""
    return text.strip()


def load_model(model_size: str):
    """Load and cache Whisper model."""
    if model_size not in WHISPER_MODELS:
        raise ValueError(f"Unsupported model size: {model_size}")

    if model_size not in _loaded_models:
        logger.info(f"Loading Whisper model: {model_size}")
        _loaded_models[model_size] = whisper.load_model(WHISPER_MODELS[model_size])

    return _loaded_models[model_size]


async def transcribe_audio(
    audio_path: str,
    model_size: str = "tiny",
    temperature: float = 0.0,
    language: Optional[str] = None,
    word_timestamps: bool = False,
    verbose: bool = False,
    prompt: Optional[str] = None,  # Context for the model
    client_id: Optional[str] = None,
) -> dict:
    """Enhanced transcription with context support."""
    try:
        mx.set_default_device(mx.Device(mx.DeviceType.gpu))
        model = load_model(model_size)

        options = {
            "temperature": temperature,
            "language": language,
            "verbose": verbose,
            "word_timestamps": word_timestamps,
            "prompt": prompt,  # Pass context to Whisper
        }

        processed_path = os.path.join(tempfile.gettempdir(), "processed_audio.wav")

        if client_id:
            current_loop = asyncio.get_running_loop()
            with patch(
                "tqdm.tqdm",
                new=lambda *args, **kwargs: TqdmProgressWrapper(
                    client_id, current_loop, *args, **kwargs
                ),
            ):
                result = mlx_whisper.transcribe(
                    model=model,
                    audio=audio_path,  # Single audio argument
                    **options,  # All other options as kwargs
                )
        else:
            result = mlx_whisper.transcribe(
                model=model,
                audio=audio_path,  # Single audio argument
                **options,  # All other options as kwargs
            )

        return {
            "text": clean_transcription(result["text"]),
            "language": result.get("language", ""),
            "segments": result.get("segments", []),
            "duration": result.get("duration", 0),
        }

    except Exception as e:
        logger.exception(f"Transcription failed with model {model_size}")
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        # Clean up temporary files if any were created
        if os.path.exists(processed_path):
            os.remove(processed_path)
