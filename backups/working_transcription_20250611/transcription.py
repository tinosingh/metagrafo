"""Audio transcription service using MLX Whisper."""

import asyncio
import logging
import os
from typing import Optional
from unittest.mock import patch

import mlx.core as mx
import mlx_whisper

from .websocket_manager import TqdmProgressWrapper, WebSocketManager

logger = logging.getLogger(__name__)
manager = WebSocketManager()


# Clean transcription text
def clean_transcription(text: str) -> str:
    """Clean and normalize transcription text."""
    return text.strip()


async def transcribe_audio(client_id: Optional[str], audio_path: str) -> str:
    """
    Transcribe audio using MLX Whisper.
    Args:
        client_id: Optional WebSocket client ID for progress updates
        audio_path: Path to audio file to transcribe
    Returns:
        Cleaned transcription text
    """
    try:
        mx.set_default_device(mx.Device(mx.DeviceType.gpu))

        if client_id:
            current_loop = asyncio.get_running_loop()
            with patch(
                "tqdm.tqdm",
                new=lambda *args, **kwargs: TqdmProgressWrapper(
                    client_id, current_loop, *args, **kwargs
                ),
            ):
                result = mlx_whisper.transcribe(audio_path)
        else:
            result = mlx_whisper.transcribe(audio_path)

        return clean_transcription(result["text"])

    except Exception as e:
        logger.exception("Transcription failed")
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        # Clean up temporary files if any were created
        if "processed_path" in locals() and os.path.exists(processed_path):
            os.remove(processed_path)
