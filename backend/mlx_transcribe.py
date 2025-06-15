"""MLX Whisper transcription with optional WebSocket progress."""

import mlx.core as mx
import mlx_whisper
from typing import Optional

# WebSocket progress is optional
try:
    from unittest.mock import patch
    from .websocket_manager import TqdmProgressWrapper

    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


async def transcribe(audio_path: str, client_id: Optional[str] = None) -> str:
    """Transcribe audio with optional WebSocket progress."""
    try:
        mx.set_default_device(mx.Device(mx.DeviceType.gpu))

        if client_id and HAS_WEBSOCKET:
            import asyncio

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

        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
