"""Utility functions for audio processing and device management."""

# Standard library imports
import logging
import os
import subprocess
from pathlib import Path

# Third-party imports
import mlx.core as mx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def get_optimal_device():
    """
    Determines the optimal device for MLX computation.

    Returns:
        mx.Device: The optimal device (GPU if available, otherwise CPU).
    """
    try:
        # Check for Apple Silicon GPU
        if mx.metal.is_available():
            return mx.gpu
    except AttributeError:
        logging.warning("mx.metal module not available. Falling back to CPU.")
    return mx.cpu


def validate_audio_file(audio_path: str) -> None:
    """
    Validate audio file existence and size.

    Args:
        audio_path (str): Path to the audio file

    Raises:
        HTTPException: If file doesn't exist or is too small
    """
    if not os.path.exists(audio_path):
        logging.error("Audio file not found: %s", audio_path)
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {os.path.basename(audio_path)}",
        )

    if os.path.getsize(audio_path) < 1024:  # 1KB
        logging.error("Audio file too small: %s", audio_path)
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too small: {os.path.basename(audio_path)}",
        )


def preprocess_audio(audio_path: str, output_dir: str) -> str:
    """
    Convert audio to 16kHz WAV format.

    Args:
        audio_path (str): Path to input audio file
        output_dir (str): Directory for processed audio

    Returns:
        str: Path to processed WAV file

    Raises:
        HTTPException: If processing fails
    """
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Generate output path
        output_path = os.path.join(output_dir, Path(audio_path).stem + "_processed.wav")

        # FFmpeg conversion command
        command = [
            "ffmpeg",
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            output_path,
        ]

        # Execute conversion
        subprocess.run(command, check=True, capture_output=True)
        logging.info("Audio processed successfully: %s", output_path)
        return output_path

    except subprocess.CalledProcessError as e:
        logging.error("Audio processing failed: %s", e.stderr.decode())
        raise HTTPException(
            status_code=500, detail=f"Audio processing failed: {e.stderr.decode()}"
        ) from e
    except FileNotFoundError as e:
        logging.error("FFmpeg not installed")
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed. This is a required dependency.",
        ) from e
