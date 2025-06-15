import os
import tempfile
import subprocess


def preprocess_audio(audio_path: str) -> str:
    """
    Converts audio to 16kHz mono WAV format using ffmpeg.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        str: Path to the processed temporary WAV file.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        subprocess.CalledProcessError: If ffmpeg fails during processing.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    fd, processed_path = tempfile.mkstemp(suffix="_processed.wav")
    os.close(fd)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                processed_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg error: {' '.join(e.cmd)}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        ) from e

    return processed_path
