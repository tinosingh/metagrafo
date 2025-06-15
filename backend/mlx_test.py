"""Minimal test for MLX Whisper API"""

import mlx_whisper

# Test with silent audio file
try:
    result = mlx_whisper.transcribe(
        "test_audio.wav",
        path_or_hf_repo="mlx-community/whisper-tiny",  # Full HF path
        verbose=True,
    )
    print("Test successful!")
    print(result)
except Exception as e:
    print(f"Test failed: {str(e)}")
