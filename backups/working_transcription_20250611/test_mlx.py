"""Test script for MLX transcription with real audio."""

import asyncio

from mlx_transcribe import transcribe


async def test():
    try:
        text = await transcribe("../frontend/punainen_linnake.mp3")
        print(f"Success! Transcription: {text}")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test())
