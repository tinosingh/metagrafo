"""Test script for MLX transcription with real audio."""

import asyncio

from mlx_transcribe import transcribe


async def test():
    try:
        print("Testing transcription")
        result = await transcribe("../frontend/punainen_linnake.mp3")

        # Handle different possible return types
        if isinstance(result, str):
            print(f"Transcription: {result[:100]}...")
        elif isinstance(result, dict) and "text" in result:
            print(f"Transcription: {result['text'][:100]}...")
        else:
            print(f"Raw result: {str(result)[:100]}...")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test())
