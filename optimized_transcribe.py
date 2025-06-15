"""Optimized real-time transcription with MLX memory management"""

import sounddevice as sd
import numpy as np
import mlx.core as mx
from backend.transcription import transcribe_audio
import asyncio
import time

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 16000  # 1-second chunks
MAX_DURATION = 30  # seconds


class AudioProcessor:
    def __init__(self):
        self.buffer = mx.array([], dtype=mx.float32)
        self.last_process_time = time.time()

    async def process_chunk(self, chunk_np):
        """Convert numpy to MLX array and transcribe"""
        try:
            # Convert to MLX array with proper memory management
            chunk_mlx = mx.array(chunk_np, dtype=mx.float32)

            # Process through Whisper
            result = await transcribe_audio(audio_path=chunk_mlx, model_size="base")
            print(f"\nPartial: {result['text']}")
            return result
        except Exception as e:
            print(f"\nProcessing error: {str(e)}")
            return None


def main():
    print("Optimized MLX Whisper Transcription\n")
    print(f"Sample rate: {SAMPLE_RATE}Hz | Chunk size: {CHUNK_SIZE / SAMPLE_RATE:.1f}s")

    processor = AudioProcessor()

    def callback(indata, frames, time_info, status):
        """Audio callback with memory-safe processing"""
        if status:
            print(f"Audio status: {status}")

        # Visual feedback
        level = min(30, int(np.max(np.abs(indata)) * 50))
        print("■" * level + " " * (30 - level), end="\r")

        # Process chunk
        asyncio.run(processor.process_chunk(indata))

    print("Press Enter to start recording...")
    input()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, dtype="float32", callback=callback
    ):
        print(f"Recording (max {MAX_DURATION}s) - Press Enter to stop")
        input()

    print("\nProcessing complete")


if __name__ == "__main__":
    main()
