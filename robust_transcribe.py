"""Robust real-time transcription with proper MLX memory handling"""

import asyncio
import time

import mlx.core as mx
import numpy as np
import sounddevice as sd

from backend.transcription import transcribe_audio


class AudioTranscriber:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.sample_rate = 16000
        self.chunk_size = 16000  # 1-second chunks
        self.max_duration = 30  # seconds

    async def process_chunk(self, audio_np):
        """Safely process audio chunk with MLX"""
        try:
            # Convert to MLX array with explicit memory management
            audio_mlx = mx.array(audio_np.ravel(), dtype=mx.float32)

            # Transcribe with error handling
            result = await transcribe_audio(
                audio_path=audio_mlx, model_size=self.model_size
            )
            print(f"\nPartial: {result['text']}")
            return result
        except Exception as e:
            print(f"\nError processing chunk: {str(e)}")
            return None

    def run(self):
        print(f"MLX Whisper Transcription (model: {self.model_size})\n")
        print(
            f"Sample rate: {self.sample_rate}Hz | Chunk size: {self.chunk_size / self.sample_rate:.1f}s"
        )

        def callback(indata, frames, time_info, status):
            """Audio callback with visualization"""
            if status:
                print(f"\nAudio status: {status}")

            # Visual feedback
            level = min(30, int(np.max(np.abs(indata)) * 50))
            print("■" * level + " " * (30 - level), end="\r")

            # Process chunk in background
            asyncio.create_task(self.process_chunk(indata))

        print("Press Enter to start recording...")
        input()

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="float32",
            callback=callback,
        ):
            print(f"Recording (max {self.max_duration}s) - Press Enter to stop")
            start_time = time.time()
            while time.time() - start_time < self.max_duration:
                if input() == "":  # Stop on Enter
                    break

        print("\nTranscription complete")


if __name__ == "__main__":
    transcriber = AudioTranscriber(model_size="base")
    asyncio.run(transcriber.run())
