"""Optimized macOS transcription based on maeda.pm implementation"""

import sounddevice as sd
import numpy as np
import mlx.core as mx
import asyncio
from backend.transcription import transcribe_audio


class MacOSTranscriber:
    def __init__(self, model="base"):
        self.model = model
        self.sr = 16000
        self.chunk = self.sr // 10  # 100ms chunks
        self.buffer = mx.array([], dtype=mx.float32)

    async def process_audio(self, audio_np):
        """MLX-optimized audio processing"""
        try:
            # Convert with proper memory management
            audio_mlx = mx.array(audio_np.ravel(), dtype=mx.float32)

            # Run transcription
            result = await transcribe_audio(audio_path=audio_mlx, model_size=self.model)
            print(f"\n{result['text']}", end=" ", flush=True)
            return result
        except Exception as e:
            print(f"\nError: {str(e)}")
            return None

    def run(self):
        print(f"macOS Optimized Whisper (model: {self.model})\n")

        def callback(indata, frames, time_info, status):
            """Low-latency audio callback"""
            if status:
                print(f"\nAudio status: {status}")

            # Visual feedback
            level = int(np.clip(np.max(np.abs(indata)) * 50, 0, 30))
            print("■" * level + " " * (30 - level), end="\r")

            # Process in background
            asyncio.create_task(self.process_audio(indata))

        print("Press Enter to start/stop recording")
        input()

        with sd.InputStream(
            samplerate=self.sr, blocksize=self.chunk, dtype="float32", callback=callback
        ):
            print("Recording... (Press Enter to stop)")
            input()

        print("\n\nTranscription complete")


if __name__ == "__main__":
    transcriber = MacOSTranscriber(model="base")
    asyncio.run(transcriber.run())
