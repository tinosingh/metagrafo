"""Robust macOS transcription with MLX Whisper"""

import sounddevice as sd
import mlx.core as mx
import mlx_whisper


class WhisperTranscriber:
    def __init__(self, model_size="tiny"):
        self.sample_rate = 16000
        self.chunk_size = self.sample_rate // 10  # 100ms chunks
        self.model = mlx_whisper.load_model(model_size)
        self.running = True

    def audio_callback(self, indata, frames, time, status):
        """Process audio chunks"""
        try:
            audio = mx.array(indata.ravel(), dtype=mx.float32)
            result = self.model.transcribe(audio)
            if "text" in result:
                print(f"\n>> {result['text']}", end="\r")
        except Exception as e:
            print(f"\nError: {e}")

    def run(self):
        """Start transcription"""
        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback,
        ):
            print("Recording... Press Ctrl+C to stop")
            while self.running:
                sd.sleep(100)


if __name__ == "__main__":
    transcriber = WhisperTranscriber("tiny")
    transcriber.run()
