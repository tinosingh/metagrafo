"""Enhanced real-time transcription with VAD and context."""

import numpy as np
import webrtcvad
import asyncio
from typing import Callable, List
import mlx_whisper  # Import MLX-Whisper library
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SIZE = 0.5  # 500ms chunks for VAD
VAD_AGGRESSIVENESS = 2


class StreamProcessor:
    def __init__(self, callback: Callable[[np.ndarray], None]):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.buffer: List[np.ndarray] = []
        self.speech_context: str = ""
        self.is_speaking = False
        self.callback = callback

    def process_chunk(self, audio: np.ndarray):
        """Process audio chunk with VAD."""
        # Convert to 16-bit PCM for VAD
        pcm = (audio * 32767).astype("int16")

        # Detect speech
        chunk_speech = self.vad.is_speech(pcm.tobytes(), sample_rate=SAMPLE_RATE)

        if chunk_speech:
            self.buffer.append(audio)
            if not self.is_speaking:
                self.is_speaking = True
        else:
            if self.is_speaking and len(self.buffer) > 0:
                self._process_speech()
            self.is_speaking = False

    def _process_speech(self):
        """Process buffered speech segments directly with MLX-Whisper."""
        audio = np.concatenate(self.buffer)
        self.buffer.clear()

        # Convert to float32 in [-1, 1] range
        audio_np = audio.astype("float32")

        result = mlx_whisper.transcribe(
            audio=audio_np, language="", temperature=0.0, task="transcribe"
        )
        text = result.get("text", "[No transcription]")
        self.speech_context += " " + text
        self.callback(text)


async def transcribe_stream(client_id: str, on_transcription: Callable[[str], None]):
    """Enhanced streaming with VAD and context."""
    processor = StreamProcessor(on_transcription)
    processor.client_id = client_id

    def callback(indata, frames, time, status):
        processor.process_chunk(indata[:, 0])

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * CHUNK_SIZE),
        callback=callback,
    )

    try:
        stream.start()
        while True:
            await asyncio.sleep(0.1)
    finally:
        stream.stop()
        stream.close()
