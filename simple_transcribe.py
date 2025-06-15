import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

from backend.transcription import transcribe_audio

# Configure audio settings at the top of the file
default_device = 0  # MacBook Air Microphone
fs = 16000  # Restore original sample rate
dtype = "float32"  # Lower precision
max_mb = 100  # Max MB per chunk
chunk_samples = 16000  # Process 1-second chunks (fs * 1s)
max_chunks = 30  # 30 seconds max


def record_and_transcribe():
    print("Press Enter to start recording...")
    input()

    try:
        sd.default.device = default_device
        print(f"Using default microphone: {sd.query_devices(default_device)['name']}")

        recording = []
        start_time = time.time()
        executor = ThreadPoolExecutor(max_workers=1)

        def callback(indata, frames, current_time, status):
            nonlocal start_time

            if status:
                print(f"Audio status: {status}")

            if len(recording) >= max_chunks:
                raise sd.CallbackStop

            # Validate input size
            if frames != chunk_samples:
                print(f"\nWARNING: Expected {chunk_samples} samples, got {frames}")
                return

            # Pre-allocated copy
            chunk = np.empty_like(indata, dtype="float32")
            np.copyto(chunk, indata)
            recording.append(chunk)

            # Process immediately
            executor.submit(process_chunk, chunk)

            # Visual feedback
            max_level = min(30, int(np.max(np.abs(indata)) * 50))
            print("■" * max_level + " " * (30 - max_level), end="\r")

        def process_chunk(chunk):
            if chunk.nbytes > max_mb * 1024 * 1024:
                print(f"\nSkipping oversized chunk: {chunk.nbytes / 1024 / 1024:.1f}MB")
                return
            try:
                result = asyncio.run(
                    transcribe_audio(audio_path=chunk, model_size="base")
                )
                print(f"\nPartial: {result['text']}")
            except Exception as e:
                print(f"\nTranscription error: {str(e)}")

        with sd.InputStream(
            samplerate=fs,
            channels=1,
            blocksize=chunk_samples,
            dtype="float32",
            callback=callback,
        ):
            input()

        executor.shutdown(wait=True)

        if not recording:
            raise ValueError("No audio recorded - check microphone permissions")

        audio = np.concatenate(recording)
        print(f"Recorded {len(audio) / fs:.2f} seconds of audio")

        if input("Save recording? (y/n): ").lower() == "y":
            from scipy.io import wavfile

            filename = f"recording_{int(time.time())}.wav"
            wavfile.write(filename, fs, audio)
            print(f"Saved to {filename}")

        if len(audio) > 0:
            results = []
            for i, chunk in enumerate(recording):
                print(f"Chunk {i + 1}/{len(recording)} ({len(chunk) / fs:.2f}s)")
                results.append(
                    asyncio.run(transcribe_audio(audio_path=chunk, model_size="base"))
                )
            result = {"text": " ".join(r["text"] for r in results)}
        else:
            print("Warning: No audio recorded")
            result = {"text": ""}

        print("\nTranscription:")
        print(result["text"])

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check System Preferences > Security & Privacy > Microphone")
        print("2. Ensure terminal has microphone access")
        print("3. Try different audio device if available")


if __name__ == "__main__":
    while True:
        record_and_transcribe()
        if input("Record again? (y/n): ").lower() != "y":
            break
