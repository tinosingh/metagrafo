#!/usr/bin/env python3
"""
Bulletproof Real-time Transcriber
Works with any version of MLX Whisper
"""

import mlx_whisper
import numpy as np
import sounddevice as sd


def transcribe():
    print("🎤 Real-time Transcription Started!")
    print("📍 Speak clearly, 3-second chunks")
    print("🛑 Press Ctrl+C to stop\n")

    # First, let's see what mlx_whisper expects
    try:
        # Try to transcribe silence to see the API
        test = mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32), path_or_hf_repo="tiny"
        )
        use_path = True
    except:
        try:
            # Try without path_or_hf_repo
            test = mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), "tiny")
            use_path = False
        except Exception as e:
            print(f"Testing MLX Whisper API... {e}")
            use_path = True

    print("Ready! Start speaking...\n")

    while True:
        # Record 3 seconds of audio
        print("🔴 Recording...", end="", flush=True)
        audio = sd.rec(int(16000 * 3), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        print("\r⚡ Processing...", end="", flush=True)

        # Transcribe based on API version
        try:
            if use_path:
                result = mlx_whisper.transcribe(
                    audio[:, 0], path_or_hf_repo="tiny", fp16=False, language="en"
                )
            else:
                result = mlx_whisper.transcribe(audio[:, 0], "tiny")

            # Extract text (handle dict or object)
            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)

            # Clear line and print result
            print(f"\r{'  ' * 30}\r💬 {text.strip()}")

        except Exception as e:
            print(f"\r❌ Error: {e}")
            print("Trying alternative approach...")
            # Fallback: try the simplest possible call
            try:
                result = mlx_whisper.transcribe(audio[:, 0])
                text = (
                    result.get("text", "") if isinstance(result, dict) else str(result)
                )
                print(f"💬 {text.strip()}")
            except Exception as e2:
                print(f"Error: {e2}")


# Auto-run
if __name__ == "__main__":
    try:
        transcribe()
    except KeyboardInterrupt:
        print("\n\n✅ Transcription stopped. Bye! 👋")
