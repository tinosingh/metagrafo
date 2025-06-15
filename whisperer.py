"""Fixed Real-time macOS transcription with MLX Whisper"""

import sys
import time
import threading
import numpy as np
import sounddevice as sd
from collections import deque
from typing import Any, Union
import warnings
import os

warnings.filterwarnings("ignore")

DEBUG = True


def debug_print(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def extract_text(result: Union[dict, str, Any]) -> str:
    """Extract transcribed text from different result types."""
    if isinstance(result, dict):
        return result.get("text", "").strip()
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()


def print_transcription(text: str, debug: bool):
    """Display transcribed text inline for real-time output."""
    if text and text not in ["", " ", ".", ","]:
        if not debug:
            print(f"\r💬 {text}{' ' * 20}", end="", flush=True)
        else:
            print(f"\n[Transcribed] '{text}'")


class MacWhisperTranscriber:
    """
    Real-time audio transcription using MLX Whisper model.
    Uses a circular buffer for continuous transcription.
    """

    def __init__(self, model_size: str = "tiny", debug: bool = True):
        self.debug = debug
        self.sample_rate = 16000
        self.buffer_duration = 2.0  # 2 second buffer
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.process_interval = 0.3  # process every 300ms
        self.model_size = model_size
        self.channels = 1
        self.blocksize = 512  # Increase for stability

        print(f"\n🔄 Loading MLX Whisper {model_size} model...")

        # Import and detect the correct API
        try:
            import mlx_whisper

            # Check what's available in mlx_whisper
            debug_print(f"MLX Whisper attributes: {dir(mlx_whisper)}")

            # Try different API patterns
            model_path = f"models/{model_size}.bin"
            config_path = f"models/{model_size}.json"

            if hasattr(mlx_whisper, "transcribe"):
                # Use local model files
                if os.path.exists(model_path) and os.path.exists(config_path):
                    self.transcribe_fn = lambda audio: mlx_whisper.transcribe(
                        audio, model_path=model_path, config_path=config_path
                    )
                else:
                    raise FileNotFoundError(
                        f"Model files not found at {model_path} and {config_path}"
                    )

            elif hasattr(mlx_whisper, "load"):
                # Alternative API
                self.model = mlx_whisper.load(model_size)
                self.transcribe_fn = lambda audio: self.model.transcribe(audio)

            elif hasattr(mlx_whisper, "whisper"):
                # Try to find whisper submodule
                whisper_module = mlx_whisper.whisper
                if hasattr(whisper_module, "load_model"):
                    self.model = whisper_module.load_model(model_size)
                    self.transcribe_fn = lambda audio: self.model.transcribe(audio)
                else:
                    raise ImportError(
                        "Cannot find proper load function in mlx_whisper.whisper"
                    )
            else:
                raise ImportError("Cannot find proper API in mlx_whisper module")

            print("✅ Model loaded successfully!")

            # Test the transcription
            test_audio = np.zeros(self.sample_rate, dtype=np.float32)
            test_result = self.transcribe_fn(test_audio)
            debug_print(f"Model test passed: {type(test_result)}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure mlx-whisper is installed: pip install mlx-whisper")
            print("2. Try upgrading: pip install --upgrade mlx-whisper")
            print("3. Check MLX compatibility with your system")

            # Try to show more info
            try:
                import mlx_whisper

                print("\nMLX Whisper version info:")
                if hasattr(mlx_whisper, "__version__"):
                    print(f"  Version: {mlx_whisper.__version__}")
                print(
                    f"  Available functions: {[x for x in dir(mlx_whisper) if not x.startswith('_')]}"
                )
            except Exception as e:
                print(f"  Could not get version info: {e}")

            sys.exit(1)

        self.running = False
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.process_lock = threading.Lock()
        self.last_processed_time = 0
        self.audio_level = 0
        self.frames_processed = 0
        self.transcription_count = 0
        self.error_count = 0
        self.last_error = None
        self.last_text = ""

    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback function."""
        try:
            if status:
                debug_print(f"Audio callback status: {status}")

            if indata is None or len(indata) == 0:
                debug_print("Empty audio data received")
                return

            # Extract mono audio
            audio_data = (
                indata[:, 0].astype(np.float32)
                if indata.ndim > 1
                else indata.astype(np.float32)
            )
            self.audio_level = float(np.max(np.abs(audio_data)))

            # Add to buffer
            with self.process_lock:
                self.audio_buffer.extend(audio_data)
                self.frames_processed += len(audio_data)

            # Visual feedback
            if not self.debug:
                bars = int(self.audio_level * 30)
                print(f"\r🎤 {'█' * bars}{'░' * (30 - bars)} ", end="", flush=True)

            # Check if it's time to process
            current_time = time.time()
            if (current_time - self.last_processed_time) >= self.process_interval:
                self.last_processed_time = current_time
                thread = threading.Thread(target=self.process_audio, daemon=True)
                thread.start()

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            debug_print(f"Audio callback error: {e}")

    def process_audio(self):
        """Processes audio buffer for transcription."""
        if not self.running:
            return

        try:
            with self.process_lock:
                buffer_len = len(self.audio_buffer)
                if buffer_len < self.sample_rate * 0.5:  # Need at least 0.5s
                    debug_print(f"Buffer too small: {buffer_len} samples")
                    return

                # Convert deque to numpy array
                audio_data = np.array(list(self.audio_buffer), dtype=np.float32)

            # Check for silence
            if np.max(np.abs(audio_data)) < 0.001:
                debug_print("Silent audio, skipping")
                return

            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
                debug_print(f"Normalized audio from max {max_val}")

            # Transcribe using the detected API
            result = self.transcribe_fn(audio_data)
            text = extract_text(result)

            # Only print if different from last
            if text and text != self.last_text:
                self.transcription_count += 1
                print_transcription(text, self.debug)
                self.last_text = text

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            if self.debug:
                print(f"\n❌ Transcription error: {e}")
                import traceback

                traceback.print_exc()

    def run(self):
        """Start real-time transcription."""
        print("\n" + "=" * 60)
        print("🎙️  MLX Whisper Real-time Transcription")
        print(f"📊 Model: {self.model_size}")
        print(f"🔧 Sample rate: {self.sample_rate} Hz")
        print(f"⏱️  Process interval: {self.process_interval}s")
        print("=" * 60)

        # List audio devices
        print("\n📊 Available audio devices:")
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        print(f"Default input: [{default_input}] {devices[default_input]['name']}")

        print("\n📝 Press Enter to start recording...")
        input()

        self.running = True
        self.audio_buffer.clear()
        self.frames_processed = 0
        self.transcription_count = 0
        self.error_count = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self.audio_callback,
            ):
                print("\n✅ Recording started! Speak now...")
                print("🛑 Press Enter to stop\n")

                if not self.debug:
                    print("🎤 " + "░" * 30)

                # Wait for Enter key
                input()

        except sd.PortAudioError as e:
            print(f"\n❌ Audio device error: {e}")
            print(
                "Check microphone permissions in System Settings > Privacy & Security > Microphone"
            )

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()

        finally:
            self.running = False

            # Clean up MLX resources safely
            try:
                import mlx.core as mx

                if hasattr(mx, "metal"):
                    mx.metal.clear_cache()
                elif hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception:
                pass  # Ignore cleanup errors

            print("\n\n📊 Session Statistics:")
            print(f"   Frames processed: {self.frames_processed:,}")
            print(f"   Transcriptions: {self.transcription_count}")
            print(f"   Errors: {self.error_count}")
            if self.last_error:
                print(f"   Last error: {self.last_error}")

            print("\n✨ Done!\n")


def diagnose_mlx_whisper():
    """Diagnose MLX Whisper installation and show available APIs."""
    print("🔍 Diagnosing MLX Whisper installation...\n")

    try:
        import mlx_whisper

        print("✅ MLX Whisper is installed")

        # Show version if available
        if hasattr(mlx_whisper, "__version__"):
            print(f"   Version: {mlx_whisper.__version__}")

        # List all available attributes
        attrs = [x for x in dir(mlx_whisper) if not x.startswith("_")]
        print("\n📋 Available functions/attributes:")
        for attr in attrs:
            obj = getattr(mlx_whisper, attr)
            obj_type = type(obj).__name__
            print(f"   - {attr} ({obj_type})")

        # Check for submodules
        print("\n📦 Checking for submodules:")
        for attr in attrs:
            obj = getattr(mlx_whisper, attr)
            if hasattr(obj, "__module__"):
                print(f"   - {attr} is a module/class")
                sub_attrs = [x for x in dir(obj) if not x.startswith("_")][:5]
                if sub_attrs:
                    print(f"     Contains: {', '.join(sub_attrs)}...")

    except ImportError as e:
        print(f"❌ MLX Whisper not installed: {e}")
        print("\nInstall with: pip install mlx-whisper")
        return False

    # Try to test transcription
    print("\n🧪 Testing transcription methods:")
    try:
        test_audio = np.zeros(16000, dtype=np.float32)

        # Test direct transcribe
        if hasattr(mlx_whisper, "transcribe"):
            print("   Testing mlx_whisper.transcribe()...")
            result = mlx_whisper.transcribe(test_audio, path_or_hf_repo="tiny")
            print(f"   ✅ Direct transcribe works! Result type: {type(result)}")
            return True
    except Exception as e:
        print(f"   ❌ Transcribe test failed: {e}")

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX Whisper Real-time Transcription")
    parser.add_argument(
        "--diagnose", action="store_true", help="Diagnose MLX Whisper installation"
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: tiny)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")

    args = parser.parse_args()

    if args.diagnose:
        diagnose_mlx_whisper()
    else:
        DEBUG = not args.no_debug
        transcriber = MacWhisperTranscriber(model_size=args.model, debug=DEBUG)
        transcriber.run()
