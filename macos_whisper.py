"""Debugged Real-time macOS transcription with MLX Whisper"""

import sys
import time
import threading
from collections import deque
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Debug mode flag
DEBUG = True  # Set to True for verbose output


def debug_print(msg: str):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print(f"[DEBUG] {msg}")


# Check for required packages
try:
    import numpy as np

    debug_print("✓ NumPy imported successfully")
except ImportError:
    print("❌ NumPy not found. Install with: pip install numpy")
    sys.exit(1)

try:
    import sounddevice as sd

    debug_print("✓ Sounddevice imported successfully")
    debug_print(f"  Available devices: {sd.query_devices()}")
except ImportError:
    print("❌ Sounddevice not found. Install with: pip install sounddevice")
    sys.exit(1)

try:
    import mlx_whisper

    debug_print("✓ MLX Whisper imported successfully")
except ImportError:
    print("❌ MLX Whisper not found. Install with: pip install mlx-whisper")
    print("  Note: MLX Whisper requires Apple Silicon (M1/M2/M3)")
    sys.exit(1)

# Test MLX availability
try:
    import mlx.core as mx

    debug_print(f"✓ MLX Core available - Device: {mx.default_device()}")
except ImportError:
    print("❌ MLX not available. This requires Apple Silicon Mac.")
    sys.exit(1)


class MacWhisperTranscriber:
    """Debugged real-time audio transcription using MLX Whisper."""

    def __init__(self, model_size: str = "tiny", debug: bool = True):
        self.debug = debug
        self.sample_rate = 16000
        self.window_duration = 3.0  # seconds
        self.window_size = int(self.sample_rate * self.window_duration)
        self.process_interval = 0.5  # Process every 500ms
        self.model_size = model_size

        # Audio settings
        self.channels = 1
        self.blocksize = 512  # Smaller block size for lower latency

        # Initialize model
        print(f"\n🔄 Loading MLX Whisper {model_size} model...")
        try:
            from mlx_whisper import Whisper

            self.model = Whisper(model_size)
            print("✅ Model loaded successfully!")

            # Test model with real audio sample
            test_audio = (
                np.random.rand(self.sample_rate).astype(np.float32) * 0.1
            )  # 10% volume
            test_result = self.model.transcribe(test_audio, language="en")
            if not test_result or "text" not in test_result:
                print("❌ Model test failed - invalid response format")
                sys.exit(1)

            debug_print(f"Model test passed: {type(test_result)}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("   Try: pip install --upgrade mlx-whisper")
            sys.exit(1)

        # State variables
        self.running = False
        self.audio_buffer = deque(maxlen=self.window_size)
        self.process_lock = threading.Lock()
        self.last_processed_time = 0
        self.accumulated_text = []
        self.last_segment = ""
        self.audio_level = 0
        self.frames_processed = 0
        self.transcription_count = 0

        # Error tracking
        self.error_count = 0
        self.last_error = None

    def check_audio_devices(self):
        """Check and display available audio devices"""
        print("\n📊 Audio Device Information:")
        print("-" * 50)

        devices = sd.query_devices()
        default_input = sd.default.device[0]

        print(f"Default input device: {default_input}")
        print(f"Device info: {devices[default_input]}")
        print(f"Supported sample rates: {devices[default_input]['default_samplerate']}")

        # List all input devices
        print("\nAvailable input devices:")
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(
                    f"  [{i}] {device['name']} - {device['max_input_channels']} channels"
                )

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback with debugging"""
        try:
            if status:
                debug_print(f"Audio callback status: {status}")

            # Check for valid audio data
            if indata is None or len(indata) == 0:
                debug_print("Empty audio data received")
                return

            # Extract mono audio
            if indata.ndim > 1:
                audio_data = indata[:, 0].astype(np.float32)
            else:
                audio_data = indata.astype(np.float32)

            # Calculate audio level
            self.audio_level = float(np.max(np.abs(audio_data)))

            # Add to buffer
            with self.process_lock:
                self.audio_buffer.extend(audio_data)
                self.frames_processed += len(audio_data)

            # Visual feedback
            if not self.debug:
                bars = int(self.audio_level * 30)
                print(f"\r🎤 {'█' * bars}{'░' * (30 - bars)} ", end="", flush=True)
            else:
                debug_print(
                    f"Audio level: {self.audio_level:.3f}, Buffer size: {len(self.audio_buffer)}"
                )

            # Check if it's time to process
            current_time = time.time()
            if (current_time - self.last_processed_time) >= self.process_interval:
                self.last_processed_time = current_time
                # Process in separate thread
                thread = threading.Thread(target=self.process_audio, daemon=True)
                thread.start()

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            debug_print(f"Audio callback error: {e}")

    def process_audio(self):
        """Process audio with comprehensive error handling"""
        if not self.running:
            return

        try:
            with self.process_lock:
                buffer_len = len(self.audio_buffer)
                if buffer_len < self.sample_rate * 0.5:  # Need at least 0.5s
                    debug_print(f"Buffer too small: {buffer_len} samples")
                    return

                audio_data = np.array(self.audio_buffer, dtype=np.float32)

            debug_print(
                f"Processing {len(audio_data)} samples, max amplitude: {np.max(np.abs(audio_data)):.3f}"
            )

            # Check for silence
            if np.max(np.abs(audio_data)) < 0.001:
                debug_print("Audio is silent, skipping transcription")
                return

            # Normalize audio if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
                debug_print(f"Normalized audio from max {max_val}")

            # Transcribe
            start_time = time.time()

            result = self.model.transcribe(
                audio_data,
                language="en",
                fp16=True,
                verbose=False,
                beam_size=1,  # Faster with beam_size=1
                best_of=1,  # Faster with best_of=1
            )

            transcribe_time = time.time() - start_time
            debug_print(f"Transcription took {transcribe_time:.3f}s")

            # Extract text from result
            text = None
            if isinstance(result, dict):
                text = result.get("text", "").strip()
                debug_print(f"Result dict keys: {result.keys()}")
            elif isinstance(result, str):
                text = result.strip()
            elif hasattr(result, "text"):
                text = str(result.text).strip()
            else:
                debug_print(f"Unknown result type: {type(result)}")
                text = str(result).strip()

            if text and text not in ["", " ", ".", ","]:
                self.transcription_count += 1

                if not self.debug:
                    # Clear line and show result
                    print(f"\r{' ' * 80}\r💬 {text}", end="", flush=True)
                else:
                    print(f"\n[{self.transcription_count}] Transcribed: '{text}'")

                self.last_segment = text
                print(f"\n{result['text']}", end=" ", flush=True)

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            print(f"\n❌ Transcription error: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def run(self):
        """Main loop with debugging"""
        print("\n" + "=" * 60)
        print("🎙️  MLX Whisper Real-time Transcription (DEBUG MODE)")
        print(f"📊 Model: {self.model_size}")
        print(f"🔧 Sample rate: {self.sample_rate} Hz")
        print(f"⏱️  Process interval: {self.process_interval}s")
        print("=" * 60)

        # Check audio devices
        self.check_audio_devices()

        print("\n📝 Instructions:")
        print("  - Press Enter to start recording")
        print("  - Speak clearly into your microphone")
        print("  - Press Enter again to stop")
        print("\nReady? ", end="")
        input()

        self.running = True
        self.audio_buffer.clear()
        self.frames_processed = 0
        self.transcription_count = 0
        self.error_count = 0

        try:
            # Configure audio stream
            stream_params = {
                "samplerate": self.sample_rate,
                "channels": self.channels,
                "blocksize": self.blocksize,
                "dtype": "float32",
                "callback": self.audio_callback,
                "finished_callback": lambda: debug_print("Stream finished"),
            }

            debug_print(f"Opening audio stream with params: {stream_params}")

            with sd.InputStream(**stream_params):
                print("\n✅ Recording started! Speak now...")
                if self.debug:
                    print("\n[DEBUG MODE - Verbose output enabled]\n")
                    # Start monitoring thread
                    monitor_thread = threading.Thread(
                        target=self.monitor_status, daemon=True
                    )
                    monitor_thread.start()

                input()  # Wait for Enter to stop

        except sd.PortAudioError as e:
            print(f"\n❌ Audio device error: {e}")
            print(
                "   Try: Check if microphone is connected and permissions are granted"
            )

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()

        finally:
            self.running = False
            print("\n📊 Session Statistics:")
            print(f"   Frames processed: {self.frames_processed}")
            print(f"   Transcriptions: {self.transcription_count}")
            print(f"   Errors: {self.error_count}")
            if self.last_error:
                print(f"   Last error: {self.last_error}")

            print("\n✨ Done!\n")

    def monitor_status(self):
        """Monitor thread for debugging"""
        while self.running:
            time.sleep(2)
            print(
                f"\n[MONITOR] Buffer: {len(self.audio_buffer)}, "
                f"Level: {self.audio_level:.3f}, "
                f"Transcriptions: {self.transcription_count}, "
                f"Errors: {self.error_count}"
            )


def test_installation():
    """Test if all components are properly installed"""
    print("🔍 Testing installation...")

    # Test audio
    try:
        print("\n1. Testing audio recording (2 seconds)...")
        recording = sd.rec(
            int(2 * 16000), samplerate=16000, channels=1, dtype="float32"
        )
        sd.wait()
        print(f"   ✅ Recorded {len(recording)} samples")
        print(f"   Max amplitude: {np.max(np.abs(recording)):.3f}")
    except Exception as e:
        print(f"   ❌ Audio test failed: {e}")
        return False

    # Test MLX Whisper
    try:
        print("\n2. Testing MLX Whisper model...")
        model = mlx_whisper.load_models("tiny")

        # Create test audio (sine wave)
        t = np.linspace(0, 1, 16000)
        test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = model.transcribe(test_audio)
        print("   ✅ Model test passed")
        print(f"   Result type: {type(result)}")

    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX Whisper Real-time Transcription")
    parser.add_argument("--test", action="store_true", help="Run installation test")
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: tiny)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")

    args = parser.parse_args()

    if args.test:
        test_installation()
    else:
        DEBUG = not args.no_debug
        transcriber = MacWhisperTranscriber(model_size=args.model, debug=DEBUG)
        transcriber.run()
