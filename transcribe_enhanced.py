#!/usr/bin/env python3
"""
Low-latency, real-time transcriber using a streaming architecture with improved efficiency.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import torch
from colorama import Fore, Style, init

import whisper

# Initialize colorama
init(autoreset=True)

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Set up global variables
    ENABLE_COLORS = config.get("globals", {}).get("enable_colors", True)
    DEBUG_MODE = config.get("globals", {}).get("debug_mode", False)
    MAX_RETRIES = config.get("globals", {}).get("max_retries", 3)

except Exception as e:
    print(
        f"{Fore.RED}❌ Failed to load config: {e}{Style.RESET_ALL}"
        if ENABLE_COLORS
        else f"Error: Failed to load config: {e}"
    )
    config = {}
    ENABLE_COLORS = True
    DEBUG_MODE = False
    MAX_RETRIES = 3

# Supported models (update this list)
MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
]


# Determine available device type
def get_device_type() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RealTimeTranscriber:
    """
    A real-time audio transcriber using a producer-consumer pattern.
    The audio callback (producer) continuously records audio into a queue.
    The transcription worker (consumer) processes the queued audio.
    """

    def __init__(self, model_name: str = None, device: Optional[int] = None):
        """
        Initializes the real-time transcriber.
        Args:
            model_name (str): The name of the Whisper model to use.
            device (int, optional): The index of the audio device to use. Defaults to None.
        """
        self.model_name = model_name or config.get("defaults", {}).get("model", "base")
        self.device = device
        self.sample_rate = config.get("defaults", {}).get("sample_rate", 16000)
        self.model: Optional[whisper.Whisper] = None
        self.audio_queue: deque[npt.NDArray[np.float32]] = deque()
        self.is_running = False
        # Set a threshold (in seconds) for the minimum amount of audio to transcribe.
        self.chunk_duration_sec = config.get("defaults", {}).get(
            "chunk_duration_sec", 1.0
        )
        # A simple silence threshold for VAD; experiment with this value.
        self.silence_threshold = 0.01
        self.device_type = get_device_type()
        self.cache_dir = self.get_model_cache_dirs()[0]

        # Performance tracking
        self.stats = {
            "total_chunks": 0,
            "total_audio_sec": 0.0,
            "total_processing_sec": 0.0,
            "last_latency": 0.0,
        }

    def get_model_cache_dirs(self):
        """Returns all configured cache directories that exist with valid models."""
        valid_dirs = []
        for path in config.get("paths", {}).get("model_cache_paths", []):
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                valid_dirs.append(str(expanded_path))
        return valid_dirs or [str(Path.home() / ".cache" / "whisper")]

    def _is_model_cached(self, model_name: str) -> bool:
        """Check if model exists in cache with support for custom models."""
        for cache_dir in self.get_model_cache_dirs():
            # Standard .pt files
            if (Path(cache_dir) / f"{model_name}.pt").exists():
                return True

            # Hugging Face format (both OpenAI and custom)
            hf_patterns = [
                f"models--openai--whisper-{model_name.split('-')[0]}/snapshots/*",
                f"models--*--whisper-*/snapshots/*",  # Custom models
            ]

            for pattern in hf_patterns:
                if list(Path(cache_dir).glob(pattern)):
                    return True
        return False

    def load_model(self) -> bool:
        if self.model is None:
            print(
                f"\n{Fore.CYAN}📦 Loading '{self.model_name}' model...{Style.RESET_ALL}"
            )

            if not self._is_model_cached(self.model_name):
                print(
                    f"{Fore.YELLOW}⚠️ Model not found in cache - will download{Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.GREEN}✅ Model found in cache{Style.RESET_ALL}")

            use_fp16 = self.device_type in ("cuda", "mps")
            device_name = {"cuda": "NVIDIA GPU", "mps": "Apple Silicon", "cpu": "CPU"}[
                self.device_type
            ]
            print(
                f"{Fore.GREEN}⚡ Using {device_name} ({'fp16' if use_fp16 else 'fp32'}){Style.RESET_ALL}"
            )

            try:
                # Load model
                self.model = whisper.load_model(
                    self.model_name,
                    device=self.device_type,
                    download_root=self.cache_dir,
                )

                # Warm up with empty audio
                warmup_audio = np.zeros((16000,), dtype=np.float32)  # 1s of silence
                _ = self.model.transcribe(warmup_audio, fp16=use_fp16)

                print(f"{Fore.GREEN}✅ Model loaded and warmed up!{Style.RESET_ALL}")
                return True

            except Exception as e:
                print(f"{Fore.RED}❌ Failed to load model: {e}{Style.RESET_ALL}")
                return False
        return True

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Called by the sounddevice stream for each audio block. Quickly copy and enqueue the data."""
        if status:
            print(status, flush=True)
        # Avoid in-place modifications
        self.audio_queue.append(indata.copy())

    def _is_speech(self, audio: np.ndarray) -> bool:
        """Improved voice activity detection using energy thresholding."""
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio**2))
        # Dynamic threshold based on background noise level
        return energy > max(self.silence_threshold, 0.02 * np.abs(audio).max())

    def _transcription_worker(self) -> None:
        """Processes audio chunks with dynamic buffering and overlap."""
        chunks = []
        overlap_samples = int(0.2 * self.sample_rate)  # 200ms overlap

        while self.is_running:
            try:
                start_time = time.time()
                audio_chunk = self.audio_queue.popleft()
                chunks.append(audio_chunk.flatten())

                # Process if speech detected or buffer too large
                audio_buffer = np.concatenate(chunks)
                if (
                    self._is_speech(audio_buffer)
                    or len(audio_buffer)
                    >= 1.5 * self.sample_rate * self.chunk_duration_sec
                ):

                    # Keep last 200ms for overlap
                    if len(chunks) > 1:
                        chunks = [chunks[-1][-overlap_samples:]]
                    else:
                        chunks = []

                    process_start = time.time()
                    result = self.model.transcribe(
                        audio_buffer, fp16=(self.device_type in ("cuda", "mps"))
                    )

                    # Update stats
                    audio_sec = len(audio_buffer) / self.sample_rate
                    process_sec = time.time() - process_start
                    self.stats["total_chunks"] += 1
                    self.stats["total_audio_sec"] += audio_sec
                    self.stats["total_processing_sec"] += process_sec
                    self.stats["last_latency"] = process_sec

                    if DEBUG_MODE:
                        print(
                            f"\n{Fore.BLUE}📊 Chunk: {audio_sec:.2f}s | "
                            f"Process: {process_sec:.2f}s | "
                            f"RTF: {process_sec/audio_sec:.2f}{Style.RESET_ALL}"
                        )

                    text = result["text"].strip()
                    if text:
                        print(
                            f"\r{Fore.GREEN}{time.strftime('%H:%M:%S')}: {text}{Style.RESET_ALL}",
                            end="",
                            flush=True,
                        )

            except IndexError:
                time.sleep(0.01)

    def stop(self) -> None:
        """Stops the transcription gracefully."""
        if self.is_running:
            print(f"{Fore.YELLOW}🛑 Stopping transcription...{Style.RESET_ALL}")
            self.is_running = False

            # Gracefully stop audio stream
            if hasattr(self, "stream") and self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    print(
                        f"{Fore.RED}⚠️  Error stopping audio stream: {e}{Style.RESET_ALL}"
                    )

            # Wait for worker thread to finish
            if hasattr(self, "worker_thread") and self.worker_thread:
                self.worker_thread.join(timeout=2.0)
                if self.worker_thread.is_alive():
                    print(
                        f"{Fore.RED}⚠️  Worker thread did not stop gracefully{Style.RESET_ALL}"
                    )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are cleaned up."""
        self.stop()

    def run(self) -> None:
        """Runs the real-time transcription with enhanced error handling."""
        try:
            if not self.load_model():
                raise RuntimeError("Failed to load Whisper model")

            print(
                f"{Fore.CYAN}🎤 Starting real-time transcription (press Ctrl+C to stop)...{Style.RESET_ALL}"
            )

            # Context manager for audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._audio_callback,
            ) as self.stream:
                self.is_running = True
                self.worker_thread = threading.Thread(
                    target=self._transcription_worker, daemon=True
                )
                self.worker_thread.start()

                try:
                    while self.is_running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print(
                        f"{Fore.YELLOW}\n🛑 Transcription stopped by user.{Style.RESET_ALL}"
                    )
                except Exception as e:
                    print(f"{Fore.RED}\n❌ Unexpected error: {e}{Style.RESET_ALL}")
                    raise
                finally:
                    self.stop()

        except sd.PortAudioError as e:
            print(f"{Fore.RED}❌ Audio device error: {e}{Style.RESET_ALL}")
        except whisper.WhisperError as e:
            print(f"{Fore.RED}❌ Whisper model error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
            raise


def scan_and_update_models() -> None:
    """Scans for both standard and custom models."""
    transcriber = RealTimeTranscriber()

    # Standard models + any found custom ones
    all_models = MODELS.copy()

    # Find custom models in Hugging Face cache
    for cache_dir in transcriber.get_model_cache_dirs():
        for model_dir in Path(cache_dir).glob("models--*--whisper-*"):
            model_name = model_dir.name.split("--")[-1].replace("whisper-", "")
            if model_name not in all_models:
                all_models.append(model_name)

    # Get current cached models from config
    current_models = set(config.get("cached_models", {}).get("available", []))
    verified_models = set()

    # Verify existing models still exist
    for model in current_models:
        if model in all_models and transcriber._is_model_cached(model):
            verified_models.add(model)

    # Scan for new models
    for model in all_models:
        if model not in verified_models and transcriber._is_model_cached(model):
            verified_models.add(model)

    # Update config if changed
    if verified_models != current_models:
        config["cached_models"] = {
            "last_scan": time.strftime("%Y-%m-%d %H:%M:%S"),
            "available": sorted(verified_models),
        }
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
            if DEBUG_MODE:
                print(f"{Fore.BLUE}ℹ️ Updated cached models list{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Could not update config: {e}{Style.RESET_ALL}")


def select_model_and_run():
    """Lets the user select a model and then runs the transcriber."""

    # Get all models (standard + custom)
    print(f"{Fore.BLUE}🔍 Scanning for all available models...{Style.RESET_ALL}")
    scan_and_update_models()

    available_models = config.get("cached_models", {}).get("available", [])

    # Show menu with custom models marked
    print(f"\n{Fore.CYAN}🎯 Select a Model:{Style.RESET_ALL}")
    print(f"  0. {Fore.YELLOW}EXIT{Style.RESET_ALL}")

    # Standard models first
    for i, name in enumerate(MODELS, 1):
        if name in available_models:
            status = f"{Fore.GREEN}(installed){Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}(download required){Style.RESET_ALL}"
        print(f"  {i}. {name.upper()} {status}")

    # Custom models after
    custom_models = [m for m in available_models if m not in MODELS]
    if custom_models:
        print(f"\n{Fore.MAGENTA}Custom Models:{Style.RESET_ALL}")
        for i, name in enumerate(custom_models, len(MODELS) + 1):
            print(
                f"  {i}. {name.replace('-', ' ').title()} {Fore.GREEN}(installed){Style.RESET_ALL}"
            )

    while True:
        try:
            choice = int(
                input(
                    f"{Fore.YELLOW}\nSelect model (0-{len(MODELS)+len(custom_models)}): {Style.RESET_ALL}"
                )
            )
            if choice == 0:
                print(f"{Fore.YELLOW}🚪 Exiting gracefully...{Style.RESET_ALL}")
                return
            if 1 <= choice <= len(MODELS):
                model_name = MODELS[choice - 1]
                break
            elif len(MODELS) < choice <= len(MODELS) + len(custom_models):
                model_name = custom_models[choice - len(MODELS) - 1]
                break
            print(
                f"{Fore.RED}⚠️ Please enter a number between 0 and {len(MODELS)+len(custom_models)}.{Style.RESET_ALL}"
            )
        except ValueError:
            print(f"{Fore.RED}⚠️ Please enter a valid number.{Style.RESET_ALL}")

    with RealTimeTranscriber(model_name=model_name) as transcriber:
        transcriber.run()


if __name__ == "__main__":
    select_model_and_run()
