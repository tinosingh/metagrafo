"""
Handles audio capture, processing and queue management.
"""
from collections import deque
import numpy as np
import sounddevice as sd
import numpy.typing as npt
from typing import Optional, Deque

class AudioProcessor:
    """Manages audio capture and processing."""
    
    def __init__(self, sample_rate: int, chunk_duration_sec: float):
        """
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_sec: Duration of audio chunks in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.audio_queue: Deque[npt.NDArray[np.float32]] = deque()
        self.is_running = False
        
    def audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """Sounddevice callback that adds audio chunks to the queue."""
        if status:
            print(f"Audio status: {status}")
        if self.is_running:
            self.audio_queue.append(indata.copy())
    
    def start_capture(self, device: Optional[int] = None):
        """Start audio capture stream."""
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            device=device
        )
        self.stream.start()
    
    def stop_capture(self):
        """Stop audio capture stream."""
        self.is_running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
