"""Handles audio recording and processing."""
import numpy as np
import sounddevice as sd
from queue import Queue
from typing import Optional, Callable, Tuple

class AudioProcessor:
    """Handles audio capture, processing, and streaming."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_duration: float = 3.0,
                 channels: int = 1):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of each audio chunk in seconds
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = Queue()
        self.stream = None
        self.is_recording = False
        self.callback = None
        self.thread = None
    
    def _audio_callback(self, 
                      indata: np.ndarray, 
                      frames: int, 
                      time_info: dict, 
                      status: sd.CallbackFlags) -> None:
        """Callback for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        audio_data = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        
        # Add to queue for processing
        if self.callback:
            self.callback(audio_data)
    
    def start_recording(self, callback: Optional[Callable] = None) -> None:
        """Start audio recording.
        
        Args:
            callback: Function to call with audio chunks
        """
        if self.is_recording:
            print("Already recording")
            return
            
        self.callback = callback
        self.is_recording = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        
        self.stream.start()
        print("Audio recording started")
    
    def stop_recording(self) -> None:
        """Stop audio recording."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("Audio recording stopped")
    
    def record_chunk(self, duration: Optional[float] = None) -> np.ndarray:
        """Record a single chunk of audio."""
        if duration is None:
            duration = self.chunk_duration
            
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            blocking=True
        )
        
        # Convert to mono if needed
        if recording.ndim > 1:
            recording = np.mean(recording, axis=1)
            
        return recording
    
    def process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process audio data (placeholder for audio processing).
        
        Returns:
            Tuple of (processed_audio, duration_in_seconds)
        """
        duration = len(audio_data) / self.sample_rate
        return audio_data, duration
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """Play audio data through speakers."""
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        sd.play(audio_data, sample_rate)
        sd.wait()
