"""
Core transcription service using Whisper model.
"""
import whisper
import time
from typing import Optional
import numpy as np
import numpy.typing as npt

class TranscriptionService:
    """Handles core transcription functionality."""
    
    def __init__(self, model_name: str, device_type: str, model_manager):
        """
        Args:
            model_name: Name of Whisper model or Hugging Face path
            device_type: Device type ('cuda', 'mps', 'cpu')
            model_manager: ModelManager instance
        """
        self.model_name = model_name
        self.device_type = device_type
        self.model_manager = model_manager
        self.model: Optional[whisper.Whisper] = None
        
    def load_model(self):
        """Load Whisper model using ModelManager."""
        self.model = self.model_manager.load_model(self.model_name, self.device_type)
    
    def transcribe_audio(self, audio_data: npt.NDArray[np.float32], sample_rate: int) -> tuple[str, float]:
        """Transcribe audio chunk to text."""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        start_time = time.time()
        result = self.model.transcribe(
            audio_data,
            sample_rate=sample_rate,
            fp16=(self.device_type == "cuda")
        )
        processing_time = time.time() - start_time
        
        return result["text"].strip(), processing_time
