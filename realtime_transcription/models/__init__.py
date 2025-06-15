"""AI models for the transcription system."""
from realtime_transcription.models.base_model import BaseModel
from realtime_transcription.models.whisper_model import WhisperModel

__all__ = ['BaseModel', 'WhisperModel']
