"""Implementation of Whisper model for speech recognition."""
from pathlib import Path
from typing import Dict, Any
import whisper
import torch
from realtime_transcription.models.base_model import BaseModel

class WhisperModel(BaseModel):
    """Wrapper for Whisper speech recognition models."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.model = None
        self.current_model_name = None
        self.model_info_cache = {}
    
    def load(self, model_name: str, device: str = None) -> None:
        """Load a Whisper model by name."""
        if not device:
            device = self._get_device()
            
        print(f"Loading Whisper model: {model_name} on {device}")
        try:
            self.model = whisper.load_model(model_name, device=device)
            self.current_model_name = model_name
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def process(self, audio_data: Any, **kwargs) -> Dict[str, Any]:
        """Transcribe audio data."""
        if not self.model:
            raise ValueError("No model loaded. Call load() first.")
            
        try:
            result = self.model.transcribe(
                audio_data,
                fp16=torch.cuda.is_available(),
                **kwargs
            )
            return {"text": result["text"], "segments": result.get("segments", [])}
        except Exception as e:
            return {"error": str(e), "text": ""}
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List available Whisper models in cache."""
        cache_dir = Path("~/.cache/huggingface/hub").expanduser()
        models = {}
        
        if not cache_dir.exists():
            return {}
            
        # Look for whisper model directories
        for model_dir in cache_dir.glob("models--*"):
            try:
                # Check if this is a Whisper model
                if not model_dir.name.startswith("models--openai"):
                    continue
                    
                # Get the model name from the directory
                model_name = model_dir.name.replace("models--openai--whisper-", "")
                
                # Find the latest snapshot
                snapshots = list((model_dir / "snapshots").glob("*"))
                if not snapshots:
                    continue
                    
                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                
                # Add to available models
                models[f"whisper:{model_name}"] = {
                    "path": str(latest_snapshot),
                    "last_modified": latest_snapshot.stat().st_mtime
                }
            except Exception as e:
                print(f"Error scanning {model_dir}: {str(e)}")
                
        return models
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get info about the currently loaded model.
        
        Returns:
            Dictionary containing model information including name, type, device, and parameter count.
        """
        if not self.model or not self.current_model_name:
            return {"error": "No model loaded"}
            
        if self.current_model_name not in self.model_info_cache:
            self.model_info_cache[self.current_model_name] = {
                "name": self.current_model_name,
                "type": "whisper",
                "model_name": self.current_model_name,
                "device": str(self.model.device) if self.model else "unknown",
                "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
            }
            
        return self.model_info_cache[self.current_model_name]
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
