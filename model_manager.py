"""
Model management for Whisper models using Hugging Face cache.
"""
import torch
import whisper
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

class ModelManager:
    """Manages Whisper models using Hugging Face cache."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.model_list_path = Path(__file__).parent / "model_list.json"
        self.hf_cache = Path("~/.cache/huggingface/hub").expanduser()
        
        # Initialize model list if it doesn't exist
        if not self.model_list_path.exists():
            with open(self.model_list_path, 'w') as f:
                json.dump({"last_scan": "", "models": {}, "warnings": []}, f)
    
    def _load_model_list(self) -> Dict[str, Any]:
        """Load the model list from JSON."""
        with open(self.model_list_path) as f:
            return json.load(f)
            
    def _save_model_list(self, model_info: Dict[str, Any]) -> None:
        """Save the model list to JSON."""
        with open(self.model_list_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def scan_models(self) -> Dict[str, Any]:
        """
        Scan Hugging Face cache for available Whisper models.
        Returns dict with model info and scan metadata.
        """
        model_info = {
            "last_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": {},
            "warnings": []
        }

        if not self.hf_cache.exists():
            model_info["warnings"].append("Hugging Face cache not found")
            return model_info

        # Scan HF cache for Whisper models
        for org_dir in self.hf_cache.glob("models--*"):
            try:
                if not org_dir.is_dir():
                    continue
                    
                # Extract model name (convert -- to / in org/repo format)
                model_name = org_dir.name.replace("models--", "").replace("--", "/")
                
                # Look for snapshots
                snapshots_dir = org_dir / "snapshots"
                if not snapshots_dir.exists():
                    continue
                    
                # Get most recent snapshot
                snapshots = sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                if not snapshots:
                    continue
                    
                latest_snapshot = snapshots[0]
                
                # Check if this is a Whisper model
                if self._is_whisper_model(latest_snapshot):
                    model_info["models"][model_name] = {
                        "cache_path": str(latest_snapshot),
                        "type": "huggingface",
                        "last_modified": latest_snapshot.stat().st_mtime
                    }
                    
            except Exception as e:
                model_info["warnings"].append(f"Error scanning {org_dir.name}: {str(e)}")
        
        self._save_model_list(model_info)
        print(f"Found {len(model_info['models'])} models in Hugging Face cache")
        return model_info
    
    def _is_whisper_model(self, model_dir: Path) -> bool:
        """Check if a directory contains a valid Whisper model."""
        # Check for standard Whisper model files
        if (model_dir / "pytorch_model.bin").exists() or \
           any(model_dir.glob("*.pt")) or \
           any(model_dir.glob("*.safetensors")):
            return True
        return False
    
    def load_model(self, model_name: str, device: str = None) -> whisper.Whisper:
        """
        Load a Whisper model by its Hugging Face model ID.
        Example: "openai/whisper-large-v3"
        """
        model_info = self._load_model_list()
        
        if model_name not in model_info["models"]:
            raise ValueError(f"Model {model_name} not found in cache. Available: {list(model_info['models'].keys())}")
            
        model_data = model_info["models"][model_name]
        model_path = Path(model_data["cache_path"])
        
        if not self._is_whisper_model(model_path):
            raise ValueError(f"Invalid model files at {model_path}")
            
        device_type = device or self.get_device_type()
        return whisper.load_model(str(model_path), device=device_type)
    
    def get_device_type(self) -> str:
        """Determine the best available device type."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of available models with their metadata."""
        model_info = self._load_model_list()
        return model_info.get("models", {})
