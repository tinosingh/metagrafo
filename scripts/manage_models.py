#!/usr/bin/env python3
"""
Model management utility for the real-time transcription system.

This script provides commands to list, download, and manage AI models.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from our package
from realtime_transcription.models.whisper_model import WhisperModel
from realtime_transcription.config.config_manager import ConfigManager
from realtime_transcription.utils.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging(console=True, json_format=False)
logger = get_logger(__name__)

class ModelManagerCLI:
    """Command-line interface for managing models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config path."""
        self.config = ConfigManager(config_path)
        self.whisper_model = WhisperModel()
        self.available_models: Dict[str, List[str]] = {}
        self.loaded_models: Dict[str, Any] = {}
    
    def list_available_models(self, task: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available models, optionally filtered by task."""
        models = {}
        
        # Get Whisper models
        if task is None or task == "transcription":
            whisper_models = self.whisper_model.get_available_models()
            models["transcription"] = [f"whisper:{m}" for m in whisper_models]
        
        # Add other model types here as they're implemented
        
        self.available_models = models
        return models
    
    def download_model(self, model_id: str) -> bool:
        """Download a model by ID."""
        logger.info(f"Downloading model: {model_id}")
        
        # Handle Whisper models
        if model_id.startswith("whisper:"):
            model_name = model_id.split(":")[1]
            try:
                # This will trigger the download if not already cached
                success = self.whisper_model.load_model(model_name)
                if success:
                    logger.info(f"Successfully downloaded model: {model_id}")
                    return True
                else:
                    logger.error(f"Failed to download model: {model_id}")
                    return False
            except Exception as e:
                logger.error(f"Error downloading model {model_id}: {str(e)}")
                return False
        else:
            logger.error(f"Unsupported model type: {model_id}")
            return False
    
    def list_downloaded_models(self) -> Dict[str, List[str]]:
        """List all downloaded models."""
        downloaded = {}
        
        # Check Whisper cache
        whisper_cache = Path.home() / ".cache" / "whisper"
        if whisper_cache.exists():
            whisper_models = [f.name for f in whisper_cache.glob("*")
                            if not f.name.startswith(".")]
            if whisper_models:
                downloaded["transcription"] = [f"whisper:{m}" for m in whisper_models]
        
        # Add other model types here as they're implemented
        
        return downloaded
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_id.startswith("whisper:"):
            if hasattr(self.whisper_model, 'model_info'):
                info = self.whisper_model.model_info()
                if isinstance(info, dict) and 'error' not in info:
                    return {"whisper": {model_id: info}}
        return {"error": f"No information available for {model_id}"}

def main():
    """Main entry point for the model management CLI."""
    parser = argparse.ArgumentParser(description="Manage AI models for the transcription system")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available or downloaded models")
    list_parser.add_argument(
        "--task", 
        choices=["transcription", "spellcheck", "summarization"],
        help="Filter models by task"
    )
    list_parser.add_argument(
        "--downloaded", 
        action="store_true",
        help="List only downloaded models"
    )
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="ID of the model to download")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get info about a model")
    info_parser.add_argument("model_id", help="ID of the model to get info about")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize manager
    manager = ModelManagerCLI()
    
    # Handle commands
    if args.command == "list":
        if args.downloaded:
            models = manager.list_downloaded_models()
            print("\nDownloaded models:")
        else:
            models = manager.list_available_models(args.task)
            print("\nAvailable models:")
        
        for task, model_list in models.items():
            print(f"\n{task.upper()}:")
            for model_id in model_list:
                print(f"  - {model_id}")
    
    elif args.command == "download":
        success = manager.download_model(args.model_id)
        sys.exit(0 if success else 1)
    
    elif args.command == "info":
        info = manager.get_model_info(args.model_id)
        if info:
            print("\nModel information:")
            for key, value in info.items():
                print(f"{key}: {value}")
        else:
            logger.error(f"No information available for model: {args.model_id}")
            sys.exit(1)

if __name__ == "__main__":
    main()
