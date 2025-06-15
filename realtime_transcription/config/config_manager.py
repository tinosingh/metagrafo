"""Handles configuration loading and validation."""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

class ConfigManager:
    """Manages application configuration."""
    
    DEFAULTS = {
        "environment": {
            "python_path": "/Users/tinosingh/Documents/whisper_workspace/W3/whisper/bin/python"
        },
        "tasks": {
            "transcription": {
                "model": "openai/whisper-base",
                "device": "auto"
            },
            "spellcheck": {
                "enabled": True,
                "model": "t5-base"
            },
            "summarization": {
                "enabled": True,
                "model": "facebook/bart-large-cnn"
            }
        },
        "audio": {
            "sample_rate": 16000,
            "chunk_duration_sec": 3.0,
            "silence_threshold": 0.03
        },
        "debug": {
            "enabled": False,
            "log_level": "INFO"
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize with optional config file path."""
        self.config_path = Path(config_path) if config_path else None
        self.config = self.DEFAULTS.copy()
        
        if self.config_path and self.config_path.exists():
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file, merging with defaults."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
                self._merge_configs(file_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load config file: {e}")
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for a specific task.
        
        Args:
            task_name: Name of the task (e.g., 'transcription', 'spellcheck')
            
        Returns:
            Dictionary containing the task configuration
        """
        return self.config.get("tasks", {}).get(task_name, {})
        
    def _merge_configs(self, new_config: Dict[str, Any]) -> None:
        """Recursively merge new config with existing config."""
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self._merge_configs(value)
            else:
                self.config[key] = value
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current config to file."""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No path provided to save config")
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation key."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set config value by dot notation key."""
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
    
    def update_task_config(self, task_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific task."""
        if "tasks" not in self.config:
            self.config["tasks"] = {}
        if task_name not in self.config["tasks"]:
            self.config["tasks"][task_name] = {}
            
        self.config["tasks"][task_name].update(updates)
