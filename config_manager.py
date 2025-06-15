"""
Handles all configuration loading and validation for the transcription system.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

class ConfigManager:
    """Manages application configuration with validation."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {
            "environment": {},
            "paths": {},
            "defaults": {},
            "globals": {}
        }
        self._load_config()
    
    def _load_config(self):
        """Load and validate configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                loaded_config = json.load(f)
                
            # Merge with defaults while preserving structure
            for section in self.config.keys():
                if section in loaded_config:
                    self.config[section].update(loaded_config[section])
        except Exception as e:
            print(f"Failed to load config: {e}")
    
    def get_environment(self, key: str, default: Any = None) -> Any:
        """Get environment configuration value."""
        return self.config["environment"].get(key, default)
    
    def get_paths(self, key: str, default: Any = None) -> Any:
        """Get paths configuration value."""
        return self.config["paths"].get(key, default)
    
    def get_defaults(self, key: str, default: Any = None) -> Any:
        """Get defaults configuration value."""
        return self.config["defaults"].get(key, default)
    
    def get_globals(self, key: str, default: Any = None) -> Any:
        """Get globals configuration value."""
        return self.config["globals"].get(key, default)
    
    def get_cached_models(self) -> List[str]:
        """Get list of available cached models."""
        return []  # Now handled by ModelManager

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values while preserving existing ones.
        
        Args:
            updates: Dictionary of config sections to update
        """
        with open(self.config_path, 'r+') as f:
            config = json.load(f)
            for section, values in updates.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
            f.seek(0)
            json.dump(config, f, indent=2)
            f.truncate()
