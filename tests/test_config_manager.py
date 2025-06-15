"""Tests for the ConfigManager class."""
import os
import json
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager

class TestConfigManager:
    """Test suite for the ConfigManager class."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
            json.dump({
                "environment": {
                    "python_path": "/test/path"
                },
                "tasks": {
                    "transcription": {
                        "model": "test-model"
                    }
                }
            }, f)
            f.flush()
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        config = ConfigManager()
        
        # Check some default values
        assert config.get("tasks.transcription.model") == "openai/whisper-base"
        assert config.get("audio.sample_rate") == 16000
        assert config.get("debug.enabled") is False
    
    def test_initialization_with_config_file(self, temp_config_file):
        """Test initialization with a config file."""
        config = ConfigManager(temp_config_file)
        
        # Check values from config file
        assert config.get("environment.python_path") == "/test/path"
        assert config.get("tasks.transcription.model") == "test-model"
        
        # Check that defaults are still available for non-specified values
        assert config.get("audio.sample_rate") == 16000
    
    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = ConfigManager()
        
        # Test getting nested values
        assert config.get("tasks.transcription.device") == "auto"
        assert config.get("tasks.transcription.fp16") is False
        
        # Test getting non-existent value with default
        assert config.get("nonexistent.key", "default") == "default"
    
    def test_set_nested_value(self):
        """Test setting nested configuration values."""
        config = ConfigManager()
        
        # Set a new nested value
        config.set("test.nested.value", 42)
        
        # Verify it was set correctly
        assert config.get("test.nested.value") == 42
        
        # Update an existing value
        config.set("tasks.transcription.device", "cuda")
        assert config.get("tasks.transcription.device") == "cuda"
    
    def test_get_task_config(self, temp_config_file):
        """Test getting configuration for a specific task."""
        config = ConfigManager(temp_config_file)
        
        # Get task config
        task_config = config.get_task_config("transcription")
        
        # Verify
        assert task_config["model"] == "test-model"
        
        # Verify it's a copy, not a reference
        task_config["model"] = "new-model"
        assert config.get("tasks.transcription.model") == "test-model"
    
    def test_update_task_config(self, temp_config_file):
        """Test updating configuration for a specific task."""
        config = ConfigManager(temp_config_file)
        
        # Update task config
        updates = {
            "model": "updated-model",
            "device": "cuda"
        }
        config.update_task_config("transcription", updates)
        
        # Verify updates
        assert config.get("tasks.transcription.model") == "updated-model"
        assert config.get("tasks.transcription.device") == "cuda"
        
        # Verify other tasks are not affected
        assert config.get("tasks.spellcheck.model") == "t5-base"
    
    def test_save_config(self, temp_config_file):
        """Test saving configuration to a file."""
        # Create a new temp file for saving
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Initialize with original config
            config = ConfigManager(temp_config_file)
            
            # Modify a value
            config.set("test.save", True)
            
            # Save to new location
            config.save_config(temp_path)
            
            # Load the saved config
            with open(temp_path, 'r') as f:
                saved_config = json.load(f)
            
            # Verify the saved config
            assert saved_config["test"]["save"] is True
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__])
