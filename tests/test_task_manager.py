"""Tests for the TaskManager class."""
import pytest
from unittest.mock import MagicMock
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.task_manager import TaskManager
from models.whisper_model import WhisperModel

class TestTaskManager:
    """Test suite for the TaskManager class."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a TaskManager instance for testing."""
        return TaskManager()
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Create a mock WhisperModel for testing."""
        model = MagicMock(spec=WhisperModel)
        model.load_model.return_value = True
        model.process.return_value = {"text": "test transcription"}
        model.get_available_models.return_value = ["base", "small", "medium"]
        model.model_info.return_value = {
            "name": "whisper-base",
            "task": "transcription",
            "supports_language_detection": True
        }
        return model
    
    def test_initialization(self, task_manager):
        """Test TaskManager initialization."""
        assert task_manager.tasks == {}
        assert task_manager.available_models == {}
    
    def test_register_model(self, task_manager, mock_whisper_model):
        """Test registering a model with the task manager."""
        # Register a model
        task_manager.register_model("whisper", "transcription", mock_whisper_model)
        
        # Check registration
        assert "whisper" in task_manager.available_models
        assert task_manager.available_models["whisper"] == mock_whisper_model
        
        # Check task mapping
        assert "transcription" in task_manager.tasks
        assert task_manager.tasks["transcription"] == ["whisper"]
    
    def test_load_model_success(self, task_manager, mock_whisper_model):
        """Test successfully loading a model."""
        # Register the model first
        task_manager.register_model("whisper", "transcription", mock_whisper_model)
        
        # Load the model
        result = task_manager.load_model("transcription", "whisper")
        
        # Check results
        assert result is True
        mock_whisper_model.load_model.assert_called_once()
        assert task_manager.active_models["transcription"] == mock_whisper_model
    
    def test_load_model_not_registered(self, task_manager):
        """Test loading a model that hasn't been registered."""
        # Try to load an unregistered model
        result = task_manager.load_model("transcription", "nonexistent")
        
        # Should return False
        assert result is False
    
    def test_process_success(self, task_manager, mock_whisper_model):
        """Test processing audio with a loaded model."""
        # Setup
        task_manager.register_model("whisper", "transcription", mock_whisper_model)
        task_manager.load_model("transcription", "whisper")
        
        # Test data
        audio_data = b"test audio data"
        
        # Process
        result = task_manager.process("transcription", audio_data, language="en")
        
        # Verify
        assert result == {"text": "test transcription"}
        mock_whisper_model.process.assert_called_once_with(
            audio_data, 
            language="en"
        )
    
    def test_process_no_model_loaded(self, task_manager):
        """Test processing when no model is loaded for the task."""
        # Try to process without loading a model
        result = task_manager.process("transcription", b"test")
        
        # Should return an error message
        assert "error" in result
        assert "No model loaded for task" in result["error"]
    
    def test_get_available_models_for_task(self, task_manager, mock_whisper_model):
        """Test getting available models for a specific task."""
        # Register model for transcription task
        task_manager.register_model("whisper", "transcription", mock_whisper_model)
        
        # Get available models
        models = task_manager.get_available_models("transcription")
        
        # Should return the registered model
        assert models == ["whisper"]
    
    def test_get_available_models_all(self, task_manager, mock_whisper_model):
        """Test getting all available models."""
        # Register a model
        task_manager.register_model("whisper", "transcription", mock_whisper_model)
        
        # Get all models
        models = task_manager.get_available_models()
        
        # Should return all registered models
        assert models == {"whisper": mock_whisper_model}
    
    def test_get_model_info(self, task_manager, mock_whisper_model):
        """Test getting model information."""
        # Register and load model
        task_manager.register_model("whisper", "transcription", mock_whisper_model)
        task_manager.load_model("transcription", "whisper")
        
        # Get model info
        info = task_manager.get_model_info("transcription")
        
        # Should return model info
        assert info == {
            "name": "whisper-base",
            "task": "transcription",
            "supports_language_detection": True
        }
    
    def test_get_model_info_not_loaded(self, task_manager):
        """Test getting info for a model that isn't loaded."""
        # Get info for non-loaded model
        info = task_manager.get_model_info("nonexistent")
        
        # Should return None
        assert info is None

if __name__ == "__main__":
    pytest.main([__file__])
