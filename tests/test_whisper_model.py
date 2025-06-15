"""Tests for the WhisperModel class."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.whisper_model import WhisperModel

class TestWhisperModel:
    """Test suite for the WhisperModel class."""
    
    @pytest.fixture
    def whisper_model(self):
        """Create a WhisperModel instance for testing."""
        return WhisperModel()
    
    @patch('whisper.load_model')
    def test_load_model_success(self, mock_load_model, whisper_model):
        """Test successfully loading a Whisper model."""
        # Setup mock
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Execute
        result = whisper_model.load_model("base")
        
        # Verify
        assert result is True
        mock_load_model.assert_called_once_with("base")
        assert whisper_model.model == mock_model
        assert whisper_model.model_name == "base"
    
    @patch('whisper.load_model')
    def test_load_model_failure(self, mock_load_model, whisper_model):
        """Test failing to load a Whisper model."""
        # Setup mock to raise an exception
        mock_load_model.side_effect = Exception("Failed to load model")
        
        # Execute
        result = whisper_model.load_model("nonexistent")
        
        # Verify
        assert result is False
        assert whisper_model.model is None
    
    @patch('numpy.array')
    @patch('whisper.load_model')
    def test_process_audio(self, mock_load_model, mock_np_array, whisper_model):
        """Test processing audio data with Whisper."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test transcription"}
        mock_load_model.return_value = mock_model
        
        # Load the model first
        whisper_model.load_model("base")
        
        # Test data
        audio_data = np.random.rand(16000)  # 1 second of random audio
        
        # Execute
        result = whisper_model.process(audio_data, language="en")
        
        # Verify
        assert result == {"text": "test transcription"}
        mock_model.transcribe.assert_called_once()
        
        # Check that the audio data was converted to float32
        args, kwargs = mock_model.transcribe.call_args
        assert kwargs["language"] == "en"
        assert kwargs["fp16"] is False
    
    @patch('whisper.load_model')
    def test_process_audio_no_model(self, mock_load_model, whisper_model):
        """Test processing audio when no model is loaded."""
        # Try to process without loading a model
        result = whisper_model.process(np.random.rand(16000))
        
        # Should return an error message
        assert "error" in result
        assert "Model not loaded" in result["error"]
    
    @patch('whisper.available_models')
    def test_get_available_models(self, mock_available_models, whisper_model):
        """Test getting available Whisper models."""
        # Setup mock
        mock_available_models.return_value = ["tiny", "base", "small"]
        
        # Execute
        models = whisper_model.get_available_models()
        
        # Verify
        assert models == ["tiny", "base", "small"]
    
    @patch('whisper.available_models')
    def test_model_info(self, mock_available_models, whisper_model):
        """Test getting model information."""
        # Setup mock
        mock_available_models.return_value = ["tiny", "base", "small"]
        
        # Load a model first
        with patch('whisper.load_model'):
            whisper_model.load_model("base")
        
        # Get model info
        info = whisper_model.model_info()
        
        # Verify
        assert info == {
            "name": "base",
            "type": "whisper",
            "supports_language_detection": True,
            "available_models": ["tiny", "base", "small"]
        }
    
    @patch('whisper.load_model')
    def test_process_with_language_detection(self, mock_load_model, whisper_model):
        """Test processing with automatic language detection."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "test transcription",
            "language": "en"
        }
        mock_load_model.return_value = mock_model
        
        # Load the model first
        whisper_model.load_model("base")
        
        # Test data
        audio_data = np.random.rand(16000)
        
        # Execute with language=None to trigger detection
        result = whisper_model.process(audio_data, language=None)
        
        # Verify
        assert result == {
            "text": "test transcription",
            "language": "en"
        }
        mock_model.transcribe.assert_called_once()
        
        # Check that language was not passed to transcribe (let Whisper detect)
        args, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs

if __name__ == "__main__":
    pytest.main([__file__])
