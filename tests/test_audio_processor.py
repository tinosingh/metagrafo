"""Tests for the AudioProcessor class."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.audio_processor import AudioProcessor

def test_audio_processor_initialization():
    """Test AudioProcessor initialization with default parameters."""
    # Initialize with default parameters
    processor = AudioProcessor()
    
    # Check default values
    assert processor.sample_rate == 16000
    assert processor.chunk_duration == 3.0
    assert processor.channels == 1
    assert processor.chunk_size == 48000  # 3.0 * 16000
    assert not processor.is_recording
    assert processor.stream is None
    assert processor.callback is None

def test_audio_processor_custom_parameters():
    """Test AudioProcessor initialization with custom parameters."""
    # Initialize with custom parameters
    processor = AudioProcessor(
        sample_rate=44100,
        chunk_duration=5.0,
        channels=2
    )
    
    # Check custom values
    assert processor.sample_rate == 44100
    assert processor.chunk_duration == 5.0
    assert processor.channels == 2
    assert processor.chunk_size == 220500  # 5.0 * 44100

@patch('sounddevice.InputStream')
def test_start_recording(mock_stream):
    """Test starting audio recording."""
    # Setup
    processor = AudioProcessor()
    callback = MagicMock()
    
    # Execute
    processor.start_recording(callback)
    
    # Verify
    assert processor.is_recording
    assert processor.callback == callback
    mock_stream.assert_called_once()
    mock_stream.return_value.start.assert_called_once()

def test_stop_recording():
    """Test stopping audio recording."""
    # Setup
    processor = AudioProcessor()
    processor.stream = MagicMock()
    processor.is_recording = True
    
    # Execute
    processor.stop_recording()
    
    # Verify
    assert not processor.is_recording
    processor.stream.stop.assert_called_once()
    processor.stream.close.assert_called_once()
    assert processor.stream is None

def test_record_chunk():
    """Test recording a single audio chunk."""
    # Setup
    with patch('sounddevice.rec') as mock_rec:
        mock_rec.return_value = np.zeros((16000, 1))  # 1 second of silence
        processor = AudioProcessor(sample_rate=16000, chunk_duration=1.0)
        
        # Execute
        result = processor.record_chunk()
        
        # Verify
        assert result.shape == (16000,)  # Should be mono
        mock_rec.assert_called_once_with(
            int(1.0 * 16000),  # samples = duration * sample_rate
            samplerate=16000,
            channels=1,
            blocking=True
        )

def test_process_audio():
    """Test processing audio data."""
    # Setup
    processor = AudioProcessor(sample_rate=16000)
    audio_data = np.random.rand(16000)  # 1 second of random audio
    
    # Execute
    processed_audio, duration = processor.process_audio(audio_data)
    
    # Verify
    assert duration == 1.0  # 16000 samples / 16000 Hz = 1.0 second
    assert np.array_equal(processed_audio, audio_data)  # No processing done yet

@patch('sounddevice.play')
@patch('sounddevice.wait')
def test_play_audio(mock_wait, mock_play):
    """Test playing audio data."""
    # Setup
    processor = AudioProcessor(sample_rate=16000)
    audio_data = np.random.rand(16000)  # 1 second of random audio
    
    # Execute
    processor.play_audio(audio_data)
    
    # Verify
    mock_play.assert_called_once_with(audio_data, 16000)
    mock_wait.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
