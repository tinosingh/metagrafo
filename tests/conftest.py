"""Configuration and fixtures for tests."""
import sys
from pathlib import Path

# Add parent directory to path before local imports
sys.path.append(str(Path(__file__).parent.parent))

# Standard library imports
import json

# Third-party imports
import numpy as np
import pytest

@pytest.fixture
def audio_sample():
    """Generate a sample audio signal for testing."""
    # 1 second of random audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "test_config.json"
    config_data = {
        "environment": {
            "python_path": "/test/path"
        },
        "tasks": {
            "transcription": {
                "model": "test-model"
            }
        },
        "audio": {
            "sample_rate": 16000
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    return str(config_path)
