# Real-Time Transcription with Whisper AI

A modular, real-time transcription system supporting multiple AI models for transcription, spell-checking, and summarization.

## Features

- 🎙️ Real-time audio transcription with Whisper models
- ✍️ Integrated spell-checking
- 📝 Automatic summarization of transcriptions
- 🎚️ Configurable model selection per task
- 📊 Performance monitoring and logging
- 🎛️ Modular architecture for easy extension

## Setup Instructions

1. **Activate Virtual Environment**:
   ```bash
   source /Users/tinosingh/Documents/whisper_workspace/W3/whisper/bin/activate
   ```

2. **Install Dependencies (using uv)**:
   ```bash
   uv pip install -r requirements.txt
   ```
   
   For GPU acceleration (recommended):
   ```bash
   uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Configuration**:
   - Edit `config.json` to customize:
     - Model selection for each task
     - Audio device and processing settings
     - Debug and logging options

4. **Running the Application**:
   ```bash
   # Basic usage
   python main.py
   
   # With custom config
   python main.py --config path/to/config.json
   ```

## Project Structure

```
W3/
├── config/                  # Configuration management
│   ├── __init__.py
│   └── config_manager.py     # Config loading and validation
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── audio_processor.py   # Audio capture and processing
│   └── task_manager.py      # Manages AI tasks and models
├── models/                  # AI model implementations
│   ├── __init__.py
│   ├── base_model.py        # Abstract base class for models
│   └── whisper_model.py     # Whisper transcription model
├── services/                # Service layer
│   └── __init__.py
├── config.json              # Application configuration
├── main.py                  # Entry point
└── requirements.txt         # Dependencies
```

## Configuration

### Tasks Configuration

Each task can be configured with its own model and settings:

```json
"tasks": {
  "transcription": {
    "model": "openai/whisper-base",
    "device": "auto",
    "language": "en",
    "fp16": false,
    "enabled": true
  },
  "spellcheck": {
    "model": "t5-base",
    "enabled": true
  },
  "summarization": {
    "model": "facebook/bart-large-cnn",
    "enabled": true,
    "max_length": 130,
    "min_length": 30
  }
}
```

### Audio Settings

```json
"audio": {
  "sample_rate": 16000,
  "chunk_duration_sec": 3.0,
  "channels": 1,
  "silence_threshold": 0.03
}
```

## Adding New Models

1. Create a new model class in `models/` that inherits from `BaseModel`
2. Implement the required methods:
   - `load_model()`
   - `process(audio_data, **kwargs)`
   - `get_available_models()`
   - `model_info()`
3. Register the model in the TaskManager

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses:
- Black for code formatting
- Mypy for static type checking
- Ruff for linting

Run formatting and checks:

```bash
black .
mypy .
ruff check .
```

## Troubleshooting

### Common Issues

1. **No Audio Input Detected**
   - Check your default audio input device
   - Ensure the sample rate matches your microphone's capabilities

2. **Model Loading Errors**
   - Verify internet connection for model downloads
   - Check available disk space in the cache directory
   - Ensure CUDA is properly configured for GPU acceleration

3. **Performance Issues**
   - Try a smaller model if using CPU
   - Adjust `chunk_duration_sec` in config
   - Enable FP16 if using CUDA

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
