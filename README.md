# Professional Meeting Transcription App

A web application for transcribing meetings with speaker diarization and domain-specific optimizations, built with Whisper for high-quality speech recognition.

## Features

### Core Features
- 🎤 Audio file upload and processing
- 🗣️ High-accuracy transcription using Whisper
- 👥 Speaker diarization (when available in audio)
- 📝 Clean, readable transcription output
- ⚡ Real-time transcription capabilities
- 🏷️ Domain-specific optimizations for better accuracy

### Technical Highlights
- 🐍 Python backend with FastAPI
- 🔄 Asynchronous processing
- 📊 Performance monitoring and optimization
- 🛠️ Comprehensive test suite
- 🧹 Code quality enforcement with pre-commit hooks

## Getting Started

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- FFmpeg (for audio processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/meeting-transcriber.git
   cd meeting-transcriber
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install pre-commit hooks (for development):
   ```bash
   pre-commit install
   ```

### Usage

1. Start the application:
   ```bash
   uvicorn backend.main:app --reload
   ```

2. Open your browser and navigate to `http://localhost:8000`

3. Upload an audio file and wait for the transcription to complete

## Development

### Project Structure

```
.
├── backend/                 # Backend application code
├── tests/                   # Test files
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── pyproject.toml           # Project configuration and dependencies
└── README.md                # This file
```

### Running Tests

```bash
pytest
```

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **isort** for import sorting

Format your code before committing:
```bash
black .
ruff check --fix .
isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Whisper for the speech recognition model
- All open-source libraries and tools used in this project
