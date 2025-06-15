import argparse
from backend.transcription import transcribe_audio


def transcribe_file(file_path, model_size="base"):
    """Transcribe an audio file without microphone."""
    try:
        result = transcribe_audio(audio_path=file_path, model_size=model_size)
        print("\nTranscription:")
        print(result["text"])
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    args = parser.parse_args()

    transcribe_file(args.file, args.model)
