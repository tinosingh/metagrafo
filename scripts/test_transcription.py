#!/usr/bin/env python3
"""
Test script for transcription functionality.

This script demonstrates how to use the transcription system with Whisper models.
"""

import argparse
import sys
import time
from pathlib import Path

# Third-party imports
import numpy as np
import soundfile as sf

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from realtime_transcription.core.task_manager import TaskManager
from realtime_transcription.utils.logging_utils import get_logger, setup_logging

# Set up logging
setup_logging(console=True, json_format=False)
logger = get_logger(__name__)

class TranscriptionTester:
    """Class to test transcription functionality."""
    
    def __init__(self, model_name: str = "base", device: str = "auto"):
        """Initialize with model parameters."""
        self.task_manager = TaskManager()
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def load_model(self) -> bool:
        """Load the transcription model."""
        logger.info(f"Loading model: {self.model_name}")
        # TaskManager initializes models automatically, just load the specific model
        return self.task_manager.load_model("transcription", self.model_name)
    
    def transcribe_audio_file(self, audio_file: str) -> dict:
        """Transcribe an audio file."""
        logger.info(f"Transcribing audio file: {audio_file}")
        
        # Load audio file
        try:
            audio_data, sample_rate = sf.read(audio_file)
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Process with the model
            result = self.task_manager.process(
                "transcription",
                audio_data,
                sample_rate=sample_rate,
                language="en"  # Optional: specify language
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {"error": str(e)}
    
    def real_time_transcription(self, duration: float = 10.0):
        """Test real-time transcription from microphone."""
        from core.audio_processor import AudioProcessor
        
        logger.info(f"Starting real-time transcription for {duration} seconds...")
        logger.info("Speak into your microphone...")
        
        audio_processor = AudioProcessor(sample_rate=16000, chunk_duration=3.0)
        
        def audio_callback(audio_data):
            """Process audio chunks and transcribe."""
            # Process with the model
            result = self.task_manager.process(
                "transcription",
                audio_data,
                sample_rate=16000,
                language="en"  # Optional: specify language
            )
            
            if "text" in result:
                print(f"\nTranscription: {result['text']}")
        
        try:
            # Start recording
            audio_processor.start_recording(callback=audio_callback)
            
            # Run for the specified duration
            time.sleep(duration)
            
        except KeyboardInterrupt:
            logger.info("Transcription interrupted by user")
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
        finally:
            # Clean up
            audio_processor.stop_recording()
            logger.info("Transcription completed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test transcription functionality")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # File transcription command
    file_parser = subparsers.add_parser("file", help="Transcribe an audio file")
    file_parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    file_parser.add_argument(
        "--model", 
        default="base",
        help="Model to use for transcription (default: base)"
    )
    
    # Real-time transcription command
    realtime_parser = subparsers.add_parser("realtime", help="Test real-time transcription")
    realtime_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of the test in seconds (default: 10)"
    )
    realtime_parser.add_argument(
        "--model",
        default="base",
        help="Model to use for transcription (default: base)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the tester
    tester = TranscriptionTester(model_name=args.model)
    
    # Load the model
    if not tester.load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Run the appropriate command
    if args.command == "file":
        result = tester.transcribe_audio_file(args.audio_file)
        if "text" in result:
            print("\nTranscription:")
            print(result["text"])
    elif args.command == "realtime":
        tester.real_time_transcription(duration=args.duration)
