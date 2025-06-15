#!/usr/bin/env python3
"""
Test script for audio processing functionality.

This script demonstrates how to use the AudioProcessor class to record and process
audio in real-time.
"""
import argparse
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import from our package
from realtime_transcription.core.audio_processor import AudioProcessor
from realtime_transcription.utils.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging(console=True, json_format=False)
logger = get_logger(__name__)

class AudioTester:
    """Class to test audio processing functionality."""
    
    def __init__(self, sample_rate=16000, chunk_duration=3.0):
        """Initialize with audio parameters."""
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration
        )
        self.is_running = False
    
    def audio_callback(self, audio_data):
        """Process audio data from the audio processor."""
        if not self.is_running:
            return
            
        # Calculate some basic audio statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        
        # Log the statistics
        logger.info(f"Audio chunk - RMS: {rms:.6f}, Peak: {peak:.6f}")
        
        # You can add more processing here
        
    def run_test(self, duration=10):
        """Run the audio test for the specified duration."""
        logger.info(f"Starting audio test for {duration} seconds...")
        logger.info("Speak into your microphone...")
        
        try:
            # Start recording
            self.is_running = True
            self.audio_processor.start_recording(callback=self.audio_callback)
            
            # Run for the specified duration
            time.sleep(duration)
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Error during audio test: {str(e)}")
        finally:
            # Clean up
            self.is_running = False
            self.audio_processor.stop_recording()
            logger.info("Audio test completed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test audio processing functionality")
    parser.add_argument(
        "--duration", 
        type=float, 
        default=10.0,
        help="Duration of the test in seconds (default: 10)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Duration of each audio chunk in seconds (default: 3.0)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize and run the test
    tester = AudioTester(
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration
    )
    tester.run_test(duration=args.duration)
