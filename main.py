"""
Main entry point for the real-time transcription system.
"""
import argparse
import time

import numpy as np
from colorama import Fore, Style, init as init_colorama

# Import from our package
from realtime_transcription.config.config_manager import ConfigManager
from realtime_transcription.core.task_manager import TaskManager
from realtime_transcription.core.audio_processor import AudioProcessor

# Initialize colorama
init_colorama(autoreset=True)

class RealTimeTranscriber:
    """Orchestrates real-time transcription using modular components."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the transcriber with configuration."""
        # Initialize components
        self.config = ConfigManager(config_path)
        self.task_manager = TaskManager()
        
        # Set up audio processing
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.get("audio.sample_rate"),
            chunk_duration=self.config.get("audio.chunk_duration_sec")
        )
        
        # Load models for different tasks
        self._load_models()
        
        # Performance tracking
        self.performance = {
            "audio_chunks_processed": 0,
            "transcription_time": 0.0,
            "last_update": time.time()
        }
    
    def _load_models(self) -> None:
        """Load models for different tasks based on config."""
        print(f"{Fore.YELLOW}Loading models...{Style.RESET_ALL}")
        
        # Load transcription model
        trans_config = self.config.get_task_config("transcription")
        if not self.task_manager.load_model("transcription", trans_config["model"]):
            print(f"{Fore.RED}Failed to load transcription model{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Loaded transcription model: {trans_config['model']}{Style.RESET_ALL}")
        
        # Load other models (spellcheck, summarization) would go here
        
        print(f"{Fore.GREEN}All models loaded{Style.RESET_ALL}")
    
    def _audio_callback(self, audio_data: np.ndarray) -> None:
        """Process audio data from the audio processor."""
        start_time = time.time()
        
        try:
            # Process audio with transcription model
            result = self.task_manager.process(
                "transcription",
                audio_data,
                language="en",  # Make configurable
                fp16=False  # Disable FP16 for now as it requires CUDA
            )
            
            if "error" in result:
                print(f"{Fore.RED}Transcription error: {result['error']}{Style.RESET_ALL}")
                return
            
            # Display transcription
            print(f"\n{Fore.CYAN}Transcription:{Style.RESET_ALL}")
            print(f"{result['text']}")
            
            # Update performance metrics
            self.performance["audio_chunks_processed"] += 1
            self.performance["transcription_time"] += time.time() - start_time
            
        except Exception as e:
            print(f"{Fore.RED}Error processing audio: {str(e)}{Style.RESET_ALL}")
    
    def run(self) -> None:
        """Run the transcription system."""
        print(f"{Fore.GREEN}Starting real-time transcription...{Style.RESET_ALL}")
        print("Press Ctrl+C to stop")
        
        try:
            self.audio_processor.start_recording(callback=self._audio_callback)
            
            # Keep the main thread alive
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            self.audio_processor.stop_recording()
            
            # Print performance summary
            print(f"\n{Fore.YELLOW}Performance Summary:{Style.RESET_ALL}")
            print(f"Audio chunks processed: {self.performance['audio_chunks_processed']}")
            print(f"Total processing time: {time.time() - self.performance['last_update']:.2f} seconds")
            print(f"Total processing time: {self.performance['transcription_time']:.2f}s")
            print(f"Average processing time per chunk: {self.performance['transcription_time']/max(1, self.performance['audio_chunks_processed']):.3f}s")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time Whisper transcription")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to config file")
    args = parser.parse_args()
    
    transcriber = RealTimeTranscriber(config_path=args.config)
    transcriber.run()

if __name__ == "__main__":
    main()
