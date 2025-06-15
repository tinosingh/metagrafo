#!/usr/bin/env python3
"""
Utility to scan for available Whisper models and update the model list.
"""
from pathlib import Path
from model_manager import ModelManager
from config_manager import ConfigManager

def main():
    print("Scanning for available Whisper models in Hugging Face cache...")
    
    # Initialize managers
    config_path = Path(__file__).parent / "config.json"
    config_manager = ConfigManager(config_path)
    model_manager = ModelManager(config_manager)
    
    # Scan models
    result = model_manager.scan_models()
    
    # Get available models
    available_models = list(model_manager.get_available_models().keys())
    print(f"\nFound {len(available_models)} models:")
    for model in available_models:
        print(f"- {model}")
    
    print(f"\nLast scan: {result.get('last_scan', 'unknown')}")
    
    if result.get("warnings"):
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

if __name__ == "__main__":
    main()
