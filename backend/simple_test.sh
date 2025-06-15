#!/bin/bash
# This script tests five of the largest cached models with temperature=0.0 and best_of=5.

# List of models (without the "whisper-" prefix; the loader will add it)
models=(
    "large-finnish-v3"   # becomes whisper-large-finnish-v3 -> repository: Finnish-NLP
    "large-mlx"          # becomes whisper-large-mlx -> repository: mlx-community
    "large-v3-mlx"       # becomes whisper-large-v3-mlx -> repository: mlx-community
    "large-v3-mlx-4bit"  # becomes whisper-large-v3-mlx-4bit -> repository: mlx-community
    "large-v3"           # becomes whisper-large-v3 -> repository: openai
)

# Path to your audio file (adjust as necessary)
# Corrected path: Assumes script is run from W2/backend and audio is in W2/frontend
AUDIO_FILE="../frontend/punainen_linnake.mp3"

# Default inference parameters
TEMPERATURE="0.0"
BEST_OF="5"

echo "Testing top 5 cached models with temperature=${TEMPERATURE} and best_of=${BEST_OF}..."
for model in "${models[@]}"; do
    echo "--------------------------------------------"
    echo "Testing model: $model"
    output_file="transcription_whisper-$model.txt"
    
    # Run the Python script, directing its stdout to the output file
    python core/model_loader.py --model "$model" --audio "$AUDIO_FILE" --temperature "$TEMPERATURE" --best_of "$BEST_OF" --language "fi" --fp16 > "$output_file" 2>&1
    
    # Check the exit status of the python command
    if [ $? -eq 0 ]; then
        echo "Output saved to ${output_file}"
    else
        echo "Error running model: $model. See ${output_file} for details."
    fi
done
echo "--------------------------------------------"
echo "All tests complete."

