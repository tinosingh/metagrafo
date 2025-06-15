import base64
import numpy as np

# Read first 10 seconds of audio
with open("frontend/punainen_linnake.mp3", "rb") as f:
    audio_bytes = f.read(16000 * 10)  # Approx 10 seconds

# Convert to base64 and back
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
decoded_bytes = base64.b64decode(audio_base64)

try:
    # Try converting to numpy array
    audio_array = np.frombuffer(decoded_bytes, dtype=np.float32)
    print(f"Success! Audio array shape: {audio_array.shape}")
except Exception as e:
    print(f"Error converting audio: {str(e)}")
