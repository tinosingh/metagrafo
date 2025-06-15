import numpy as np
import sounddevice as sd


def test_microphone():
    print("Testing microphone access...")
    print("\nAvailable devices:")
    print(sd.query_devices())

    try:
        duration = 3  # seconds
        fs = 16000
        print(f"\nRecording for {duration} seconds...")

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished

        print("Recording complete. Check if audio was captured:")
        print(f"- Audio shape: {recording.shape}")
        print(f"- Max amplitude: {np.max(np.abs(recording)):.4f}")

        if np.max(np.abs(recording)) < 0.001:
            print("\nWARNING: Very quiet recording - check microphone")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. MacOS: System Preferences > Security & Privacy > Microphone")
        print("2. Terminal: Add terminal app to microphone access list")
        print("3. Try: sd.default.device = 'BlackHole' or other device name")


if __name__ == "__main__":
    test_microphone()
