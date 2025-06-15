"""Test script for transcription endpoint."""

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from tqdm import tqdm


def test_transcription():
    print("Starting test within virtual environment...")
    url = "http://localhost:9001/transcribe"
    test_file = (
        "/Users/tinosingh/Documents/whisper_workspace/W2/frontend/punainen_linnake.mp3"
    )

    with tqdm(total=100, desc="Test Progress") as pbar:
        # Stage 1: Prepare request
        multipart_data = MultipartEncoder(
            fields={
                "file": ("punainen_linnake.mp3", open(test_file, "rb"), "audio/mpeg"),
                "model": "tiny",
            }
        )
        pbar.update(30)

        # Stage 2: Send request
        response = requests.post(
            url,
            data=multipart_data,
            headers={"Content-Type": multipart_data.content_type},
        )
        pbar.update(60)

        # Stage 3: Process results
        if response.status_code == 200:
            print("\nTest successful! Response:")
            print(response.json())
        else:
            print(f"\nTest failed with status {response.status_code}")
            print(response.text)
        pbar.update(10)

    print("Test completed")


if __name__ == "__main__":
    test_transcription()
