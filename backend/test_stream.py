"""Test real-time transcription streaming."""

import asyncio

import websockets


async def test_stream():
    uri = "ws://localhost:9001/stream"
    async with websockets.connect(uri) as websocket:
        print("Streaming started - speak into microphone")
        while True:
            transcript = await websocket.recv()
            print(f"Transcription: {transcript}")


if __name__ == "__main__":
    asyncio.run(test_stream())
