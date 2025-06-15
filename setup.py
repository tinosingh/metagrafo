from setuptools import setup, find_packages

setup(
    name="realtime_transcription",
    version="0.1.0",
    packages=find_packages(include=["realtime_transcription", "realtime_transcription.*"]),
    install_requires=[
        "numpy",
        "torch",
        "torchaudio",
        "whisper",
        "sounddevice",
        "soundfile",
        "librosa",
        "transformers",
        "huggingface-hub",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "transcribe=realtime_transcription.main:main",
        ],
    },
)
