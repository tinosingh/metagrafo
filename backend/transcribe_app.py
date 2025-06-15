"""Streamlit application for audio transcription using Whisper models."""

# Standard library imports
import os
import tempfile
import subprocess
import re
from pathlib import Path
import logging
import time

# Third-party imports
import streamlit as st
import mlx.core as mx
import mlx_whisper.transcribe as mlx_whisper

# Configure MLX to use GPU on Apple Silicon
mx.set_default_device(mx.Device(mx.DeviceType.gpu))

# Suppress excessive logging from mlx_whisper for cleaner Streamlit output.
logging.getLogger("mlx_whisper").setLevel(logging.WARNING)


def preprocess_audio(audio_path):
    """
    Converts audio to 16kHz mono WAV format using ffmpeg.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        str: Path to the processed temporary WAV file.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        subprocess.CalledProcessError: If ffmpeg fails during processing.
    """
    if not os.path.exists(audio_path):
        st.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create a temporary file for the processed audio.
    # mkstemp creates a file and returns its file descriptor and path.
    # We close the file descriptor immediately as ffmpeg will write to the path.
    fd, processed_path = tempfile.mkstemp(suffix="_processed.wav")
    os.close(fd)

    try:
        # Run ffmpeg command to convert audio.
        # -y: Overwrite output file without asking.
        # -ac 1: Convert to mono audio.
        # -ar 16000: Set audio sample rate to 16kHz.
        # -c:a pcm_s16le: Force audio codec to PCM signed 16-bit little-endian (widely compatible).
        # -f wav: Output format is WAV.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                processed_path,
            ],
            check=True,  # Raise CalledProcessError if return code is non-zero.
            capture_output=True,  # Capture stdout and stderr.
            text=True,  # Decode stdout/stderr as text.
        )
    except subprocess.CalledProcessError as e:
        # Log and display detailed ffmpeg error.
        st.error("FFmpeg error during audio preprocessing:")
        st.code(f"Command: {' '.join(e.cmd)}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        raise  # Re-raise the exception to be caught by the main logic.

    return processed_path


def clean_transcription(text):
    """
    Cleans up whitespace and punctuation in the transcription text.

    Args:
        text (str): The raw transcription text.

    Returns:
        str: The cleaned transcription text.
    """
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space.
    text = re.sub(
        r"([.,!?])(?=\S)", r"\1 ", text
    )  # Add space after punctuation if followed by non-space char.
    return text.strip()  # Remove leading/trailing whitespace.


def display_model_selection():
    """Displays model selection UI."""
    model_options = {
        "mlx-medium": "mlx-community/whisper-medium-mlx",
        "mlx-large-v3": "mlx-community/whisper-large-v3-mlx",
        "mlx-large-v3-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "mlx-tiny": "mlx-community/whisper-tiny-mlx",
    }
    selected_model = st.selectbox(
        "Select Whisper Model:",
        options=list(model_options.keys()),
        index=0,
        help="Choose an MLX-optimized Whisper model for transcription.",
    )
    return model_options[selected_model]


def main():
    st.set_page_config(
        layout="centered", page_title="Finnish Audio Transcription", page_icon="📝"
    )
    st.title("Finnish Audio Transcription")
    st.markdown(
        "Transcribe Finnish audio using `mlx-whisper` models optimized for Apple Silicon."
    )

    model_id = display_model_selection()

    # File upload widget
    audio_file = st.file_uploader(
        "Upload an audio file (.wav, .mp3 supported)",
        type=["wav", "mp3"],
        help="Please upload a Finnish audio file for transcription.",
    )

    # Initialize paths to None for robust cleanup in finally blocks
    tmp_input_path = None
    processed_audio_path = None

    if audio_file:
        st.info(f"Uploaded: {audio_file.name}")

        # Save the uploaded file to a temporary location
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(audio_file.name).suffix
            ) as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_input_path = tmp_file.name
            st.info(f"Temporary input file saved at: {tmp_input_path}")

            # Preprocess audio (convert to 16kHz mono WAV)
            with st.spinner("Preprocessing audio (converting to 16kHz mono WAV)..."):
                processed_audio_path = preprocess_audio(tmp_input_path)
            st.success("Audio preprocessing complete.")

            # Transcribe the processed audio
            with st.spinner(
                f"Transcribing with {model_id} (this may take a while for first run)..."
            ):
                try:
                    start_time = time.time()
                    # Correct call to mlx_whisper.transcribe
                    # It expects the audio path as the first positional argument,
                    # and the model ID via 'path_or_hf_repo' keyword argument.
                    result = mlx_whisper.transcribe(
                        processed_audio_path,
                        path_or_hf_repo=model_id,  # Pass the model ID here
                        language="fi",  # Explicitly set language to Finnish
                        temperature=0.0,
                        best_of=5,
                        fp16=True,  # Enable FP16 for Apple Silicon optimization
                        verbose=False,  # Suppress internal mlx_whisper verbose output
                    )
                    transcription_duration = time.time() - start_time

                    # Get and clean transcription text
                    transcription = result["text"]
                    transcription = clean_transcription(transcription)

                    st.success(
                        f"Transcription Complete! (Took {transcription_duration:.2f} seconds)"
                    )
                    st.text_area("Transcription Result", transcription, height=250)

                    # Download button for transcription
                    st.download_button(
                        label="Download Transcription",
                        data=transcription,
                        file_name=f"{Path(audio_file.name).stem}_transcription.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    st.exception(e)  # Display full traceback in Streamlit
                finally:
                    # Clean up temporary files
                    # Ensure paths exist before attempting to delete
                    if tmp_input_path and os.path.exists(tmp_input_path):
                        os.unlink(tmp_input_path)
                        st.info(f"Cleaned up temporary input file: {tmp_input_path}")
                    if processed_audio_path and os.path.exists(processed_audio_path):
                        os.unlink(processed_audio_path)
                        st.info(
                            f"Cleaned up temporary processed audio file: {processed_audio_path}"
                        )

        except Exception as e:
            st.error(f"Error during file processing: {str(e)}")
            st.exception(e)  # Display full traceback in Streamlit
            # Ensure the initial temp input file is cleaned up even if preprocessing fails
            if tmp_input_path and os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
                st.info(f"Cleaned up temporary input file: {tmp_input_path}")


if __name__ == "__main__":
    main()
