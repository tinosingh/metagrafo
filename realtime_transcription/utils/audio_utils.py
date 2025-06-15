"""Audio processing utilities."""
import numpy as np
from typing import Optional, Tuple

def normalize_audio(
    audio: np.ndarray,
    target_dBFS: float = -20.0,
    max_amplitude: float = 0.95
) -> np.ndarray:
    """
    Normalize audio to a target dBFS level.
    
    Args:
        audio: Input audio signal as a numpy array
        target_dBFS: Target loudness in dBFS
        max_amplitude: Maximum allowed amplitude (0.0 to 1.0)
        
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
        
    # Calculate current loudness (RMS)
    current_dBFS = 20 * np.log10(np.maximum(1e-6, np.sqrt(np.mean(audio**2))))
    
    # Calculate gain to reach target dBFS
    gain = 10 ** ((target_dBFS - current_dBFS) / 20)
    
    # Apply gain and clip to prevent clipping
    normalized = np.clip(audio * gain, -max_amplitude, max_amplitude)
    
    return normalized

def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to a target sample rate.
    
    Args:
        audio: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio signal
    """
    if orig_sr == target_sr:
        return audio
        
    # Calculate new length
    duration = len(audio) / orig_sr
    new_length = int(duration * target_sr)
    
    # Resample using linear interpolation
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, new_length)
    
    return np.interp(x_new, x_old, audio)

def split_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float,
    overlap: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Split audio into overlapping or non-overlapping chunks.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks (0.0 to 1.0)
        
    Returns:
        Tuple of (chunks, num_chunks)
    """
    if len(audio) == 0:
        return np.array([]), 0
        
    samples_per_chunk = int(chunk_duration * sample_rate)
    hop_size = int(samples_per_chunk * (1 - overlap))
    
    if hop_size <= 0:
        raise ValueError("Overlap must be less than 100%")
    
    # Pad audio if needed to make it evenly divisible by hop_size
    padding = (samples_per_chunk - (len(audio) - samples_per_chunk) % hop_size) % samples_per_chunk
    padded = np.pad(audio, (0, padding), mode='constant')
    
    # Split into chunks
    num_chunks = (len(padded) - samples_per_chunk) // hop_size + 1
    chunks = np.array([
        padded[i*hop_size : i*hop_size + samples_per_chunk]
        for i in range(num_chunks)
    ])
    
    return chunks, num_chunks

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging channels.
    
    Args:
        audio: Input audio signal (shape: [samples] or [channels, samples])
        
    Returns:
        Mono audio signal (shape: [samples])
    """
    if len(audio.shape) == 1:
        return audio  # Already mono
    return np.mean(audio, axis=0)

def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.5
) -> np.ndarray:
    """
    Detect silent regions in audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        threshold_db: Threshold in dB below which audio is considered silent
        min_silence_duration: Minimum duration of silence to detect (seconds)
        
    Returns:
        Boolean array where True indicates silent regions
    """
    if len(audio) == 0:
        return np.array([], dtype=bool)
    
    # Convert to power and then to dB
    power = audio ** 2
    db = 10 * np.log10(np.maximum(1e-10, power))
    
    # Find silent regions
    is_silent = db < threshold_db
    
    # Apply minimum duration filter
    min_samples = int(min_silence_duration * sample_rate)
    if min_samples > 1:
        # Use a moving window to find regions where all samples are silent
        window = np.ones(min_samples, dtype=bool)
        is_silent = np.convolve(is_silent, window, mode='same') >= min_samples
    
    return is_silent

def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None
) -> np.ndarray:
    """
    Compute the magnitude spectrogram of an audio signal.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length between frames
        win_length: Window length for each frame
        
    Returns:
        Magnitude spectrogram (freq_bins, time_frames)
    """
    import librosa
    
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    
    # Compute STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True
    )
    
    # Convert to magnitude
    spectrogram = np.abs(stft)
    
    return spectrogram
