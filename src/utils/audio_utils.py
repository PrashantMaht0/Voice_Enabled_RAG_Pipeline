import numpy as np

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resamples audio to the target sample rate."""
    if orig_sr == target_sr:
        return audio_data
        
    try:
        import librosa
        # librosa is preferred for high-quality audio resampling
        return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback to standard linear interpolation if librosa isn't installed
        duration = len(audio_data) / orig_sr
        target_length = int(duration * target_sr)
        
        return np.interp(
            np.linspace(0.0, 1.0, target_length),
            np.linspace(0.0, 1.0, len(audio_data)),
            audio_data
        ).astype(audio_data.dtype)

def int16_to_float32(audio_data: np.ndarray) -> np.ndarray:
    """Converts 16-bit PCM audio to 32-bit float."""
    # Normalize by dividing by the maximum 16-bit integer value
    return audio_data.astype(np.float32) / 32768.0

def float32_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """Converts 32-bit float audio to 16-bit PCM."""
    # Clip to prevent distortion/clipping, then scale up
    clipped = np.clip(audio_data, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)