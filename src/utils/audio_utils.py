"""Audio format helpers: resampling and conversion between 16 bit PCM and float samples."""
import numpy as np


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """The audio at target_sr (Whisper expects 16 kHz); unchanged when the rates already match."""
    if orig_sr == target_sr:
        return audio_data
    import librosa                        # imported here: it is slow to import and only voice needs it
    return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)


def int16_to_float32(audio_data: np.ndarray) -> np.ndarray:
    """16 bit PCM samples as floats in [-1, 1)."""
    return audio_data.astype(np.float32) / 32768.0


def float32_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """Float samples as 16 bit PCM, clipped to [-1, 1] first so loud audio does not wrap around."""
    clipped = np.clip(audio_data, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)
