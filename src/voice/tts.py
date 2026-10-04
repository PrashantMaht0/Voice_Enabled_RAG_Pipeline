import importlib.metadata
import os
import shutil
import tempfile

import numpy as np

try:
    import espeakng_loader
    import kokoro_onnx
    from kokoro_onnx.config import EspeakConfig
    import soundfile as sf
    KOKORO_INSTALLED = True
except ImportError:
    KOKORO_INSTALLED = False

from src.utils.audio_utils import float32_to_int16
from src.utils.config import pick

# espeak-ng (inside espeakng-loader) ignores a data folder whose real path is 160+ characters (it resolves
# symlinks), falls back to the path it was built with, and exits the whole process. A venv in a deep folder
# hits this, so a copy of the data (about 19 MB) is made once in a short temp folder.
ESPEAK_PATH_LIMIT = 160


def espeak_data_path() -> str:
    path = espeakng_loader.get_data_path()
    if len(os.path.realpath(path)) < ESPEAK_PATH_LIMIT:
        return path
    version = importlib.metadata.version("espeakng-loader")
    short = os.path.join(tempfile.gettempdir(), f"vrag-espeak-ng-data-{version}")
    if len(os.path.realpath(short)) >= ESPEAK_PATH_LIMIT:
        raise RuntimeError(f"the temp folder path is too long for espeak-ng too: {short}")
    if not os.path.exists(os.path.join(short, "phontab")):
        shutil.rmtree(short, ignore_errors=True)   # a broken leftover, e.g. after a temp cleaner
        partial = f"{short}.{os.getpid()}"
        shutil.copytree(path, partial, dirs_exist_ok=True)
        try:
            os.replace(partial, short)          # atomic, so a parallel process never sees half a copy
        except OSError:
            shutil.rmtree(partial, ignore_errors=True)   # another process finished first
    return short


class KokoroTTS:
    """
    Ultra-low latency streaming Text-to-Speech using the Kokoro-82M ONNX model.
    """
    def __init__(self, model_path: str = None, voices_path: str = None, default_voice: str = None):
        model_path = pick(model_path, "voice.tts.model_path", "models/kokoro-v1.0.onnx")
        voices_path = pick(voices_path, "voice.tts.voices_path", "models/voices-v1.0.bin")
        self.default_voice = pick(default_voice, "voice.tts.voice", "af_bella")
        self.model = None
        self.sample_rate = 24000  # Kokoro's native output rate
        
        if KOKORO_INSTALLED:
            print("[TTS] Loading Kokoro-82M ONNX model...")
            try:
                espeak_config = EspeakConfig(data_path=espeak_data_path())
            except Exception as e:
                # Skip Kokoro rather than let espeak-ng exit the whole process on a bad path
                print(f"[TTS] espeak data copy failed: {e}")
                return
            try:
                self.model = kokoro_onnx.Kokoro(model_path, voices_path, espeak_config=espeak_config)
            except Exception as e:
                print(f"[TTS] Failed to load Kokoro ONNX model: {e}")
                print("[TTS] Download the weights into ./models — see README.")
        else:
            print("[TTS] WARNING: kokoro-onnx not installed. Running in mock mode.")

    def synthesize_array(self, text: str, voice: str = None):
        """Synthesizes speech and returns (sample_rate, int16 array)."""
        if not text.strip():
            return self.sample_rate, np.zeros(0, dtype=np.int16)

        if not self.model:
            return self.sample_rate, np.zeros(self.sample_rate // 2, dtype=np.int16)

        samples, sample_rate = self.model.create(
            text,
            voice=voice or self.default_voice,
            speed=1.0,
            lang="en-us"
        )
        self.sample_rate = sample_rate
        return sample_rate, float32_to_int16(samples)
