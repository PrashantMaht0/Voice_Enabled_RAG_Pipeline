import numpy as np

try:
    import kokoro_onnx
    import soundfile as sf
    KOKORO_INSTALLED = True
except ImportError:
    KOKORO_INSTALLED = False

from src.utils.audio_utils import float32_to_int16
from src.utils.config import pick

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
                # Initialize the ONNX session
                self.model = kokoro_onnx.Kokoro(model_path, voices_path)
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
