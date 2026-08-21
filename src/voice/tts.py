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
        """
        Same synthesis as synthesize(), but returns (sample_rate, int16 array).
        WebRTC transports need the sample rate alongside the samples.
        """
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

    def synthesize(self, text: str, voice: str = None) -> bytes:
        """
        Synthesizes a single text string (like a completed sentence) into 
        raw 16-bit PCM bytes for WebRTC transmission.
        """
        if not text.strip():
            return b""
            
        if not self.model:
            # Mock bypass: Generate 0.5 seconds of silent audio bytes
            mock_audio = np.zeros(16000 // 2, dtype=np.int16)
            return mock_audio.tobytes()

        target_voice = voice if voice else self.default_voice

        try:
            # Create audio using the Kokoro model (returns a float32 numpy array and sample rate)
            samples, sample_rate = self.model.create(
                text, 
                voice=target_voice, 
                speed=1.0, 
                lang="en-us"
            )
            
            # Convert float32 array to 16-bit PCM integer for WebRTC compatibility
            pcm_16_audio = float32_to_int16(samples)
            return pcm_16_audio.tobytes()
            
        except Exception as e:
            print(f"[TTS] Synthesis failed: {e}")
            return b""