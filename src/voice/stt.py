import numpy as np

try:
    from faster_whisper import WhisperModel
    WHISPER_INSTALLED = True
except ImportError:
    WHISPER_INSTALLED = False

from src.utils.audio_utils import int16_to_float32
from src.utils.config import pick

class WhisperSTT:
    """
    High-speed local Speech-to-Text using Faster-Whisper.
    """
    def __init__(self, model_size: str = None, device: str = None, compute_type: str = None,
                 language: str = None):
        self.model_size = pick(model_size, "voice.stt.model", "base.en")
        self.device = pick(device, "voice.stt.device", "cpu")
        self.compute_type = pick(compute_type, "voice.stt.compute_type", "int8")
        self.language = pick(language, "voice.stt.language", "en")
        self.model = None
        
        if WHISPER_INSTALLED:
            print(f"[STT] Loading Faster-Whisper ({self.model_size}) on {self.device}...")
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
            except Exception as e:
                print(f"[STT] Initialization failed: {e}")
        else:
            print("[STT] WARNING: faster-whisper not installed. Running in mock mode.")

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes an accumulated buffer of raw audio bytes into text.
        """
        if not self.model:
            # Mock bypass
            return "How do I configure SSL on the router?"

        # Convert raw bytes to 32-bit float array required by Whisper
        pcm_data = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = int16_to_float32(pcm_data)

        # Execute transcription
        segments, info = self.model.transcribe(
            audio_array, 
            beam_size=1,
            language=self.language,
            condition_on_previous_text=False
        )
        
        # Combine transcribed segments
        transcript = "".join([segment.text for segment in segments])
        return transcript.strip()