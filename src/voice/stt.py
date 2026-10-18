"""Speech to text with faster-whisper, run locally.

Without the faster-whisper package or its weights the model is None (mock mode): every recording is heard as
silence, so the dashboard asks the user to try again and the latency benchmark skips voice."""
import numpy as np

try:
    from faster_whisper import WhisperModel
    WHISPER_INSTALLED = True
except ImportError:
    WHISPER_INSTALLED = False

from src.utils.audio_utils import int16_to_float32
from src.utils.config import pick


class WhisperSTT:
    """A faster-whisper model (`voice.stt.*` in config.yaml, English base model on CPU by default)."""

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
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            except Exception as e:
                print(f"[STT] Initialization failed: {e}")
        else:
            print("[STT] WARNING: faster-whisper not installed. Running in mock mode.")

    def transcribe_detailed(self, audio_bytes: bytes):
        """(transcript, [no speech probability per segment]) of 16 kHz, 16 bit mono PCM, for the silence rule.
        Mock mode hears nothing: ("", [])."""
        if not self.model:
            return "", []
        audio_array = int16_to_float32(np.frombuffer(audio_bytes, dtype=np.int16))
        segments, _ = self.model.transcribe(audio_array, beam_size=1, language=self.language,
                                            condition_on_previous_text=False)
        segments = list(segments)
        return "".join(s.text for s in segments).strip(), [float(s.no_speech_prob) for s in segments]

    def transcribe(self, audio_bytes: bytes) -> str:
        """The transcript of 16 kHz, 16 bit mono PCM; empty in mock mode."""
        return self.transcribe_detailed(audio_bytes)[0]
