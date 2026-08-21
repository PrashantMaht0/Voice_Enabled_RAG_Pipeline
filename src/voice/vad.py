import numpy as np

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

from src.utils.audio_utils import resample_audio, int16_to_float32
from src.utils.config import pick

class SileroVAD:
    """
    Detects continuous speech boundaries to automatically trigger processing 
    when the user finishes talking, utilizing the robust Silero VAD v5 model.
    """
    def __init__(self, threshold: float = None, silence_duration_ms: int = None, sample_rate: int = 16000):
        self.threshold = pick(threshold, "voice.vad.threshold", 0.5)
        self.silence_duration_ms = pick(silence_duration_ms, "voice.vad.min_silence_duration_ms", 400)
        self.sample_rate = sample_rate
        
        # Calculate how many consecutive silent chunks are needed to trigger an endpoint
        self.required_silent_samples = int((self.silence_duration_ms / 1000.0) * self.sample_rate)
        self.current_silent_samples = 0
        self.speech_started = False

        # Silero v5 scores exactly 512 samples at 16 kHz per call, so incoming
        # chunks are buffered and consumed one frame at a time.
        self.frame_size = 512 if sample_rate == 16000 else 256
        self._buffer = np.array([], dtype=np.float32)
        
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the Silero VAD model via PyTorch Hub."""
        if not TORCH_INSTALLED:
            print("[VAD] WARNING: PyTorch not installed. Running in mock mode.")
            return
            
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                # Without this torch.hub blocks on an interactive trust prompt and
                # the VAD silently degrades to mock mode (which ends every turn).
                trust_repo=True
            )
            self.get_speech_timestamps = utils[0]
        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")

    def is_speech_ended(self, audio_chunk: bytes) -> bool:
        """
        Ingests streaming PCM bytes. Returns True if speech was detected 
        but has now concluded (based on the trailing silence threshold).
        """
        if not self.model:
            # Mock behavior: Assume speech ends immediately upon receiving any chunk
            return True

        # Convert bytes to 16-bit PCM array, then to 32-bit float for the model
        pcm_data = np.frombuffer(audio_chunk, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, int16_to_float32(pcm_data)])

        ended = False
        while len(self._buffer) >= self.frame_size:
            frame = self._buffer[:self.frame_size]
            self._buffer = self._buffer[self.frame_size:]

            speech_prob = self.model(torch.from_numpy(frame), self.sample_rate).item()

            if speech_prob > self.threshold:
                self.speech_started = True
                self.current_silent_samples = 0
            elif self.speech_started:
                self.current_silent_samples += self.frame_size

            # Trigger endpoint if silence exceeds our configured window
            if self.speech_started and self.current_silent_samples >= self.required_silent_samples:
                self.reset()
                ended = True

        return ended

    def reset(self):
        """Clears turn state so the next utterance starts from silence."""
        self.speech_started = False
        self.current_silent_samples = 0
        self._buffer = np.array([], dtype=np.float32)