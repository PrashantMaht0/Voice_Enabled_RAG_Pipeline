"""Experimental Hinglish speech synthesis with F5-TTS."""
import os

import numpy as np

try:
    from f5_tts.api import F5TTS
    F5_INSTALLED = True
except ImportError:
    F5_INSTALLED = False

from src.utils.audio_utils import float32_to_int16
from src.utils.config import pick


class F5HinglishTTS:
    """Mirrors KokoroTTS.synthesize_array so the two backends are interchangeable."""

    def __init__(self, ckpt_file: str = None, vocab_file: str = None,
                 ref_audio: str = None, ref_text_file: str = None, nfe_step: int = None):
        self.ckpt_file = pick(ckpt_file, "voice.languages.hinglish.f5.ckpt_file",
                              "models/F5-TTS-Hinglish/model_last.pt")
        self.vocab_file = pick(vocab_file, "voice.languages.hinglish.f5.vocab_file",
                               "models/F5-TTS-Hinglish/vocab.txt")
        self.ref_audio = pick(ref_audio, "voice.languages.hinglish.f5.ref_audio",
                              "models/hinglish_ref.wav")
        ref_text_path = pick(ref_text_file, "voice.languages.hinglish.f5.ref_text_file",
                             "models/hinglish_ref.txt")

        self.ref_text = ""
        if os.path.exists(ref_text_path):
            with open(ref_text_path, encoding="utf-8") as f:
                self.ref_text = f.read().strip()

        # Flow-matching steps: fewer is faster but rougher.
        self.nfe_step = pick(nfe_step, "voice.languages.hinglish.f5.nfe_step", 16)
        self.sample_rate = 24000        # F5-Hindi-24KHz
        self.model = None

        missing = [p for p in (self.ckpt_file, self.vocab_file, self.ref_audio) if not os.path.exists(p)]
        if not F5_INSTALLED:
            print("[TTS-F5] WARNING: f5-tts not installed. Hinglish speech runs in mock mode.")
        elif missing:
            print(f"[TTS-F5] WARNING: missing files {missing}. Hinglish speech runs in mock mode.")
        else:
            print("[TTS-F5] Loading F5-TTS Hinglish...")
            try:
                # The checkpoint is F5-TTS Small; the default architecture will not load it.
                self.model = F5TTS(model="F5TTS_Small", ckpt_file=self.ckpt_file,
                                   vocab_file=self.vocab_file, device="cpu")
            except Exception as e:
                print(f"[TTS-F5] Failed to load: {e}")

    def synthesize_array(self, text: str, voice: str = None):
        """Returns (sample_rate, int16 array), matching the Kokoro backend."""
        if not text.strip():
            return self.sample_rate, np.zeros(0, dtype=np.int16)

        if not self.model:
            return self.sample_rate, np.zeros(self.sample_rate // 2, dtype=np.int16)

        try:
            wav, sample_rate, _ = self.model.infer(
                ref_file=self.ref_audio,
                ref_text=self.ref_text,
                gen_text=text,
                nfe_step=self.nfe_step,
                show_info=lambda *a, **k: None,
            )
            self.sample_rate = sample_rate
            return sample_rate, float32_to_int16(np.asarray(wav, dtype=np.float32))
        except Exception as e:
            print(f"[TTS-F5] Synthesis failed: {e}")
            return self.sample_rate, np.zeros(0, dtype=np.int16)
