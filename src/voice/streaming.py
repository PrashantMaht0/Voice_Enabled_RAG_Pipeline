"""Speak an answer sentence by sentence (spec 0006).

One worker thread synthesizes sentences in the order they arrive, so sentence N+1 is being synthesized
while sentence N plays. The dashboard and the latency benchmark both use it, so they measure the same thing.
"""
import io
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from src.utils.text_sanitizer import sanitize_for_speech

_DONE = object()
SPLIT_OVER_WORDS = 12        # an approved sentence longer than this is spoken in clause sized pieces
MIN_PIECE_WORDS = 5          # so the first piece plays after about a second of synthesis, not three


def speech_pieces(spoken: str) -> List[str]:
    """A long sentence cut at commas, semicolons and colons into pieces of at least MIN_PIECE_WORDS words.

    Only how an approved sentence is fed to the voice changes; nothing unchecked is ever spoken."""
    if len(spoken.split()) <= SPLIT_OVER_WORDS:
        return [spoken]
    parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", spoken) if p.strip()]
    pieces: List[str] = []
    for part in parts:
        if pieces and len(pieces[-1].split()) < MIN_PIECE_WORDS:
            pieces[-1] += " " + part
        else:
            pieces.append(part)
    if len(pieces) > 1 and len(pieces[-1].split()) < MIN_PIECE_WORDS:
        pieces[-2] += " " + pieces.pop()
    return pieces


@dataclass
class SpokenSentence:
    """One synthesized speech piece, in the order it is played."""
    index: int
    text: str                 # what was spoken, after sanitize_for_speech
    sample_rate: int
    samples: np.ndarray
    ready_at: float           # time.perf_counter() when the audio was ready to play (synthesized)

    @property
    def duration_s(self) -> float:
        """How long the piece plays, in seconds."""
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    def wav_bytes(self) -> bytes:
        """The sentence as a WAV file, the chunk format Gradio's streaming audio converts and plays."""
        samples = np.asarray(self.samples)
        if samples.dtype != np.int16:
            samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(samples.tobytes())
        return out.getvalue()


class SentenceSpeaker:
    """Feed it approved sentences as they arrive; read finished audio in order as it becomes ready."""
    def __init__(self, tts):
        self.tts = tts
        self._todo: "queue.Queue" = queue.Queue()
        self._ready: "queue.Queue" = queue.Queue()
        self._count = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        # Kokoro is used from this one thread only (it changes its own sample rate attribute)
        while True:
            item = self._todo.get()
            if item is _DONE:
                self._ready.put(_DONE)
                return
            index, text = item
            try:
                rate, samples = self.tts.synthesize_array(text)
            except Exception as e:
                print(f"[VOICE] Could not synthesize sentence {index}: {e}")
                rate, samples = 24000, np.zeros(0, dtype=np.int16)
            self._ready.put(SpokenSentence(index, text, rate, np.asarray(samples), time.perf_counter()))

    def say(self, sentence: str) -> bool:
        """Queues one approved sentence (as one or more speech pieces); False when nothing is speakable."""
        spoken = sanitize_for_speech(sentence)
        if not spoken:
            return False
        for piece in speech_pieces(spoken):
            self._todo.put((self._count, piece))
            self._count += 1
        return True

    def close(self):
        """No more sentences: the worker finishes the queued ones, then ready(wait=True) ends."""
        self._todo.put(_DONE)

    def ready(self, wait: bool = False, timeout: Optional[float] = None) -> Iterator[SpokenSentence]:
        """Sentences finished so far (without waiting), or all remaining ones (wait=True) until close()."""
        while True:
            try:
                item = self._ready.get(block=wait, timeout=timeout)
            except queue.Empty:
                return
            if item is _DONE:
                self._ready.put(_DONE)            # later calls also stop at the end
                return
            if len(item.samples):
                yield item


def max_gap_ms(first_ready: float, sentences: List[SpokenSentence]) -> float:
    """Largest max(0, ready_i - (first audio + durations of the sentences before i)) over i >= 2 (spec 0006);
    0 for a one sentence answer."""
    gap, played = 0.0, 0.0
    for i, s in enumerate(sentences):
        if i:
            gap = max(gap, s.ready_at - (first_ready + played))
        played += s.duration_s
    return round(gap * 1000, 1)
