"""Exercises the Gradio voice handler end to end with stubbed models."""
import numpy as np
import pytest

import app as ui


class _StubSTT:
    def __init__(self, text=""):
        self.text = text
        self.calls = 0

    def transcribe(self, audio_bytes):
        self.calls += 1
        return self.text


@pytest.fixture
def stt(monkeypatch):
    stub = _StubSTT()
    monkeypatch.setattr(ui, "get_stt", lambda: stub)
    return stub


def silent_audio(seconds=0.5, sample_rate=48000):
    return sample_rate, np.zeros(int(sample_rate * seconds), dtype=np.int16)


def test_voice_query_transcribes_and_returns_the_full_output_tuple(stt):
    outputs = ui.voice_query(silent_audio(), [])

    assert len(outputs) == 10
    assert stt.calls == 1


def test_stereo_float_audio_is_accepted(stt):
    rate, mono = silent_audio()
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)

    outputs = ui.voice_query((rate, stereo), [])

    assert len(outputs) == 10 and stt.calls == 1


def test_no_audio_returns_the_full_output_tuple():
    outputs = ui.voice_query(None, [])

    assert len(outputs) == 10
    assert outputs[-1] is None
