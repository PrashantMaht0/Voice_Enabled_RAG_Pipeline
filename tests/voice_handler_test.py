"""Exercises the Gradio voice handler for every configured language."""
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
def stub_models(monkeypatch):
    stt = _StubSTT()
    seen = {}

    def fake_get_stt(language=ui.DEFAULT_LANGUAGE):
        seen["stt_language"] = language
        return stt

    monkeypatch.setattr(ui, "get_stt", fake_get_stt)
    return stt, seen


def silent_audio(seconds=0.5, sample_rate=48000):
    return sample_rate, np.zeros(int(sample_rate * seconds), dtype=np.int16)


def test_voice_query_runs_for_every_configured_language(stub_models):
    """Guards the NameError: each label must resolve without blowing up."""
    stt, seen = stub_models

    for label in ui.LANGUAGE_LABELS:
        outputs = voice = ui.voice_query(silent_audio(), [], label)
        assert len(outputs) == 10
        assert seen["stt_language"] == ui.LANGUAGE_LABELS[label]

    assert stt.calls == len(ui.LANGUAGE_LABELS)


def test_voice_query_defaults_to_the_configured_language(stub_models):
    stt, seen = stub_models

    ui.voice_query(silent_audio(), [], None)

    assert seen["stt_language"] == ui.DEFAULT_LANGUAGE


def test_no_audio_returns_the_full_output_tuple():
    outputs = ui.voice_query(None, [], "English")

    assert len(outputs) == 10
    assert outputs[-1] is None


def test_language_key_falls_back_for_unknown_labels():
    assert ui.language_key("Klingon") == ui.DEFAULT_LANGUAGE
    assert ui.language_key("English") == "english"
