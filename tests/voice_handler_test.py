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


def test_espeak_gets_a_short_data_path_when_the_venv_is_deep(tmp_path, monkeypatch):
    """espeak-ng ignores a data folder whose real path is 160+ characters and exits the process."""
    from src.voice import tts

    monkeypatch.setattr(tts.tempfile, "gettempdir", lambda: str(tmp_path))
    near = tmp_path / "espeak-ng-data"
    near.mkdir()
    monkeypatch.setattr(tts.espeakng_loader, "get_data_path", lambda: str(near))
    assert tts.espeak_data_path() == str(near), "A short path is used as it is"

    deep = tmp_path / ("d" * (tts.ESPEAK_PATH_LIMIT - len(str(tmp_path)))) / "espeak-ng-data"
    deep.mkdir(parents=True)
    (deep / "phontab").write_bytes(b"data")
    monkeypatch.setattr(tts.espeakng_loader, "get_data_path", lambda: str(deep))
    stale = tmp_path / f"vrag-espeak-ng-data-{tts.importlib.metadata.version('espeakng-loader')}"
    stale.mkdir()                                 # a leftover a temp cleaner emptied

    short = tts.espeak_data_path()

    assert len(short) < tts.ESPEAK_PATH_LIMIT
    assert (tmp_path / short).joinpath("phontab").read_bytes() == b"data"
