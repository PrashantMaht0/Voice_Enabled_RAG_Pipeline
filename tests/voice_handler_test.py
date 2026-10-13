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

    def transcribe_detailed(self, audio_bytes):
        self.calls += 1
        return self.text, [0.9]


@pytest.fixture
def stt(monkeypatch):
    stub = _StubSTT()
    monkeypatch.setattr(ui, "get_stt", lambda: stub)
    return stub


def silent_audio(seconds=0.5, sample_rate=48000):
    return sample_rate, np.zeros(int(sample_rate * seconds), dtype=np.int16)


def test_voice_query_transcribes_and_returns_the_full_output_tuple(stt):
    updates = list(ui.voice_query(silent_audio(), []))

    assert updates and all(len(outputs) == 11 for outputs in updates)
    assert stt.calls == 1
    assert updates[-1][4] == ui.NOTHING_HEARD, "Silence is never sent to the API"


def test_stereo_float_audio_is_accepted(stt):
    rate, mono = silent_audio()
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)

    updates = list(ui.voice_query((rate, stereo), []))

    assert all(len(outputs) == 11 for outputs in updates) and stt.calls == 1


def test_no_audio_returns_the_full_output_tuple():
    [outputs] = list(ui.voice_query(None, []))

    assert len(outputs) == 11
    assert outputs[4] == "_No audio captured._"


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


# --- /test pass 2026-10-11: the silence rule and the Ask tab helpers (spec 0006) ---

# covers: AC-5
@pytest.mark.parametrize("transcript, probs, expected", [
    ("You", [0.1], True), ("Thank you.", [0.1], True), ("thanks for watching!", [0.1], True), ("", [], True),
    ("What is DNS?", [0.9, 0.95], True), ("What is DNS?", [0.9, 0.2], False), ("What is DNS?", [], False),
    ("Define RBAC", [0.3], False),
])
def test_heard_nothing_for_silence_or_filler_only(transcript, probs, expected):
    assert ui.heard_nothing(transcript, probs) is expected


def test_a_silent_recording_never_reaches_the_api(stt, monkeypatch):
    stt.text = "you"
    monkeypatch.setattr(ui, "stream_events", lambda *a, **k: pytest.fail("the API must not be called"))
    outputs = list(ui.voice_query(silent_audio(), []))
    assert ui.NOTHING_HEARD in outputs[-1]


TAG = "[Source: guide.pdf | Page: 4]"


# covers: AC-15
def test_citations_are_labelled_with_their_source_kinds():
    answer = f"A value. {TAG} Another. [Source: other.pdf | Page: 9]"
    citations = [{"source": "guide.pdf", "page": 4, "kinds": ["text", "figure"]}]
    labelled = ui.label_citations(answer, citations)
    assert f"{TAG} (text, figure, generated)" in labelled
    assert labelled.endswith("[Source: other.pdf | Page: 9]"), "an unknown source is left as it was"


def test_sources_are_headed_by_file_page_and_kind():
    chunks = [f"{TAG}\nRow one\nRow two"]
    text = ui.format_sources(chunks, [{"kind": "figure"}])
    assert text.startswith("**[1] guide.pdf · p.4 · Figure (generated)**")
    assert "Row one  \nRow two" in text, "each line stays on its own line"


# covers: AC-3, AC-4
@pytest.mark.parametrize("grounded, dropped, card, note", [
    (True, [], "PASS", False), (False, ["x", "y"], "FAIL (2 removed)", True),
    (None, [], "Unchecked", False), (False, ["x"], "FAIL (1 removed)", True),
])
def test_grounded_card_and_removed_note(grounded, dropped, card, note):
    done = {"answer": "An answer.", "citations": [], "grounded": grounded, "dropped_sentences": dropped,
            "total_ms": 1200.0}
    outputs = ui.answer_outputs([], {"confidence_score": 0.9, "retrieved_context": [f"{TAG}\nbody"]}, done, [])
    sources, grounded_card = outputs[4], outputs[6]
    assert card in grounded_card
    assert ("removed: not supported by your documents" in sources) is note
    if note:
        assert sources.startswith(f"_{len(dropped)} sentence{'s' if len(dropped) > 1 else ''} removed")


def test_cleared_and_answered_cards_use_the_same_labels():
    done = {"answer": "a", "citations": [], "grounded": True, "dropped_sentences": [], "total_ms": 1.0}
    answered = ui.answer_outputs([], {"confidence_score": 0.5}, done, [])[5:9]
    labels = lambda cards: [c.split("metric-label'>")[1].split("<")[0] for c in cards]
    assert labels(answered) == labels(ui.blank_metrics())


# covers: AC-15 (controls audit)
def test_delete_document_is_usable_only_once_a_document_is_chosen():
    assert ui.picker_changed(None)["interactive"] is False
    assert ui.picker_changed("guide.pdf")["interactive"] is True


def test_enter_asks_because_the_question_box_is_one_line():
    box = next(c for c in ui.demo.blocks.values()
               if getattr(c, "placeholder", None) == "What is an Azure App Service Web App?")
    assert box.lines == 1 and box.max_lines >= 2


def test_the_empty_spoken_answer_player_says_so():
    player = next(c for c in ui.demo.blocks.values() if getattr(c, "elem_id", None) == "voice-reply")
    assert player.streaming is True
    assert '#voice-reply .empty::after { content: "No spoken answer yet."' in ui.custom_css
