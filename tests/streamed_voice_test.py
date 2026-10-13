"""The streamed answer path and the sentence speaker (spec 0006, AC-7, AC-8, AC-24)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.api import routes
from src.verification import grounding
from src.voice.streaming import SentenceSpeaker

TAG = "[Source: a.pdf | Page: 1]"
THREE = f"The warranty lasts two years. {TAG} It covers parts and labour. {TAG} Labour is included too. {TAG}"
CHUNK = f"{TAG}\nThe warranty lasts two years and covers parts and labour."
DETAILS = [{"source": "a.pdf", "page": 1, "kind": "text", "score": 0.9, "chunk_id": "x_0"}]


class FakeChecker:
    """Rejects any sentence containing one of the given words."""
    name = "judge:fake"

    def __init__(self, reject=()):
        self.reject = reject

    def supported(self, sentence, chunks):
        return not any(word in sentence for word in self.reject)


@pytest.fixture
def stream(monkeypatch):
    """Runs answer_events on a fixed generation; returns its events in a short readable form."""
    def run(text, reject=(), fail_after=None):
        def tokens(query, chunks):
            for i, word in enumerate(text.split(" ")):
                if fail_after is not None and i == fail_after:
                    raise RuntimeError("model went away")
                yield word + " "
        monkeypatch.setattr(routes, "retrieve", lambda q, tracer: ([CHUNK], 0.9, DETAILS))
        monkeypatch.setattr(routes, "generate_tokens", tokens)
        monkeypatch.setattr(routes, "get_retriever", lambda: SimpleNamespace(confidence_threshold=0.65))
        monkeypatch.setattr(grounding, "effective_checker", lambda: FakeChecker(reject))
        events = list(routes.answer_events(routes.QueryRequest(query="What is the warranty period?")))
        return [short(e) for e in events if e["type"] != "meta"], events
    return run


def short(event):
    if event["type"] == "sentence":
        return f"s{event['index']}"
    return event["type"]


# --- the stream (AC-7, AC-24) ---

def test_approved_sentences_stream_in_order_then_done(stream):
    order, events = stream(THREE)
    assert order == ["s0", "s1", "s2", "done"]
    assert events[1]["text"] == "The warranty lasts two years.", "shown text, without its tag"
    assert not any("draft" in e for e in events), "the first sentence draft was removed (AC-24)"


def test_a_rejected_middle_sentence_leaves_no_index_gap(stream):
    order, events = stream(THREE, reject=("covers",))
    assert order == ["s0", "s1", "done"]
    assert events[-1]["dropped_sentences"] == ["It covers parts and labour."]


def test_grounding_refusal_streams_one_refusal_sentence(stream):
    order, events = stream(THREE, reject=("warranty", "covers", "Labour"))
    assert order == ["s0", "done"]
    assert events[1]["text"] == "Insufficient context."
    assert events[-1]["path"] == "refused_grounding"


def test_an_error_after_an_approved_sentence_ends_with_error(stream):
    assert stream(THREE, fail_after=11)[0] == ["s0", "error"]


def test_no_draft_or_drop_event_types_remain(stream):
    for reject in ((), ("warranty",), ("warranty", "covers", "Labour")):
        _, events = stream(THREE, reject=reject)
        assert {e["type"] for e in events} <= {"meta", "sentence", "done"}


def test_query_request_has_no_drafts_field():
    assert "drafts" not in routes.QueryRequest.model_fields


# --- the speaker (AC-8, AC-24) ---

class FakeTTS:
    def __init__(self, delay=0.01):
        self.delay, self.calls = delay, []

    def synthesize_array(self, text):
        self.calls.append(text)
        time.sleep(self.delay)
        return 24000, np.ones(2400, dtype=np.int16)


def test_said_sentences_come_out_in_order_and_close_ends_the_wait():
    speaker = SentenceSpeaker(FakeTTS())
    for i in range(3):
        speaker.say(f"Sentence number {i} of the answer.")
    speaker.close()
    started = time.perf_counter()
    spoken = list(speaker.ready(wait=True, timeout=2))
    assert [s.text for s in spoken] == [f"Sentence number {i} of the answer." for i in range(3)]
    assert time.perf_counter() - started < 1.5
    assert all(a.ready_at <= b.ready_at for a, b in zip(spoken, spoken[1:]))


def test_unspeakable_sentence_is_skipped():
    speaker = SentenceSpeaker(FakeTTS())
    assert speaker.say(TAG) is False
    speaker.say("A later sentence.")
    speaker.close()
    assert [s.text for s in speaker.ready(wait=True, timeout=2)] == ["A later sentence."]


def test_speaker_has_no_draft_methods():
    for name in ("draft", "approve", "discard"):
        assert not hasattr(SentenceSpeaker, name)
