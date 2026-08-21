"""
Reproduces the reported failure: a Hinglish turn that was refused poisoned the
following English question, so a query the system answers fine on its own came
back as "Insufficient context".
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes
from src.utils.text_sanitizer import has_non_latin, is_refusal, usable_history

HINGLISH_Q = "contributor role के लिए कौन से actions allowed हैं"
ENGLISH_Q = "Which actions are allowed for a contributor role?"
REFUSAL = "Insufficient context."


class _Rewriter:
    """Mimics the observed behaviour: it echoes the topic of the last history turn."""
    def __init__(self):
        self.seen_history = None

    def resolve_query(self, query, history):
        self.seen_history = history
        return history[0]["content"] if history else query


class _Retriever:
    confidence_threshold = 0.65

    def __init__(self):
        self.seen_query = None

    def search_and_rerank(self, query, tracer=None):
        self.seen_query = query
        return ["[Source: a.pdf | Chunk: 1]\ncontent"], 0.99


class _LLM:
    generation_model = "stub"

    def normalize_query(self, query):
        return "Which actions are allowed for a contributor role?"

    def generate(self, query, chunks, tracer=None):
        return "A Contributor can manage resources of all types."


class _Guard:
    def strip_ungrounded_citations(self, answer, chunks):
        return answer, []

    def verify_assertions(self, answer, chunks):
        return True


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    rewriter, retriever = _Rewriter(), _Retriever()
    monkeypatch.setattr(routes, "_rewriter", rewriter)
    monkeypatch.setattr(routes, "_retriever", retriever)
    monkeypatch.setattr(routes, "_llm", _LLM())
    monkeypatch.setattr(routes, "_guard", _Guard())
    return TestClient(app), rewriter, retriever


REFUSED_HISTORY = [
    {"role": "user", "content": HINGLISH_Q},
    {"role": "assistant", "content": REFUSAL},
]


def test_refused_turn_is_not_used_as_context(client):
    """The reported bug: the refused Hinglish turn hijacked the next question."""
    api, rewriter, retriever = client

    api.post("/api/query", json={"query": ENGLISH_Q, "history": REFUSED_HISTORY})

    # Nothing usable survives, so the rewriter is skipped and the question stands as asked
    assert rewriter.seen_history is None, "A refused turn must not reach the rewriter."
    assert retriever.seen_query == ENGLISH_Q
    assert not has_non_latin(retriever.seen_query)


def test_non_latin_survivor_is_normalised_even_in_english_mode(client):
    """If Devanagari still leaks through the rewrite, it must be translated anyway."""
    api, rewriter, retriever = client

    api.post("/api/query", json={
        "query": ENGLISH_Q,
        "language": "english",
        "history": [
            {"role": "user", "content": HINGLISH_Q},
            {"role": "assistant", "content": "Contributor manages resources."},
        ],
    })

    assert not has_non_latin(retriever.seen_query), \
        f"A non-English query reached the English-only index: {retriever.seen_query!r}"


def test_successful_turns_are_still_used_as_context(client):
    api, rewriter, retriever = client

    api.post("/api/query", json={
        "query": "What about Reader?",
        "history": [
            {"role": "user", "content": "What is the Contributor role?"},
            {"role": "assistant", "content": "Contributor manages resources."},
        ],
    })

    assert rewriter.seen_history, "Useful history must still reach the rewriter."


def test_history_helpers():
    assert is_refusal(REFUSAL)
    assert has_non_latin(HINGLISH_Q)
    assert usable_history(REFUSED_HISTORY) == []
    good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a real answer"}]
    assert usable_history(good) == good
