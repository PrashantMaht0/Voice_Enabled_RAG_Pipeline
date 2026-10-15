"""A refused turn must not poison the following question."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes
from src.utils.text_sanitizer import is_refusal, usable_history

REFUSED_Q = "What is the price of a Motorola phone?"
ENGLISH_Q = "Which actions are allowed for a contributor role?"
REFUSAL = "Insufficient context."


class _Rewriter:
    """Mimics the observed behaviour: it echoes the topic of the last history turn."""
    def __init__(self):
        self.seen_history = None

    def resolve_query(self, query, history, tracer=None):
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

    def generate(self, query, chunks, tracer=None):
        return "A Contributor can manage resources of all types."


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    rewriter, retriever = _Rewriter(), _Retriever()
    monkeypatch.setattr(routes, "_rewriter", rewriter)
    monkeypatch.setattr(routes, "_retriever", retriever)
    monkeypatch.setattr(routes, "_llm", _LLM())
    return TestClient(app), rewriter, retriever


REFUSED_HISTORY = [
    {"role": "user", "content": REFUSED_Q},
    {"role": "assistant", "content": REFUSAL},
]


def test_refused_turn_is_not_used_as_context(client):
    """The reported bug: a refused turn hijacked the next question."""
    api, rewriter, retriever = client

    api.post("/api/query", json={"query": ENGLISH_Q, "history": REFUSED_HISTORY})

    # Nothing usable survives, so the rewriter is skipped and the question stands as asked
    assert rewriter.seen_history is None, "A refused turn must not reach the rewriter."
    assert retriever.seen_query == ENGLISH_Q


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
    assert usable_history(REFUSED_HISTORY) == []
    good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a real answer"}]
    assert usable_history(good) == good
