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


BRENT_Q = "What is the forecast Brent crude oil spot price per barrel for 2027?"
BRENT_HISTORY = [
    {"role": "user", "content": BRENT_Q},
    {"role": "assistant", "content": "The forecast Brent crude oil spot price per barrel for 2027 is $84. "
                                     "[Source: eia-short-term-energy-outlook-2026-10-pages-1-20.pdf | Page: 3]"},
]
EARNINGS_Q = "What were the 2023 median earnings of all male workers?"


def test_a_standalone_question_after_an_answer_is_searched_as_asked(client):
    """The reported bug: llama3.2:1b rewrote a new standalone question into the previous one."""
    api, rewriter, retriever = client

    api.post("/api/query", json={"query": EARNINGS_Q, "history": BRENT_HISTORY})

    assert rewriter.seen_history is None, "A question that stands alone must not reach the rewriter."
    assert retriever.seen_query == EARNINGS_Q


class _EchoingOllama:
    """Replies the way llama3.2:1b did: the previous question with its citation tag."""
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, model, messages, options):
        self.calls.append(messages)
        return {"message": {"content": self.reply}}


def test_the_client_returns_a_standalone_question_unchanged_without_calling_the_model():
    from src.llm.llm_client import LocalLLMClient
    llm = LocalLLMClient(generation_model="gen", rewriter_model="rw")
    llm.client = _EchoingOllama(BRENT_Q + " | Page: 3 [Source: eia.pdf | Page: 3] | 84")

    assert llm.rewrite_query(EARNINGS_Q, BRENT_HISTORY) == EARNINGS_Q
    assert llm.client.calls == []


def test_the_client_still_resolves_a_follow_up_without_citation_tags():
    from src.llm.llm_client import LocalLLMClient
    llm = LocalLLMClient(generation_model="gen", rewriter_model="rw")
    llm.client = _EchoingOllama("Forecast Brent crude oil spot price per barrel for 2026 [Source: eia.pdf | Page: 3]")

    assert llm.rewrite_query("What about 2026?", BRENT_HISTORY) == "Forecast Brent crude oil spot price per barrel for 2026"
    assert "[Source:" not in llm.client.calls[0][0]["content"]
