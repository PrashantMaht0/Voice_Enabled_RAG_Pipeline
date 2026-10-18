"""Model call timeouts: the answer client always has a timeout, a timed out answer raises
instead of becoming answer text, and a timed out rewrite falls back to the question as asked."""
from types import SimpleNamespace

import httpx
import pytest

from src.llm import llm_client
from src.llm.llm_client import LocalLLMClient


@pytest.fixture
def made(monkeypatch):
    """Records how each ollama.Client is built, without opening a connection."""
    clients = []

    class Client:
        def __init__(self, host=None, timeout=None, **kwargs):
            clients.append({"host": host, "timeout": timeout})

    monkeypatch.setattr(llm_client.ollama, "Client", Client)
    return clients


def timed_out(**kwargs):
    raise httpx.ReadTimeout("no chunk within 120 s")


def test_the_client_always_has_the_configured_timeout_even_without_a_host(made):
    LocalLLMClient(generation_model="gen", rewriter_model="rw")

    assert made == [{"host": None, "timeout": 120.0}]


def test_a_given_host_keeps_the_timeout(made):
    LocalLLMClient(generation_model="gen", rewriter_model="rw", host="http://ollama:11434")

    assert made == [{"host": "http://ollama:11434", "timeout": 120.0}]


def test_a_timed_out_answer_raises_instead_of_becoming_the_answer():
    """Earlier, the error text was yielded as answer tokens, and could be cached."""
    llm = LocalLLMClient(generation_model="gen", rewriter_model="rw")
    llm.client = SimpleNamespace(chat=timed_out)

    with pytest.raises(httpx.ReadTimeout):
        llm.generate("what port?", ["[Source: net.pdf | Chunk: 0]\nOpen port 443."])


def test_a_stream_that_stops_mid_answer_raises_after_the_tokens_it_gave():
    """The timeout is per chunk, so a stall partway through still ends the question."""
    def stalls(**kwargs):
        yield {"message": {"content": "Port "}}
        raise httpx.ReadTimeout("stalled")

    llm = LocalLLMClient(generation_model="gen", rewriter_model="rw")
    llm.client = SimpleNamespace(chat=stalls)
    tokens = []

    with pytest.raises(httpx.ReadTimeout):
        for token in llm.stream_generate("what port?", ["chunk"]):
            tokens.append(token)

    assert tokens == ["Port "]


def test_a_timed_out_rewrite_falls_back_to_the_question_as_asked():
    llm = LocalLLMClient(generation_model="gen", rewriter_model="rw")
    llm.client = SimpleNamespace(chat=timed_out)

    rewritten = llm.rewrite_query("And caching?", [{"role": "user", "content": "What is DNS?"}])

    assert rewritten == "And caching?"
