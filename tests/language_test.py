import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes
from src.utils import config as cfg


class _Retriever:
    confidence_threshold = 0.65

    def __init__(self):
        self.seen_query = None

    def search_and_rerank(self, query, tracer=None):
        self.seen_query = query
        return ["[Source: a.pdf | Chunk: 1]\ncontent"], 0.99


class _LLM:
    generation_model = "stub"

    def __init__(self):
        self.normalized = []

    def normalize_query(self, query):
        self.normalized.append(query)
        return "What is Blob Storage?"

    def generate(self, query, chunks, tracer=None):
        return "Blob storage stores files."


class _Guard:
    def strip_ungrounded_citations(self, answer, chunks):
        return answer, []

    def verify_assertions(self, answer, chunks):
        return True


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    retriever, llm = _Retriever(), _LLM()
    monkeypatch.setattr(routes, "_retriever", retriever)
    monkeypatch.setattr(routes, "_llm", llm)
    monkeypatch.setattr(routes, "_guard", _Guard())
    monkeypatch.setattr(routes, "_rewriter", None)
    return TestClient(app), retriever, llm


def test_hinglish_query_is_translated_before_retrieval(client):
    """Retrieval is English-only, so a Hinglish question must be normalised first."""
    api, retriever, llm = client

    body = api.post("/api/query", json={
        "query": "Blob storage kya hota hai?",
        "language": "hinglish",
    }).json()

    assert llm.normalized == ["Blob storage kya hota hai?"]
    assert retriever.seen_query == "What is Blob Storage?"
    assert body["search_query"] == "What is Blob Storage?"


def test_english_query_is_not_translated(client):
    api, retriever, llm = client

    api.post("/api/query", json={"query": "What is Blob Storage?", "language": "english"})

    assert llm.normalized == [], "English queries must not go through the translator."
    assert retriever.seen_query == "What is Blob Storage?"


def test_language_defaults_to_english(client):
    api, retriever, llm = client

    api.post("/api/query", json={"query": "What is Blob Storage?"})

    assert llm.normalized == []


def test_config_declares_both_language_modes():
    langs = cfg.get("voice.languages", {})

    assert set(langs) == {"english", "hinglish"}
    assert langs["hinglish"]["normalize_query"] is True
    assert langs["english"]["normalize_query"] is False

    # Hinglish transcribes with its own STT but speaks through Kokoro: the F5 voice
    # is not intelligible, so it stays opt-in. See README.
    assert langs["hinglish"]["stt_model"].endswith("zero-stt-hinglish-ct2")
    assert langs["hinglish"]["tts_backend"] == "kokoro"


# --- Hinglish translation parsing ---

from src.llm.llm_client import LocalLLMClient


def test_hindi_function_words_are_not_treated_as_products():
    """
    'matlab' means 'meaning'. The first prompt told the model to preserve product
    names, so it kept MATLAB and the query became "...and what is Matlab?".
    """
    from src.prompts.prompt_template import NORMALIZE_QUERY_PROMPT

    prompt = NORMALIZE_QUERY_PROMPT.format(query="Azure mai Resources ka kya matlab hota hai")

    assert "matlab=meaning" in prompt
    assert "not product names" in prompt


def test_rambling_replies_do_not_reach_the_index():
    """A small model that starts explaining itself must fall back, not poison retrieval."""
    extract = LocalLLMClient._first_usable_line

    assert extract("What is the meaning of Resources in Azure?", "FB") == \
        "What is the meaning of Resources in Azure?"
    assert extract("Here is the rewritten query:\n\nWhat is a resource group?", "FB") == \
        "What is a resource group?"
    assert extract("English: What does ERR_SSL mean?", "FB") == "What does ERR_SSL mean?"
    assert extract("Here are examples:\n1. What is X?\n2. What is Y?", "FB") == "What is X?"


def test_empty_or_useless_reply_falls_back_to_the_original():
    extract = LocalLLMClient._first_usable_line

    assert extract("", "original query") == "original query"
    assert extract("   \n  \n", "original query") == "original query"
    assert extract("x" * 500, "original query") == "original query"
