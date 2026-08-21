import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes


class _FakeRetriever:
    confidence_threshold = 0.65

    def __init__(self, chunks, score):
        self.chunks = chunks
        self.score = score

    def search_and_rerank(self, query, tracer=None):
        if tracer:
            tracer.add_static_span("bm25", "stub")
        return self.chunks, self.score


class _FakeLLM:
    generation_model = "stub-model"

    def __init__(self):
        self.calls = 0

    def generate(self, query, chunks, tracer=None):
        self.calls += 1
        return "Port 443 must be open. [Source: net.pdf | Chunk: 0]"


class _FakeRewriter:
    def __init__(self):
        self.seen_history = None

    def resolve_query(self, query, history):
        self.seen_history = history
        return "Scaling Azure App Service Web App"


class _FakeGuard:
    def strip_ungrounded_citations(self, answer, chunks):
        return answer, []

    def verify_assertions(self, answer, chunks):
        return True


@pytest.fixture
def client(monkeypatch):
    """Bare app around the router so no lifespan warmup loads real models."""
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def install(monkeypatch, retriever, llm, rewriter=None):
    monkeypatch.setattr(routes, "_retriever", retriever)
    monkeypatch.setattr(routes, "_llm", llm)
    monkeypatch.setattr(routes, "_guard", _FakeGuard())
    monkeypatch.setattr(routes, "_rewriter", rewriter or _FakeRewriter())


def test_health_returns_200(client, monkeypatch):
    install(monkeypatch, _FakeRetriever([], 0.0), _FakeLLM())

    body = client.get("/api/health").json()

    assert body["status"] == "healthy"
    assert "ollama_status" in body


def test_query_returns_the_contract_the_dashboard_consumes(client, monkeypatch):
    llm = _FakeLLM()
    install(monkeypatch, _FakeRetriever(["chunk text"], 0.99), llm)

    body = client.post("/api/query", json={"query": "what port?"}).json()

    assert body["answer"].startswith("Port 443")
    assert body["confidence_score"] == pytest.approx(0.99)
    assert body["retrieved_context"] == ["chunk text"]
    assert body["grounded"] is True
    assert body["dropped_citations"] == []
    assert body["total_latency_ms"] > 0
    assert {span["layer"] for span in body["trace"]} >= {"bm25", "confidence", "verification"}
    assert llm.calls == 1


def test_low_confidence_skips_the_llm_entirely(client, monkeypatch):
    llm = _FakeLLM()
    install(monkeypatch, _FakeRetriever([], 0.42), llm)

    body = client.post("/api/query", json={"query": "who won the world cup?"}).json()

    assert body["answer"] == "Insufficient context."
    assert body["retrieved_context"] == []
    assert llm.calls == 0, "The LLM must never run once the confidence gate fails."


def test_follow_up_is_rewritten_into_a_standalone_query(client, monkeypatch):
    """A pronoun-only follow-up must be resolved before it reaches retrieval."""
    rewriter = _FakeRewriter()
    install(monkeypatch, _FakeRetriever(["chunk text"], 0.99), _FakeLLM(), rewriter)

    body = client.post("/api/query", json={
        "query": "How do I scale it?",
        "history": [
            {"role": "user", "content": "What is an Azure App Service Web App?"},
            {"role": "assistant", "content": "A Web App is hosted in an App Service."},
        ],
    }).json()

    assert body["search_query"] == "Scaling Azure App Service Web App"
    assert rewriter.seen_history[0]["content"].startswith("What is an Azure")


def test_first_turn_skips_the_rewriter(client, monkeypatch):
    """With no history there is nothing to resolve, so the query is used verbatim."""
    rewriter = _FakeRewriter()
    install(monkeypatch, _FakeRetriever(["chunk text"], 0.99), _FakeLLM(), rewriter)

    body = client.post("/api/query", json={"query": "What is a resource group?"}).json()

    assert body["search_query"] == "What is a resource group?"
    assert rewriter.seen_history is None, "The rewriter must not run on the first turn."


# --- corpus deletion ---

@pytest.fixture
def corpus_dir(tmp_path, monkeypatch):
    """Isolated data/raw with ingestion stubbed out."""
    monkeypatch.setattr(routes, "RAW_DIR", str(tmp_path))
    rebuilds = []
    monkeypatch.setattr(routes, "trigger_ingestion_pipeline", lambda: rebuilds.append(1))
    (tmp_path / "manual.md").write_text("content")
    return tmp_path, rebuilds


def test_delete_removes_the_file_and_rebuilds(client, corpus_dir):
    tmp_path, rebuilds = corpus_dir

    body = client.delete("/api/corpus/manual.md").json()

    assert body["deleted"] == "manual.md"
    assert not (tmp_path / "manual.md").exists()
    assert rebuilds == [1], "The indexes must be rebuilt after a deletion."


def test_delete_of_unknown_file_is_404(client, corpus_dir):
    tmp_path, rebuilds = corpus_dir

    assert client.delete("/api/corpus/nope.md").status_code == 404
    assert rebuilds == [], "A failed delete must not rebuild the indexes."


def test_delete_cannot_escape_the_corpus_directory(client, corpus_dir, tmp_path_factory):
    """A traversal path must never reach a file outside data/raw."""
    tmp_path, rebuilds = corpus_dir
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("do not delete me")

    response = client.delete("/api/corpus/..%2Fsecret.txt")

    assert response.status_code in (400, 404)
    assert outside.exists(), "A file outside the corpus directory was deleted."
    assert rebuilds == []
