"""Shared fixtures: every test gets its own disabled cache, a throwaway trace store, no grounding checker and
no figure reader, plus a small real index for the integration tests."""
import pytest

from src.api import routes
from src.utils.query_cache import QueryCache
from src.vector_db.vector_store import VectorDBManager
from src.vector_db.hybrid_store import HybridStore


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """The cache is a process-wide singleton, so each test gets a disabled one."""
    monkeypatch.setattr(routes, "_cache", QueryCache(enabled=False))


@pytest.fixture(autouse=True)
def isolated_traces(monkeypatch, tmp_path):
    """Each test writes traces to its own throwaway database, never the real one beside the index."""
    from src.observability.store import TraceStore
    monkeypatch.setattr(routes, "_traces", TraceStore(str(tmp_path / "traces.sqlite3")))


@pytest.fixture(autouse=True)
def grounding_off(monkeypatch):
    """The grounding checker follows config.yaml (a live Ollama judge); tests use the pass through checker
    unless they set one themselves."""
    from src.verification import grounding
    monkeypatch.setattr(grounding, "configured_checker", lambda: "none")
    monkeypatch.setattr(grounding, "_checker", grounding.PassThrough())
    monkeypatch.setattr(grounding, "_checker_failed_at", None)


@pytest.fixture(autouse=True)
def figures_off(monkeypatch):
    """Figure reading follows config.yaml; tests turn it on themselves, so it is off by default here."""
    from src.ingestion import figures
    monkeypatch.setattr(figures, "configured_parser", lambda: "none")
    monkeypatch.setattr(figures, "_effective_parser", "none")

SAMPLE_CHUNKS = [
    {
        "id": "chunk_1",
        "content": "The router TX-9000 requires port 443 open for SSL connections.",
        "metadata": {"source": "network_guide.pdf"}
    },
    {
        "id": "chunk_2",
        "content": "Error ERR_SSL_PROTOCOL indicates a mismatched certificate.",
        "metadata": {"source": "error_codes.md"}
    }
]

@pytest.fixture(scope="session")
def sample_chunks():
    return SAMPLE_CHUNKS

@pytest.fixture(scope="session")
def db_manager(tmp_path_factory, sample_chunks):
    """Throwaway Qdrant + Tantivy pair. Never touches ./vectordb, which the API server locks."""
    root = tmp_path_factory.mktemp("vectordb")
    manager = VectorDBManager(
        qdrant_path=str(root / "qdrant"),
        tantivy_path=str(root / "tantivy"),
        collection_name="test_vocal_rag"
    )
    manager.ingest_chunks(sample_chunks)
    return manager

@pytest.fixture(scope="session")
def store(db_manager):
    return HybridStore(db_manager=db_manager)
