import pytest

from src.api import routes
from src.utils.query_cache import QueryCache
from src.vector_db.vector_store import VectorDBManager
from src.vector_db.hybrid_store import HybridStore


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """
    The query cache is a process-wide singleton, so without this one test's answer
    is served to the next. Unit tests exercise the uncached path; cache behaviour
    is covered directly in tests/cache_test.py.
    """
    monkeypatch.setattr(routes, "_cache", QueryCache(enabled=False))

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
