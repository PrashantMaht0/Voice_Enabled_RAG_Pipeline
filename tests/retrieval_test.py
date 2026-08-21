import pytest

from src.retrieval.retriever import HybridRetriever


@pytest.mark.integration
def test_sparse_bm25_retrieval(store):
    """BM25 should easily catch exact token matches like 'ERR_SSL_PROTOCOL'."""
    results = store.search_sparse("ERR_SSL_PROTOCOL", top_k=1)
    assert len(results) > 0, "BM25 failed to retrieve results."
    assert "ERR_SSL_PROTOCOL" in results[0]["content"], "BM25 retrieved the wrong chunk."


@pytest.mark.integration
def test_dense_semantic_retrieval(store):
    """Dense vectors should match conceptual questions to the right chunk."""
    # "secure connection" isn't exactly in the text, but semantically maps to "SSL connection"
    results = store.search_dense("How do I set up a secure connection?", top_k=1)
    assert len(results) > 0, "Dense search failed to retrieve results."
    assert "port 443" in results[0]["content"], "Dense search retrieved the wrong chunk."


# --- Confidence gate (no models: the gate is pure ranking arithmetic) ---

class _StubStore:
    """Returns one fixed candidate regardless of the query."""
    CANDIDATES = [{"id": "c1", "content": "stub", "metadata": {}}]

    def search_sparse(self, query, top_k):
        return self.CANDIDATES

    def search_dense(self, query, top_k):
        return self.CANDIDATES

    def fuse(self, dense_results, sparse_results, top_k=30):
        return self.CANDIDATES


class _StubReranker:
    """Emits a caller-supplied score list so the gate can be driven directly."""
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, candidates):
        return [{"chunk": f"chunk_{i}", "score": s, "metadata": {}} for i, s in enumerate(self.scores)]


def build_retriever(scores):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.store = _StubStore()
    retriever.reranker = _StubReranker(scores)
    retriever.confidence_threshold = 0.65
    retriever.top_k_rerank = 3
    retriever.top_k_candidates = 12
    retriever.top_k_retrieval = 30
    return retriever


def test_gate_rejects_low_confidence():
    """Every chunk below 0.65 must be dropped — this is the zero-hallucination gate."""
    chunks, score = build_retriever([0.64, 0.30]).search_and_rerank("anything")
    assert chunks == []
    assert score == pytest.approx(0.64)


def test_gate_keeps_only_chunks_at_or_above_threshold():
    chunks, score = build_retriever([0.90, 0.65, 0.20]).search_and_rerank("anything")
    assert chunks == ["chunk_0", "chunk_1"]
    assert score == pytest.approx(0.90)


def test_gate_respects_top_k_rerank():
    """Only the top 3 reranked results are ever considered."""
    chunks, _ = build_retriever([0.9, 0.9, 0.9, 0.9]).search_and_rerank("anything")
    assert len(chunks) == 3
