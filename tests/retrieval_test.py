"""Retrieval: keyword and vector search on a small real index (integration) and the confidence gate."""
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
    db_manager = type("Manager", (), {"active": None})()

    def search_sparse(self, query, top_k, generation=None):
        return self.CANDIDATES

    def search_dense(self, query, top_k, generation=None):
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
    retriever.raw_dir = "/nonexistent"        # no corpus folder: the deleted source filter is skipped
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


# --- spec 0008, AC-8: a deleted document stops being cited at once ---

def candidates(*sources):
    return [{"id": f"c{i}", "content": "x", "metadata": {"source": s}} for i, s in enumerate(sources)]


def test_chunks_from_a_deleted_document_are_dropped_before_reranking(tmp_path):
    """covers: AC-8"""
    (tmp_path / "kept.pdf").write_bytes(b"%PDF-")
    retriever = build_retriever([0.9])
    retriever.raw_dir = str(tmp_path)

    kept = retriever.present_sources(candidates("kept.pdf", "deleted.md", "kept.pdf"))

    assert [c["metadata"]["source"] for c in kept] == ["kept.pdf", "kept.pdf"]


def test_chunks_without_a_source_are_dropped_when_a_corpus_folder_exists(tmp_path):
    """covers: AC-8"""
    retriever = build_retriever([0.9])
    retriever.raw_dir = str(tmp_path)

    assert retriever.present_sources([{"id": "c", "content": "x", "metadata": None}]) == []


def test_without_a_corpus_folder_nothing_is_filtered():
    """covers: AC-8"""
    retriever = build_retriever([0.9])

    assert retriever.present_sources(candidates("a.md")) == candidates("a.md")


def test_one_question_reads_one_generation_for_both_searches():
    """A swap between the dense and keyword search must not mix generations. covers: AC-7"""
    seen = []

    class Store(_StubStore):
        def search_sparse(self, query, top_k, generation=None):
            seen.append(generation)
            return self.CANDIDATES

        def search_dense(self, query, top_k, generation=None):
            seen.append(generation)
            return self.CANDIDATES

    retriever = build_retriever([0.9])
    retriever.store = Store()
    retriever.store.db_manager = type("Manager", (), {"active": "generation 7"})()

    retriever.search_and_rerank("anything")

    assert seen == ["generation 7", "generation 7"]
