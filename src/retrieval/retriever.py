"""Retrieval for one question: hybrid search, cross encoder reranking, then the confidence gate."""
import os
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.reranker import CrossEncoderReranker
from src.utils.config import pick
from src.vector_db.hybrid_store import HybridStore


class HybridRetriever:
    """Keyword and vector search (one after the other: running them in parallel measured no faster in
    scope feature 3), fusion, reranking of a small pool, and the gate: only chunks scoring at least
    `retrieval.confidence_threshold` reach the answer model."""

    def __init__(self, confidence_threshold: float = None, top_k_rerank: int = None,
                 top_k_candidates: int = None, top_k_retrieval: int = None, raw_dir: str = None):
        self.store = HybridStore()
        self.raw_dir = pick(raw_dir, "paths.raw_data_dir", "data/raw")
        self.reranker = CrossEncoderReranker()
        self.confidence_threshold = pick(confidence_threshold, "retrieval.confidence_threshold", 0.65)
        self.top_k_rerank = pick(top_k_rerank, "retrieval.top_k_rerank", 3)
        self.top_k_retrieval = pick(top_k_retrieval, "retrieval.top_k_retrieval", 30)
        # Cross-encoder cost is linear in pool size, so the pool stays small.
        self.top_k_candidates = pick(top_k_candidates, "retrieval.top_k_rerank_pool", 12)

    def present_sources(self, candidates: List[Dict]) -> List[Dict]:
        """Drops chunks whose document was deleted, so it stops being cited before the rebuild finishes
        (spec 0008, AC-8). Without a corpus folder there is nothing to compare with."""
        if not os.path.isdir(self.raw_dir):
            return candidates
        present = set(os.listdir(self.raw_dir))
        return [c for c in candidates if (c.get("metadata") or {}).get("source") in present]

    def search_and_rerank(self, query: str, tracer: Optional[Any] = None) -> Tuple[List[str], float]:
        """(chunks that passed the gate, the top reranker score)."""
        chunks, score, _ = self.search_and_rerank_detailed(query, tracer)
        return chunks, score

    def search_and_rerank_detailed(self, query: str, tracer: Optional[Any] = None) -> Tuple[List[str], float, List[Dict]]:
        """Chunks, the top score, and for each kept chunk its id, source, page, kind and score (spec 0006)."""
        generation = self.store.db_manager.active         # one generation for the whole question (spec 0008)
        if tracer: tracer.start_span("bm25")
        sparse_candidates = self.store.search_sparse(query, top_k=self.top_k_retrieval, generation=generation)
        if tracer: tracer.end_span("bm25", f"Keyword search — {len(sparse_candidates)} candidates")

        if tracer: tracer.start_span("vector")
        dense_candidates = self.store.search_dense(query, top_k=self.top_k_retrieval, generation=generation)
        if tracer: tracer.end_span("vector", f"ANN search — {len(dense_candidates)} candidates")
        
        if tracer: tracer.start_span("fusion")
        candidates = self.present_sources(
            self.store.fuse(dense_candidates, sparse_candidates, top_k=self.top_k_candidates))
        if tracer: tracer.end_span("fusion", f"Hybrid merge — {len(candidates)} unique candidates")

        if not candidates:
            return [], 0.0, []

        if tracer: tracer.start_span("rerank")
        ranked_results = self.reranker.rerank(query, candidates)
        if tracer: tracer.end_span("rerank", f"Cross-encoder scored {len(candidates)} chunks")
        
        # The gate: the top top_k_rerank chunks that score at least the threshold
        valid_chunks, details = [], []
        highest_score = 0.0
        for i, res in enumerate(ranked_results[:self.top_k_rerank]):
            score = float(res["score"])           # a numpy float does not serialize to JSON
            if i == 0:
                highest_score = score
            if score >= self.confidence_threshold:
                valid_chunks.append(res["chunk"])
                meta = res.get("metadata") or {}
                doc_hash, index = meta.get("doc_hash", ""), meta.get("chunk_index")
                details.append({"chunk_id": f"{doc_hash[:16]}_{index}" if index is not None else None,
                                "source": meta.get("source"), "page": meta.get("page"),
                                "kind": meta.get("kind", "text"), "score": round(score, 4)})

        return valid_chunks, highest_score, details
