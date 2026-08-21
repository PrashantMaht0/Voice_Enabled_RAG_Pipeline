from typing import List, Tuple, Any, Optional
from src.vector_db.hybrid_store import HybridStore
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.config import pick

class HybridRetriever:
    """Hybrid retrieval, cross-encoder reranking and the confidence gate."""
    def __init__(self, confidence_threshold: float = None, top_k_rerank: int = None,
                 top_k_candidates: int = None, top_k_retrieval: int = None):
        self.store = HybridStore()
        self.reranker = CrossEncoderReranker()
        self.confidence_threshold = pick(confidence_threshold, "retrieval.confidence_threshold", 0.65)
        self.top_k_rerank = pick(top_k_rerank, "retrieval.top_k_rerank", 3)
        self.top_k_retrieval = pick(top_k_retrieval, "retrieval.top_k_retrieval", 30)
        # Cross-encoder cost is linear in pool size, so the pool stays small.
        self.top_k_candidates = pick(top_k_candidates, "retrieval.top_k_rerank_pool", 12)

    def search_and_rerank(self, query: str, tracer: Optional[Any] = None) -> Tuple[List[str], float]:
        # 1. Sparse Search
        if tracer: tracer.start_span("bm25")
        sparse_candidates = self.store.search_sparse(query, top_k=self.top_k_retrieval)
        if tracer: tracer.end_span("bm25", f"Keyword search — {len(sparse_candidates)} candidates")

        # 2. Dense Search
        if tracer: tracer.start_span("vector")
        dense_candidates = self.store.search_dense(query, top_k=self.top_k_retrieval)
        if tracer: tracer.end_span("vector", f"ANN search — {len(dense_candidates)} candidates")
        
        # 3. Hybrid Fusion over the candidates already fetched above
        if tracer: tracer.start_span("fusion")
        candidates = self.store.fuse(dense_candidates, sparse_candidates, top_k=self.top_k_candidates)
        if tracer: tracer.end_span("fusion", f"Hybrid merge — {len(candidates)} unique candidates")

        if not candidates:
            return [], 0.0

        # 4. Precision Reranking (Cross-Encoder)
        if tracer: tracer.start_span("rerank")
        ranked_results = self.reranker.rerank(query, candidates)
        if tracer: tracer.end_span("rerank", f"Cross-encoder scored {len(candidates)} chunks")
        
        # 5. Apply the Hard Threshold Gate
        valid_chunks = []
        highest_score = 0.0
        
        for i, res in enumerate(ranked_results[:self.top_k_rerank]):
            score = float(res["score"]) # Cast to float to prevent JSON serialization errors
            
            if i == 0:
                highest_score = score
                
            if score >= self.confidence_threshold:
                valid_chunks.append(res["chunk"])
                
        return valid_chunks, highest_score
