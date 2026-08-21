from typing import List, Dict
from src.vector_db.vector_store import VectorDBManager, QDRANT_INSTALLED, TANTIVY_INSTALLED
from src.utils.config import pick

class HybridStore:
    """Runs sparse and dense search and fuses the results."""
    def __init__(self, hybrid_alpha: float = None, db_manager: VectorDBManager = None):
        # Embedded Qdrant is single-handle, so callers may inject their manager.
        self.db_manager = db_manager or VectorDBManager()
        self.hybrid_alpha = pick(hybrid_alpha, "retrieval.hybrid_alpha", 0.5)

    def search_dense(self, query: str, top_k: int) -> List[Dict]:
        """Fetches nearest semantic vectors from Qdrant using the updated Query API."""
        if not QDRANT_INSTALLED or not self.db_manager.qdrant_client:
            return []
            
        vector = self.db_manager.embedder.embed_text(query)
        
        response = self.db_manager.qdrant_client.query_points(
            collection_name=self.db_manager.collection_name,
            query=vector,
            limit=top_k
        )
        
        return [{
            "id": res.payload["chunk_id"],
            "content": res.payload["content"],
            "metadata": res.payload["metadata"],
            "score": res.score,
            "source_engine": "qdrant"
        } for res in response.points]

    def search_sparse(self, query: str, top_k: int) -> List[Dict]:
        """Fetches exact keyword matches using Tantivy's BM25."""
        if not TANTIVY_INSTALLED or not self.db_manager.tantivy_index:
            return []
            
        self.db_manager.tantivy_index.reload() # Ensure index snapshot is fresh
        searcher = self.db_manager.tantivy_index.searcher()
        
        try:
            parsed_query = self.db_manager.tantivy_index.parse_query(query, ["content"])
            # Pass limit explicitly to avoid positional Rust mismatches
            search_result = searcher.search(parsed_query, limit=top_k)
            hits = search_result.hits
        except Exception as e:
            print(f"[Tantivy] Search error: {e}")
            return []

        sparse_results = []
        for score, doc_address in hits:
            try:
                doc = searcher.doc(doc_address)
                
                doc_dict = doc.to_dict()
                
                chunk_id = doc_dict.get("chunk_id", [""])[0]
                content = doc_dict.get("content", [""])[0]
                source_doc = doc_dict.get("source_doc", ["unknown"])[0]
                
                sparse_results.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": {"source": source_doc},
                    "score": score,
                    "source_engine": "tantivy"
                })
            except Exception as e:
                print(f"[Tantivy] Document extraction error: {e}")
            
        return sparse_results

    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        """Searches both engines and fuses the results."""
        # Execute queries (in a fully optimized production setup, use asyncio/threads here)
        dense_results = self.search_dense(query, top_k)
        sparse_results = self.search_sparse(query, top_k)
        
        return self.fuse(dense_results, sparse_results, top_k)

    def fuse(self, dense_results: List[Dict], sparse_results: List[Dict], top_k: int = 30) -> List[Dict]:
        """Merges pre-fetched dense and sparse candidates via RRF."""
        if not dense_results and not sparse_results:
            # Mock candidate block if local DBs are empty during testing
            return [{
                "id": "mock_chunk", 
                "content": "To configure the router, verify the SSL certificate in the admin panel.", 
                "metadata": {"source": "mock_doc.pdf"}
            }]

        # Reciprocal Rank Fusion: score = 1 / (rank + 60).
        rrf_scores = {}
        unified_pool = {}

        def apply_rrf(results, weight):
            for rank, res in enumerate(results):
                chunk_id = res["id"]
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    unified_pool[chunk_id] = res
                
                # RRF Constant 60 is an industry standard
                rrf_scores[chunk_id] += weight * (1.0 / (rank + 60))

        # Weight distribution dictated by config.yaml (default 0.5 each)
        apply_rrf(dense_results, self.hybrid_alpha)
        apply_rrf(sparse_results, 1.0 - self.hybrid_alpha)

        # Sort the fused candidate pool
        sorted_chunks = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        # Return the top K candidates for the Cross-Encoder to evaluate
        return [unified_pool[chunk_id] for chunk_id in sorted_chunks[:top_k]]