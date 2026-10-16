"""Hybrid search: dense (Qdrant) and keyword (Tantivy BM25) candidates merged by reciprocal rank fusion."""
from typing import Dict, List

from src.pipeline.retrieval import fuse
from src.utils.config import pick
from src.vector_db.vector_store import VectorDBManager


class HybridStore:
    """Searches both stores of one VectorDBManager and fuses their candidates."""

    def __init__(self, hybrid_alpha: float = None, db_manager: VectorDBManager = None):
        # Embedded Qdrant allows one handle per folder, so callers may pass the manager they already have.
        self.db_manager = db_manager or VectorDBManager()
        self.hybrid_alpha = pick(hybrid_alpha, "retrieval.hybrid_alpha", 0.5)

    def search_dense(self, query: str, top_k: int, generation=None) -> List[Dict]:
        """The top_k nearest chunks by embedding similarity, in the given generation (default: the active one)."""
        generation = generation or self.db_manager.active
        vector = self.db_manager.embedder.embed_text(query)
        response = self.db_manager.qdrant_client.query_points(
            collection_name=generation.collection_name, query=vector, limit=top_k)
        return [{"id": res.payload["chunk_id"], "content": res.payload["content"],
                 "metadata": res.payload["metadata"], "score": res.score, "source_engine": "qdrant"}
                for res in response.points]

    def search_sparse(self, query: str, top_k: int, generation=None) -> List[Dict]:
        """The top_k chunks by BM25 keyword score, in the given generation (default: the active one); empty when
        Tantivy cannot parse the query."""
        index = (generation or self.db_manager.active).tantivy_index
        index.reload()                            # see documents committed since the last search
        searcher = index.searcher()
        try:
            hits = searcher.search(index.parse_query(query, ["content"]), limit=top_k).hits
        except Exception as e:
            print(f"[Tantivy] Search error: {e}")
            return []

        results = []
        for score, doc_address in hits:
            try:
                doc = searcher.doc(doc_address).to_dict()
                results.append({"id": doc.get("chunk_id", [""])[0], "content": doc.get("content", [""])[0],
                                "metadata": {"source": doc.get("source_doc", ["unknown"])[0]},
                                "score": score, "source_engine": "tantivy"})
            except Exception as e:
                print(f"[Tantivy] Document extraction error: {e}")
        return results

    def fuse(self, dense_results: List[Dict], sparse_results: List[Dict], top_k: int = 30) -> List[Dict]:
        """Both candidate lists merged by reciprocal rank fusion (`retrieval.hybrid_alpha` is the dense weight)."""
        return fuse(dense_results, sparse_results, self.hybrid_alpha, top_k)
