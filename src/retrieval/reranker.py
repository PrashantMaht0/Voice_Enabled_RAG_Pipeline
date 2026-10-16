"""Cross encoder reranking: the retrieved candidates are scored against the query, best first."""
import os
from typing import Dict, List

from flashrank import Ranker, RerankRequest

from src.pipeline.retrieval import rerank_text  # noqa: F401  (imported from here by older callers)
from src.utils.config import pick


class CrossEncoderReranker:
    """FlashRank's ms-marco cross encoder (`retrieval.reranker_model`): it reads the query and each chunk
    together, so its score (0 to 1) is what the confidence gate compares with its threshold."""

    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name = pick(model_name, "retrieval.reranker_model", "ms-marco-MiniLM-L-12-v2")
        # A persistent cache (not /tmp), so the model downloads once; Docker points it at a volume.
        cache_dir = cache_dir or os.environ.get("FLASHRANK_CACHE_DIR") or os.path.expanduser("~/.cache/flashrank")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        print(f"Loading FlashRank Reranker: {self.model_name}...")
        self.ranker = Ranker(model_name=self.model_name, cache_dir=cache_dir)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """The candidates as `{chunk, score, metadata}`, highest score first; empty in, empty out."""
        if not candidates:
            return []
        # Scored on rerank_text; the original chunk is what goes downstream.
        passages = [{"id": str(i), "text": rerank_text(c.get("content", ""))} for i, c in enumerate(candidates)]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        ranked_chunks = []
        for res in results:
            original = candidates[int(res["id"])]
            ranked_chunks.append({
                "chunk": original.get("content", ""),
                "score": res["score"],
                "metadata": original.get("metadata", {})
            })
        return ranked_chunks