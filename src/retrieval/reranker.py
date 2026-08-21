from typing import List, Dict
from src.utils.config import pick
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_INSTALLED = True
except ImportError:
    FLASHRANK_INSTALLED = False

class CrossEncoderReranker:
    """Scores query and chunk jointly to reorder retrieval candidates."""
    def __init__(self, model_name: str = None, cache_dir: str = "/tmp"):
        self.model_name = pick(model_name, "retrieval.reranker_model", "ms-marco-MiniLM-L-12-v2")
        self.ranker = None
        
        if FLASHRANK_INSTALLED:
            print(f"Loading FlashRank Reranker: {self.model_name}...")
            self.ranker = Ranker(model_name=self.model_name, cache_dir=cache_dir)
        else:
            print("WARNING: FlashRank not installed. Reranker running in mock bypass mode.")

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Takes the top-K candidates from the broad hybrid search and performs precise scoring.
        """
        if not self.ranker or not candidates:
            # Mock bypass mode: return the top 3 with a dummy high confidence
            return [{"chunk": c.get("content", ""), "score": 0.99, "metadata": c.get("metadata", {})} for c in candidates[:3]]
        
        # Prepare the specific payload required by FlashRank
        passages = [
            {
                "id": c.get("id", str(i)), 
                "text": c.get("content", ""), 
                "meta": c.get("metadata", {})
            } 
            for i, c in enumerate(candidates)
        ]
        
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)
        
        ranked_chunks = []
        for res in results:
            ranked_chunks.append({
                "chunk": res["text"],
                "score": res["score"],
                "metadata": res["meta"]
            })
            
        return ranked_chunks