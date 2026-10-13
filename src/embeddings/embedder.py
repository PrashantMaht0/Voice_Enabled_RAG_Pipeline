"""Dense embeddings for semantic search: one normalized vector per chunk or query."""
from typing import List

from sentence_transformers import SentenceTransformer

from src.utils.config import pick


class DenseEmbedder:
    """A sentence transformer (`retrieval.dense_model`, BGE small by default) that returns unit length vectors,
    so a dot product in Qdrant is cosine similarity."""

    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = pick(model_name, "retrieval.dense_model", "BAAI/bge-small-en-v1.5")
        self.device = device
        print(f"Loading Embedding Model: {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_text(self, text: str) -> List[float]:
        """The normalized embedding of one string (a query)."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Normalized embeddings for many strings at once (ingestion)."""
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def get_dimension(self) -> int:
        """The vector size, used when the Qdrant collection is created."""
        return self.model.get_sentence_embedding_dimension()
