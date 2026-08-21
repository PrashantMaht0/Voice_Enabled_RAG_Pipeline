import numpy as np
from typing import List, Union
from src.utils.config import pick
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_INSTALLED = True
except ImportError:
    SENTENCE_TRANSFORMERS_INSTALLED = False


class DenseEmbedder:
    """Generates L2-normalized embeddings for semantic search."""
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = pick(model_name, "retrieval.dense_model", "BAAI/bge-small-en-v1.5")
        self.device = device
        self.model = None

        if SENTENCE_TRANSFORMERS_INSTALLED:
            print(f"Loading Embedding Model: {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            print(f"WARNING: sentence-transformers not installed. Embedder running in mock mode.")

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string and applies L2 normalization for cosine similarity.
        """
        if not self.model:
            # Fallback mock for development environment
            return [0.0] * 384 

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of texts for high-throughput ingestion.
        """
        if not self.model:
            # Fallback mock for development environment
            return [[0.0] * 384 for _ in texts]

        # Process the batch
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()
        
    def get_dimension(self) -> int:
        """Returns the embedding dimension size for vector DB initialization."""
        if not self.model:
            return 384 
        return self.model.get_sentence_embedding_dimension()