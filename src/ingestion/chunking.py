import tiktoken
from typing import List, Dict
from src.utils.config import pick

class DocumentChunker:
    """Splits documents into overlapping token-aware chunks."""
    def __init__(self, chunk_size: int = None, overlap: int = None, encoding_name: str = "cl100k_base"):
        self.chunk_size = pick(chunk_size, "ingestion.chunk_size", 500)
        self.overlap = pick(overlap, "ingestion.chunk_overlap", 50)
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def split_text(self, text: str) -> List[str]:
        """Sliding window chunking using actual token lengths."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move window forward, factoring in the overlap
            start += (self.chunk_size - self.overlap)
            
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Processes a list of documents and injects citation markers into each chunk.
        """
        all_chunks = []
        global_chunk_id = 0
        
        for doc in documents:
            text = doc["content"]
            source = doc["metadata"].get("source", "Unknown")
            doc_hash = doc["metadata"].get("doc_hash", "")
            
            text_chunks = self.split_text(text)
            
            for i, chunk_str in enumerate(text_chunks):
                # Inject a citation marker so the answer can be traced to its source.
                augmented_chunk_text = f"[Source: {source} | Chunk: {i}]\n{chunk_str}"
                
                all_chunks.append({
                    "id": f"chunk_{global_chunk_id}",
                    "content": augmented_chunk_text,
                    "metadata": {
                        "source": source,
                        "doc_hash": doc_hash,
                        "chunk_index": i
                    }
                })
                global_chunk_id += 1
                
        return all_chunks