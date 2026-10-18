"""Chunking: pages and files split into overlapping token windows; table and figure chunks kept whole."""
from typing import Dict, List

import tiktoken

from src.utils.config import pick
from src.ingestion.tables import MAX_TABLE_TOKENS
from src.ingestion.figures import figure_body

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
            if end >= len(tokens):
                break  # this window already reached the end; another would repeat it

            # Move window forward, factoring in the overlap
            start += (self.chunk_size - self.overlap)
            
        return chunks

    def split_table(self, markdown: str, max_tokens: int = MAX_TABLE_TOKENS) -> List[str]:
        """Splits a Markdown table by rows, repeating the header; a single oversize row stands alone."""
        lines = markdown.splitlines()
        header, rows = lines[:2], lines[2:]
        size = lambda part: len(self.tokenizer.encode("\n".join(header + part)))
        parts, current = [], []
        for row in rows:
            if current and size(current + [row]) > max_tokens:
                parts.append(current)
                current = []
            current.append(row)
        if current or not parts:
            parts.append(current)
        return ["\n".join(header + part) for part in parts]

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Processes a list of documents and injects citation markers into each chunk.
        """
        all_chunks = []
        
        for doc in documents:
            text = doc["content"]
            source = doc["metadata"].get("source", "Unknown")
            doc_hash = doc["metadata"].get("doc_hash", "")
            
            # PDFs are split page by page, so no chunk ever spans a page break.
            if "pages" in doc:
                pieces = [(page, chunk) for page, page_text in doc["pages"]
                          for chunk in self.split_text(page_text)]
            else:
                pieces = [(None, chunk) for chunk in self.split_text(text)]

            for i, (page, chunk_str) in enumerate(pieces):
                # Inject a citation marker so the answer can be traced to its source.
                locator = f"Page: {page}" if page is not None else f"Chunk: {i}"
                augmented_chunk_text = f"[Source: {source} | {locator}]\n{chunk_str}"

                metadata = {
                    "source": source,
                    "doc_hash": doc_hash,
                    "chunk_index": i,
                    "kind": "text",
                }
                if page is not None:
                    metadata["page"] = page

                all_chunks.append({
                    # Content addressed, so re-ingesting the same text gives the same ids
                    "id": f"{doc_hash[:16]}_{i}",
                    "content": augmented_chunk_text,
                    "metadata": metadata
                })

            # Tables come after every text chunk, so text chunk ids match under every parser.
            i = len(pieces)
            for page, _top, markdown in doc.get("tables", []):
                for part in self.split_table(markdown):
                    all_chunks.append({
                        "id": f"{doc_hash[:16]}_{i}",
                        "content": f"[Source: {source} | Page: {page}]\n{part}",
                        "metadata": {
                            "source": source,
                            "doc_hash": doc_hash,
                            "chunk_index": i,
                            "page": page,
                            "kind": "table",
                            "table_parser": doc.get("table_parser"),
                        },
                    })
                    i += 1

            # Figures come after the tables and are never split, so table chunk ids stay put too.
            for page, _top, caption, figure in doc.get("figures", []):
                all_chunks.append({
                    "id": f"{doc_hash[:16]}_{i}",
                    "content": f"[Source: {source} | Page: {page}]\n{figure_body(caption, figure)}",
                    "metadata": {
                        "source": source,
                        "doc_hash": doc_hash,
                        "chunk_index": i,
                        "page": page,
                        "kind": "figure",
                        "figure_parser": doc.get("figure_parser"),
                        "figure": figure,
                    },
                })
                i += 1

        return all_chunks