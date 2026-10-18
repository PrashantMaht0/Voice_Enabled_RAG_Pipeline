"""The browser demo's in memory index: dense search by cosine over unit vectors (numpy) and keyword
search by BM25, over the sample documents plus the visitor's uploads, in Pyodide.

Keyword search follows tantivy's defaults (the server's keyword engine): split on anything that is not a letter or
digit, drop tokens over 40 characters, lowercase, any query term may match, BM25 with k1 1.2 and b 0.75. Document
frequencies and the average length are computed over the documents in scope, at query time (cheap at demo size).
Candidates have the server's shape: {id, content, score, metadata: {source, doc_hash, chunk_index, page, kind}}."""
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np

K1, B, MAX_TOKEN = 1.2, 0.75, 40
_SPLIT = re.compile(r"[^0-9a-z]+")


def tokens(text: str) -> List[str]:
    """tantivy's default analyzer: lowercase, split on non alphanumerics, drop tokens over 40 characters."""
    return [t for t in _SPLIT.split(text.lower()) if t and len(t) <= MAX_TOKEN]


class DemoIndex:
    """Chunks and vectors per document; a document is all sample or all upload."""

    def __init__(self, dims: int = 384):
        self.dims = dims
        self.docs: Dict[str, dict] = {}          # name -> {chunks, vectors, counts, lengths}

    def add_doc(self, name: str, chunks: List[Dict], vectors) -> None:
        """Replaces a document's chunks; `vectors` is float32 bytes or an array, one row per chunk, unit length."""
        array = np.frombuffer(vectors, dtype=np.float32) if isinstance(vectors, (bytes, bytearray, memoryview)) \
            else np.asarray(vectors, dtype=np.float32)
        array = array.reshape(len(chunks), self.dims) if len(chunks) else np.zeros((0, self.dims), np.float32)
        counts = [Counter(tokens(c["content"])) for c in chunks]
        self.docs[name] = {"chunks": chunks, "vectors": array, "counts": counts,
                           "lengths": [sum(c.values()) for c in counts]}

    def load(self, chunks: List[Dict], vectors) -> None:
        """Many documents at once (the sample index): chunks carry `doc`, rows follow chunk order."""
        array = np.frombuffer(vectors, dtype=np.float32) if isinstance(vectors, (bytes, bytearray, memoryview)) \
            else np.asarray(vectors, dtype=np.float32)
        array = array.reshape(len(chunks), self.dims)
        rows: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            rows.setdefault(c["doc"], []).append(i)
        for name, idx in rows.items():
            self.add_doc(name, [chunks[i] for i in idx], array[idx])

    def remove_doc(self, name: str) -> None:
        self.docs.pop(name, None)

    def names(self) -> List[str]:
        return list(self.docs)

    def _scope(self, docs: Optional[Iterable[str]]) -> List[str]:
        return list(self.docs) if docs is None else [d for d in docs if d in self.docs]

    @staticmethod
    def candidate(chunk: Dict, score: float) -> Dict:
        return {"id": chunk["id"], "content": chunk["content"], "score": float(score),
                "metadata": {"source": chunk["doc"], "doc_hash": chunk.get("doc_hash") or "",
                             "chunk_index": chunk.get("chunk_index"), "page": chunk.get("page"),
                             "kind": chunk.get("kind") or "text"}}

    def search_dense(self, vector, top_k: int, docs: Optional[Iterable[str]] = None) -> List[Dict]:
        """The top_k chunks by cosine similarity (vectors are unit length, so a dot product)."""
        query = np.asarray(vector, dtype=np.float32)
        scored = []
        for name in self._scope(docs):
            d = self.docs[name]
            if len(d["chunks"]):
                for chunk, score in zip(d["chunks"], d["vectors"] @ query):
                    scored.append((float(score), chunk))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [self.candidate(chunk, score) for score, chunk in scored[:top_k]]

    def search_sparse(self, query: str, top_k: int, docs: Optional[Iterable[str]] = None) -> List[Dict]:
        """The top_k chunks by BM25 over the documents in scope; empty when the query has no searchable term."""
        terms = list(dict.fromkeys(tokens(query)))
        scope = self._scope(docs)
        entries = [(chunk, counts, length) for name in scope
                   for chunk, counts, length in zip(self.docs[name]["chunks"], self.docs[name]["counts"],
                                                    self.docs[name]["lengths"])]
        if not terms or not entries:
            return []
        n = len(entries)
        avgdl = sum(e[2] for e in entries) / n or 1.0
        df = {t: sum(1 for e in entries if t in e[1]) for t in terms}
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in terms if df[t]}
        scored = []
        for chunk, counts, length in entries:
            score = 0.0
            for t, weight in idf.items():
                tf = counts.get(t, 0)
                if tf:
                    score += weight * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avgdl))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [self.candidate(chunk, score) for score, chunk in scored[:top_k]]
