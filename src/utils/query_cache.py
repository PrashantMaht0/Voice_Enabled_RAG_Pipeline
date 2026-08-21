"""Two-level query cache invalidated whenever the corpus changes."""
import hashlib
import os
import threading
from collections import OrderedDict

import numpy as np


def corpus_fingerprint(raw_dir: str) -> str:
    """Cheap stat-based signature of the source corpus."""
    if not os.path.isdir(raw_dir):
        return "empty"

    digest = hashlib.sha256()
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


class QueryCache:
    """Exact + semantic cache. Thread-safe; the API serves requests from a pool."""

    def __init__(self, embedder=None, raw_dir: str = "data/raw", max_entries: int = 256,
                 similarity_threshold: float = 0.95, enabled: bool = True,
                 embedder_provider=None):
        # Without an embedder the cache still works, as exact-match only.
        self.embedder = embedder
        self.embedder_provider = embedder_provider
        self.raw_dir = raw_dir
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled

        self._lock = threading.Lock()
        self._entries = OrderedDict()          # key -> payload
        self._vectors = {}                     # key -> np.ndarray (L2-normalized)
        self._fingerprint = corpus_fingerprint(raw_dir)
        self.hits_exact = 0
        self.hits_semantic = 0
        self.misses = 0
        self.invalidations = 0

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join((query or "").lower().split())

    def _resolve_embedder(self):
        if self.embedder is None and self.embedder_provider is not None:
            try:
                self.embedder = self.embedder_provider()
            except Exception:
                self.embedder_provider = None      # do not retry a broken provider
        return self.embedder

    def _embed(self, query: str):
        if self._resolve_embedder() is None:
            return None
        try:
            vector = np.asarray(self.embedder.embed_text(query), dtype=np.float32)
        except Exception:
            return None
        norm = np.linalg.norm(vector)
        return vector / norm if norm else None

    def _check_corpus_unlocked(self):
        """Drops everything when the source corpus no longer matches."""
        current = corpus_fingerprint(self.raw_dir)
        if current != self._fingerprint:
            self._fingerprint = current
            if self._entries:
                self.invalidations += 1
            self._entries.clear()
            self._vectors.clear()
            return True
        return False

    # ------------------------------------------------------------------- api
    def get(self, query: str):
        """Returns (payload, "exact" | "semantic") or (None, None)."""
        if not self.enabled:
            return None, None

        key = self._normalize(query)
        with self._lock:
            self._check_corpus_unlocked()

            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits_exact += 1
                return self._entries[key], "exact"

            candidates = list(self._vectors.items())

        # Embedding happens outside the lock: it is the slow part.
        vector = self._embed(query) if candidates else None
        if vector is None:
            with self._lock:
                self.misses += 1
            return None, None

        best_key, best_score = None, -1.0
        for cached_key, cached_vector in candidates:
            score = float(np.dot(vector, cached_vector))
            if score > best_score:
                best_key, best_score = cached_key, score

        with self._lock:
            if best_key is not None and best_score >= self.similarity_threshold \
                    and best_key in self._entries:
                self._entries.move_to_end(best_key)
                self.hits_semantic += 1
                return self._entries[best_key], "semantic"
            self.misses += 1
            return None, None

    def put(self, query: str, payload: dict):
        if not self.enabled:
            return

        key = self._normalize(query)
        vector = self._embed(query)

        with self._lock:
            self._check_corpus_unlocked()
            self._entries[key] = payload
            self._entries.move_to_end(key)
            if vector is not None:
                self._vectors[key] = vector

            while len(self._entries) > self.max_entries:
                evicted, _ = self._entries.popitem(last=False)
                self._vectors.pop(evicted, None)

    def invalidate(self, reason: str = "explicit"):
        """Called when the pipeline itself changes the corpus."""
        with self._lock:
            had_entries = bool(self._entries)
            self._entries.clear()
            self._vectors.clear()
            self._fingerprint = corpus_fingerprint(self.raw_dir)
            if had_entries:
                self.invalidations += 1
        print(f"[Cache] Invalidated ({reason}).")

    def stats(self) -> dict:
        with self._lock:
            lookups = self.hits_exact + self.hits_semantic + self.misses
            hits = self.hits_exact + self.hits_semantic
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits_exact": self.hits_exact,
                "hits_semantic": self.hits_semantic,
                "misses": self.misses,
                "invalidations": self.invalidations,
                "hit_rate": round(hits / lookups, 3) if lookups else 0.0,
            }
