"""The index on disk: an embedded Qdrant collection (vectors) and a Tantivy index (keywords), built together."""
import json
import os
import shutil
import threading
import uuid
from typing import Dict, List, NamedTuple, Optional

import tantivy
from qdrant_client import QdrantClient, models

from src.embeddings.embedder import DenseEmbedder
from src.utils.config import pick

# Bump when the chunk layout changes, so an older index is rebuilt at startup.
# 2: PDFs are chunked page by page and their headers cite "Page: N".
# 3: chunk ids come from a SHA256 content fingerprint, so they are the same on every run.
# 4: table chunks (spec 0004), `kind` in metadata, and the table fingerprint in the marker.
# 5: figure chunks (spec 0005); the marker's fingerprint covers tables and figures.
INDEX_FORMAT_VERSION = 5
INDEX_MARKER = "index_version.json"
UPSERT_BATCH = 256


class Generation(NamedTuple):
    """One complete copy of the index (spec 0008): generation 0 uses the base names, N >= 1 adds `__gN`."""
    number: int
    collection_name: str
    tantivy_path: str
    tantivy_index: object


class BuildCancelled(Exception):
    """The API is shutting down; the partial generation is removed."""


def tantivy_schema():
    """The BM25 keyword index schema: chunk id, content and source, all stored."""
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("chunk_id", stored=True)
    schema_builder.add_text_field("content", stored=True)
    schema_builder.add_text_field("source_doc", stored=True)
    return schema_builder.build()


class VectorDBManager:
    """Owns both stores: opening them, building new generations from chunks and swapping them in, and the format
    marker beside them.

    Embedded Qdrant allows one process per folder, so only the API (or a test with its own folder) opens one.
    `collection_name`, `tantivy_path` and `tantivy_index` always describe the active generation."""

    def __init__(self, qdrant_path: str = None, tantivy_path: str = None, collection_name: str = None):
        self.qdrant_path = pick(qdrant_path, "paths.qdrant_storage", "./vectordb/qdrant_storage")
        self.base_tantivy_path = pick(tantivy_path, "paths.tantivy_index", "./vectordb/tantivy_index")
        self.base_collection = pick(collection_name, "paths.collection_name", "vocal_rag_docs")

        os.makedirs(self.qdrant_path, exist_ok=True)
        self.embedder = DenseEmbedder()
        self.qdrant_client = QdrantClient(path=self.qdrant_path)
        self._lock = threading.Lock()
        marker = self.read_marker()
        # Remembered for cleanup_orphans: a startup rebuild writes a fresh marker, which must not count as proof
        # that the generations already on disk are orphans (spec 0008, AC-7)
        self.opened_with_marker = marker is not None
        number = int(marker.get("generation", 0)) if marker else 0
        self._active = self._open_generation(number)

    # --- generations ---

    def collection_for(self, number: int) -> str:
        return self.base_collection if number == 0 else f"{self.base_collection}__g{number}"

    def tantivy_path_for(self, number: int) -> str:
        return self.base_tantivy_path if number == 0 else f"{self.base_tantivy_path}__g{number}"

    @property
    def active(self) -> Generation:
        """The generation questions read from; one snapshot per question (spec 0008, AC-7)."""
        with self._lock:
            return self._active

    @property
    def collection_name(self) -> str:
        return self.active.collection_name

    @property
    def tantivy_path(self) -> str:
        return self.active.tantivy_path

    @property
    def tantivy_index(self):
        return self.active.tantivy_index

    def _ensure_collection(self, name: str = None):
        """Creates the collection if missing, reusing the existing client."""
        name = name or self.collection_name
        collections = [col.name for col in self.qdrant_client.get_collections().collections]
        if name not in collections:
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.embedder.get_dimension(),
                    distance=models.Distance.COSINE,   # the embeddings are normalized
                )
            )

    def _open_generation(self, number: int) -> Generation:
        """Opens (creating when missing) a generation's collection and keyword index."""
        name, path = self.collection_for(number), self.tantivy_path_for(number)
        self._ensure_collection(name)
        os.makedirs(path, exist_ok=True)
        try:
            index = tantivy.Index(tantivy_schema(), path=path)
        except ValueError:
            if number != 0:
                raise                                  # never wipe a generation folder
            # Generation 0 may hold an index with an older schema: recreate it (as before spec 0008)
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            index = tantivy.Index(tantivy_schema(), path=path)
        return Generation(number, name, path, index)

    def drop_generation(self, number: int):
        """Deletes a generation's collection and folder; never the active one."""
        if number == self.active.number:
            return
        name, path = self.collection_for(number), self.tantivy_path_for(number)
        if name in [c.name for c in self.qdrant_client.get_collections().collections]:
            self.qdrant_client.delete_collection(collection_name=name)
        shutil.rmtree(path, ignore_errors=True)

    def _generations_on_disk(self) -> set:
        """Every generation number N >= 1 that has a collection or a keyword folder."""
        prefix = f"{self.base_collection}__g"
        numbers = {c.name[len(prefix):] for c in self.qdrant_client.get_collections().collections
                   if c.name.startswith(prefix)}
        folder, base = os.path.split(os.path.abspath(self.base_tantivy_path))
        if os.path.isdir(folder):
            numbers |= {entry[len(base) + 3:] for entry in os.listdir(folder) if entry.startswith(base + "__g")}
        return {int(n) for n in numbers if n.isdigit()}

    def build_generation(self, chunks: List[Dict], cancel: Optional[threading.Event] = None) -> Generation:
        """Builds the next generation from these chunks without touching the active one, and checks it.
        Any failure or cancel removes the partial build. The caller commits it (commit_generation)."""
        # Above every generation on disk, not just the active one: after a lost marker the active one is 0 while
        # the real index may be __g1, and a crash leftover is never reused (cleanup_orphans removes it)
        number = max({self.active.number} | self._generations_on_disk()) + 1
        try:
            generation = self._open_generation(number)
            self._fill(generation, chunks, cancel)
            self._check(generation, chunks)
            return generation
        except BaseException:
            self.drop_generation(number)
            raise

    def _fill(self, generation: Generation, chunks: List[Dict], cancel: Optional[threading.Event]):
        def check_cancel():
            if cancel is not None and cancel.is_set():
                raise BuildCancelled("index build cancelled")

        if chunks:
            embeddings = self.embedder.embed_batch([chunk["content"] for chunk in chunks])
            check_cancel()
            points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),   # stable across runs
                                         vector=embeddings[i],
                                         payload={"content": chunk["content"], "metadata": chunk["metadata"],
                                                  "chunk_id": chunk["id"]})
                      for i, chunk in enumerate(chunks)]
            for start in range(0, len(points), UPSERT_BATCH):
                check_cancel()
                self.qdrant_client.upsert(collection_name=generation.collection_name,
                                          points=points[start:start + UPSERT_BATCH])
        writer = generation.tantivy_index.writer()
        for chunk in chunks:
            doc = tantivy.Document()
            doc.add_text("chunk_id", chunk["id"])
            doc.add_text("content", chunk["content"])
            doc.add_text("source_doc", chunk["metadata"].get("source", "unknown"))
            writer.add_document(doc)
        writer.commit()
        generation.tantivy_index.reload()
        check_cancel()

    def _check(self, generation: Generation, chunks: List[Dict]):
        """Both stores hold the whole corpus: one point per unique chunk id, one keyword document per chunk."""
        unique = len({chunk["id"] for chunk in chunks})
        points = self.qdrant_client.count(collection_name=generation.collection_name, exact=True).count
        documents = generation.tantivy_index.searcher().num_docs
        if points != unique or documents != len(chunks):
            raise RuntimeError(f"generation {generation.number} is incomplete: {points} of {unique} vectors, "
                               f"{documents} of {len(chunks)} keyword documents")

    def commit_generation(self, generation: Generation, tables: str, corpus: List[str]) -> Generation:
        """Writes the marker naming the new generation (the commit point), then makes it the active one.
        Returns the generation it replaced, for the caller to retire."""
        self.mark_index_format(tables, generation=generation.number, corpus=corpus)
        with self._lock:
            previous, self._active = self._active, generation
        return previous

    def cleanup_orphans(self):
        """At startup: removes every generation the marker does not name. Does nothing when the marker could not
        be read when the index was opened, or cannot be read now, so a lost marker never costs a good index."""
        if not self.opened_with_marker or self.read_marker() is None:
            return
        keep = self.active.number
        prefix = f"{self.base_collection}__g"
        for collection in [c.name for c in self.qdrant_client.get_collections().collections]:
            number = 0 if collection == self.base_collection else (
                int(collection[len(prefix):]) if collection.startswith(prefix) and
                collection[len(prefix):].isdigit() else None)
            if number is not None and number != keep:
                self.drop_generation(number)
        folder, base = os.path.dirname(os.path.abspath(self.base_tantivy_path)), \
            os.path.basename(os.path.abspath(self.base_tantivy_path))
        for entry in os.listdir(folder):
            suffix = entry[len(base) + 3:] if entry.startswith(base + "__g") else None
            if suffix is not None and suffix.isdigit() and int(suffix) != keep:
                shutil.rmtree(os.path.join(folder, entry), ignore_errors=True)
        if keep != 0:
            shutil.rmtree(self.base_tantivy_path, ignore_errors=True)

    @property
    def marker_path(self) -> str:
        """The format marker sits beside both stores, so test indexes get their own."""
        return os.path.join(os.path.dirname(os.path.abspath(self.qdrant_path)), INDEX_MARKER)

    def read_marker(self) -> Optional[dict]:
        """The marker's contents, or None when it is missing or unreadable."""
        try:
            with open(self.marker_path, encoding="utf-8") as f:
                marker = json.load(f)
            return marker if isinstance(marker, dict) else None
        except (OSError, ValueError):
            return None

    def index_format(self) -> int:
        """Format of the index on disk; a missing or unreadable marker counts as the old format 1."""
        try:
            return int((self.read_marker() or {})["format"])
        except (KeyError, TypeError, ValueError):
            return 1

    def index_tables(self):
        """Table and figure fingerprint the index was built with; None when the marker predates it."""
        return (self.read_marker() or {}).get("tables")

    def index_corpus(self) -> Optional[List[str]]:
        """The corpus fingerprint the active generation was built from; None when the marker predates it."""
        return (self.read_marker() or {}).get("corpus")

    def mark_index_format(self, tables: str = "none", generation: int = None, corpus: List[str] = None):
        """Records that the index on disk is current: format, reader fingerprint, generation and corpus. Written to a
        temporary file and moved into place, so a crash never leaves half a marker."""
        marker = {"format": INDEX_FORMAT_VERSION, "tables": tables,
                  "generation": self.active.number if generation is None else generation}
        if corpus is not None:
            marker["corpus"] = corpus
        temporary = f"{self.marker_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(marker, f)
        os.replace(temporary, self.marker_path)

    def clear_index_format(self):
        """Dropped before any rebuild, so a crash partway never looks like a healthy index."""
        try:
            os.remove(self.marker_path)
        except FileNotFoundError:
            pass

    def reset_indexes(self):
        """Empties both stores of the active generation in place (tests and tools; the API builds generations)."""
        self.qdrant_client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        writer = self.tantivy_index.writer()
        writer.delete_all_documents()
        writer.commit()
        self.tantivy_index.reload()

    def ingest_chunks(self, chunks: List[Dict]):
        """Replaces the active generation's contents with these chunks, in place (tests and tools; the API builds
        a new generation instead, so questions never see a half built index)."""
        if not chunks:
            return
        embeddings = self.embedder.embed_batch([chunk["content"] for chunk in chunks])

        self.qdrant_client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),   # stable across runs
                                     vector=embeddings[i],
                                     payload={"content": chunk["content"], "metadata": chunk["metadata"],
                                              "chunk_id": chunk["id"]})
                  for i, chunk in enumerate(chunks)]
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        writer = self.tantivy_index.writer()
        writer.delete_all_documents()                 # Tantivy has no upsert
        for chunk in chunks:
            doc = tantivy.Document()
            doc.add_text("chunk_id", chunk["id"])
            doc.add_text("content", chunk["content"])
            doc.add_text("source_doc", chunk["metadata"].get("source", "unknown"))
            writer.add_document(doc)
        writer.commit()
        self.tantivy_index.reload()                   # so searchers see the new documents