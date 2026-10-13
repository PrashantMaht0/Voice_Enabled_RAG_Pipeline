"""The index on disk: an embedded Qdrant collection (vectors) and a Tantivy index (keywords), built together."""
import json
import os
import shutil
import uuid
from typing import Dict, List

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


class VectorDBManager:
    """Owns both stores: opening them, rebuilding them from chunks, and the format marker beside them.

    Embedded Qdrant allows one process per folder, so only the API (or a test with its own folder) opens one."""

    def __init__(self, qdrant_path: str = None, tantivy_path: str = None, collection_name: str = None):
        self.qdrant_path = pick(qdrant_path, "paths.qdrant_storage", "./vectordb/qdrant_storage")
        self.tantivy_path = pick(tantivy_path, "paths.tantivy_index", "./vectordb/tantivy_index")
        self.collection_name = pick(collection_name, "paths.collection_name", "vocal_rag_docs")
        
        os.makedirs(self.qdrant_path, exist_ok=True)
        os.makedirs(self.tantivy_path, exist_ok=True)
        self.embedder = DenseEmbedder()
        self.qdrant_client = QdrantClient(path=self.qdrant_path)
        self._ensure_collection()
        self.tantivy_index = self._open_tantivy()

    def _ensure_collection(self):
        """Creates the collection if missing, reusing the existing client."""
        collections = [col.name for col in self.qdrant_client.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.get_dimension(),
                    distance=models.Distance.COSINE,   # the embeddings are normalized
                )
            )

    def _open_tantivy(self):
        """The BM25 keyword index (chunk id, content and source, all stored), created when missing and
        recreated when the folder holds an index with another schema."""
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("chunk_id", stored=True)
        schema_builder.add_text_field("content", stored=True)
        schema_builder.add_text_field("source_doc", stored=True)
        schema = schema_builder.build()
        try:
            return tantivy.Index(schema, path=self.tantivy_path)
        except ValueError:
            shutil.rmtree(self.tantivy_path)
            os.makedirs(self.tantivy_path, exist_ok=True)
            return tantivy.Index(schema, path=self.tantivy_path)

    @property
    def marker_path(self) -> str:
        """The format marker sits beside both stores, so test indexes get their own."""
        return os.path.join(os.path.dirname(os.path.abspath(self.qdrant_path)), INDEX_MARKER)

    def index_format(self) -> int:
        """Format of the index on disk; a missing or unreadable marker counts as the old format 1."""
        try:
            with open(self.marker_path, encoding="utf-8") as f:
                return int(json.load(f)["format"])
        except (OSError, ValueError, KeyError, TypeError):
            return 1

    def index_tables(self):
        """Table and figure fingerprint the index was built with; None when the marker predates it."""
        try:
            with open(self.marker_path, encoding="utf-8") as f:
                return json.load(f).get("tables")
        except (OSError, ValueError, AttributeError):
            return None

    def mark_index_format(self, tables: str = "none"):
        """Records that the index on disk is current: the format version and the reader fingerprint."""
        with open(self.marker_path, "w", encoding="utf-8") as f:
            json.dump({"format": INDEX_FORMAT_VERSION, "tables": tables}, f)

    def clear_index_format(self):
        """Dropped before any rebuild, so a crash partway never looks like a healthy index."""
        try:
            os.remove(self.marker_path)
        except FileNotFoundError:
            pass

    def reset_indexes(self):
        """Empties both stores. Used when the corpus drops to zero documents."""
        self.qdrant_client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        writer = self.tantivy_index.writer()
        writer.delete_all_documents()
        writer.commit()
        self.tantivy_index.reload()

    def ingest_chunks(self, chunks: List[Dict]):
        """Replaces both stores' contents with these chunks (the whole corpus is always ingested at once, so
        starting empty means a removed document leaves nothing behind)."""
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