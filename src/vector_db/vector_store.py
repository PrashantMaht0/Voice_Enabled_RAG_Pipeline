import json
import os
import uuid
from typing import List, Dict
from src.embeddings.embedder import DenseEmbedder
from src.utils.config import pick

# Qdrant Imports
try:
    from qdrant_client import QdrantClient, models
    QDRANT_INSTALLED = True
except ImportError:
    QDRANT_INSTALLED = False

# Tantivy Imports
try:
    import tantivy
    TANTIVY_INSTALLED = True
except ImportError:
    TANTIVY_INSTALLED = False

# Bump when the chunk layout changes, so an older index is rebuilt at startup.
# 2: PDFs are chunked page by page and their headers cite "Page: N".
# 3: chunk ids come from a SHA256 content fingerprint, so they are the same on every run.
INDEX_FORMAT_VERSION = 3
INDEX_MARKER = "index_version.json"


class VectorDBManager:
    """Owns the Qdrant and Tantivy index lifecycle."""
    def __init__(
        self, 
        qdrant_path: str = None,
        tantivy_path: str = None,
        collection_name: str = None
    ):
        self.qdrant_path = pick(qdrant_path, "paths.qdrant_storage", "./vectordb/qdrant_storage")
        self.tantivy_path = pick(tantivy_path, "paths.tantivy_index", "./vectordb/tantivy_index")
        self.collection_name = pick(collection_name, "paths.collection_name", "vocal_rag_docs")
        
        os.makedirs(self.qdrant_path, exist_ok=True)
        os.makedirs(self.tantivy_path, exist_ok=True)
        
        self.embedder = DenseEmbedder()
        self.qdrant_client = None
        self.tantivy_index = None
        
        self._init_qdrant()
        self._init_tantivy()

    def _init_qdrant(self):
        """Initializes the Qdrant client and HNSW collection."""
        if not QDRANT_INSTALLED:
            print("WARNING: Qdrant not installed. Running in mock mode.")
            return
            
        self.qdrant_client = QdrantClient(path=self.qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if missing, reusing the existing client."""
        collections = [col.name for col in self.qdrant_client.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.get_dimension(),
                    distance=models.Distance.COSINE # Requires L2 Normalized embeddings
                )
            )

    def _init_tantivy(self):
        """Initializes the Tantivy schema and index for exact BM25 sparse matching."""
        if not TANTIVY_INSTALLED:
            print("WARNING: Tantivy not installed. Running in mock mode.")
            return
            
        # Define Schema: Text fields must be indexed (TEXT) and optionally saved (STORED).
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("chunk_id", stored=True)
        schema_builder.add_text_field("content", stored=True)
        schema_builder.add_text_field("source_doc", stored=True)
        schema = schema_builder.build()
        
        # Open existing index or create a new one in the directory
        try:
            self.tantivy_index = tantivy.Index(schema, path=self.tantivy_path)
        except ValueError:
            # Recreate if schema mismatch or empty directory
            import shutil
            shutil.rmtree(self.tantivy_path)
            os.makedirs(self.tantivy_path, exist_ok=True)
            self.tantivy_index = tantivy.Index(schema, path=self.tantivy_path)

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

    def mark_index_format(self):
        with open(self.marker_path, "w", encoding="utf-8") as f:
            json.dump({"format": INDEX_FORMAT_VERSION}, f)

    def clear_index_format(self):
        """Dropped before any rebuild, so a crash partway never looks like a healthy index."""
        try:
            os.remove(self.marker_path)
        except FileNotFoundError:
            pass

    def reset_indexes(self):
        """Empties both stores. Used when the corpus drops to zero documents."""
        if self.qdrant_client:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()

        if self.tantivy_index:
            writer = self.tantivy_index.writer()
            writer.delete_all_documents()
            writer.commit()
            self.tantivy_index.reload()

    def ingest_chunks(self, chunks: List[Dict]):
        """Indexes chunks into both the dense and sparse stores."""
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 1. Upload to Qdrant
        if self.qdrant_client:
            # Chunk ids are positional, so a re-chunked corpus leaves orphans behind.
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()

            qdrant_points = []
            for i, chunk in enumerate(chunks):
                qdrant_points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),   # stable across runs
                        vector=embeddings[i],
                        payload={
                            "content": chunk["content"],
                            "metadata": chunk["metadata"],
                            "chunk_id": chunk["id"]
                        }
                    )
                )
            # Batch upsert points
            self.qdrant_client.upsert(
                collection_name=self.collection_name, 
                points=qdrant_points
            )

        # 2. Upload to Tantivy
        if self.tantivy_index:
            writer = self.tantivy_index.writer()
            # Tantivy has no upsert, so re-adding would duplicate every document.
            writer.delete_all_documents()
            for chunk in chunks:
                doc = tantivy.Document()
                doc.add_text("chunk_id", chunk["id"])
                doc.add_text("content", chunk["content"])
                doc.add_text("source_doc", chunk["metadata"].get("source", "unknown"))
                writer.add_document(doc)
                
            writer.commit() # Save index to disk
            # Reload to ensure the searcher sees the newly committed documents
            self.tantivy_index.reload()