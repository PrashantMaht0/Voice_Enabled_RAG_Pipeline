"""Helpers shared by the ingestion, table and figure tests: a tiny real PDF writer and a fake index manager."""
import threading

from src.api import routes
from src.ingestion.loader import DocumentLoader
from src.vector_db.vector_store import Generation, VectorDBManager


def write_pdf(path, page_texts):
    """A minimal real PDF, one Helvetica text line per page ("" makes a blank page)."""
    count = len(page_texts)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{4 + 2 * i} 0 R" for i in range(count)), count),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


class FakeManager:
    """An index manager that records builds and borrows the real marker and generation commit logic."""
    marker_path = VectorDBManager.marker_path
    read_marker = VectorDBManager.read_marker
    index_format = VectorDBManager.index_format
    index_tables = VectorDBManager.index_tables
    index_corpus = VectorDBManager.index_corpus
    mark_index_format = VectorDBManager.mark_index_format
    commit_generation = VectorDBManager.commit_generation
    active = VectorDBManager.active

    def __init__(self, root, fail=False):
        self.qdrant_path = str(root / "qdrant")
        self.fail = fail
        self.ingested, self.dropped = [], []
        self._lock = threading.Lock()
        self._active = Generation(0, "fake", str(root / "tantivy"), None)

    def build_generation(self, chunks, cancel=None):
        if self.fail:
            raise RuntimeError("disk full")
        self.ingested.append(chunks)
        return Generation(self._active.number + 1, "fake", "fake", None)

    def drop_generation(self, number):
        self.dropped.append(number)


def use_corpus(monkeypatch, folder):
    """Points the API's document loader at this folder."""
    monkeypatch.setattr(routes, "DocumentLoader", lambda: DocumentLoader(raw_data_dir=str(folder)))
