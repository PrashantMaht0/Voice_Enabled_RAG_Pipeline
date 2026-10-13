"""Page level citations (spec 0001): PDFs cite a page, text files keep their chunk."""
import pytest

from src.api import routes
from src.ingestion.chunking import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.verification.citations import strip_ungrounded_citations
from src.vector_db.vector_store import INDEX_FORMAT_VERSION, VectorDBManager


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


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_pdf(raw / "guide.pdf", ["Blob storage holds files.", "", "Queues carry messages."])
    (raw / "notes.md").write_text("Tables hold rows.")
    return raw


# --- loading and chunking ---

def test_pdf_pages_keep_their_numbers_and_blank_pages_are_skipped(raw_dir):
    docs = {d["metadata"]["source"]: d for d in DocumentLoader(raw_data_dir=str(raw_dir)).load_documents()}

    assert docs["guide.pdf"]["pages"] == [(1, "Blob storage holds files."), (3, "Queues carry messages.")]
    assert "pages" not in docs["notes.md"]


def test_pdf_chunks_cite_their_page_and_text_chunks_keep_chunk(raw_dir):
    docs = DocumentLoader(raw_data_dir=str(raw_dir)).load_documents()
    chunks = {c["content"].splitlines()[0]: c for c in DocumentChunker().chunk_documents(docs)}

    assert set(chunks) == {
        "[Source: guide.pdf | Page: 1]",
        "[Source: guide.pdf | Page: 3]",
        "[Source: notes.md | Chunk: 0]",
    }
    assert chunks["[Source: guide.pdf | Page: 3]"]["metadata"]["page"] == 3
    assert "page" not in chunks["[Source: notes.md | Chunk: 0]"]["metadata"]


def test_no_chunk_spans_a_page_break():
    chunker = DocumentChunker(chunk_size=20, overlap=5)
    long_page = " ".join(f"alpha{i}" for i in range(60))
    doc = {"content": "", "pages": [(1, long_page), (2, "Short second page.")],
           "metadata": {"source": "a.pdf", "doc_hash": "h"}}

    chunks = chunker.chunk_documents([doc])

    assert len([c for c in chunks if c["metadata"]["page"] == 1]) > 1
    # A page shorter than one window gives exactly one chunk (the window used to step past the end)
    assert [c["content"] for c in chunks if c["metadata"]["page"] == 2] == \
        ["[Source: a.pdf | Page: 2]\nShort second page."]
    # chunk_index keeps counting across pages, so it stays unique within the document
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


# --- citations in the API and the grounding check ---

CHUNKS = [
    "[Source: a.pdf | Page: 3]\nBlobs hold files.",
    "[Source: a.pdf | Page: 3]\nMore about blobs.",
    "[Source: b.pdf | Page: 3]\nQueues carry messages.",
    "[Source: notes.md | Chunk: 2]\nTables hold rows.",
]


def test_citations_name_the_page_for_pdfs_and_the_chunk_for_text():
    assert routes.build_citations(CHUNKS) == [
        {"source": "a.pdf", "page": 3, "chunk": None},
        {"source": "b.pdf", "page": 3, "chunk": None},
        {"source": "notes.md", "page": None, "chunk": 2},
    ]


def test_the_same_page_in_another_file_is_still_appended():
    answer = "Blobs hold files. [Source: a.pdf | Page: 3]"

    completed = routes.append_citations(answer, routes.build_citations(CHUNKS))

    assert completed.count("[Source: a.pdf | Page: 3]") == 1
    assert "[Source: b.pdf | Page: 3]" in completed
    assert "[Source: notes.md | Chunk: 2]" in completed


def test_grounding_check_drops_pages_that_were_not_retrieved():
    answer = ("Blobs hold files. [Source: a.pdf | Page: 3] "
              "Blobs are cheap. [Source: a.pdf | Page: 7] "
              "Tables hold rows. [Source: notes.md | Page: 2]")

    cleaned, removed = strip_ungrounded_citations(answer, CHUNKS)

    assert removed == ["[Source: a.pdf | Page: 7]", "[Source: notes.md | Page: 2]"]
    assert "[Source: a.pdf | Page: 3]" in cleaned


# --- index format marker and rebuild ---

class _FakeManager:
    """Records ingest calls; borrows the real marker logic."""
    marker_path = VectorDBManager.marker_path
    index_format = VectorDBManager.index_format
    index_tables = VectorDBManager.index_tables
    mark_index_format = VectorDBManager.mark_index_format
    clear_index_format = VectorDBManager.clear_index_format

    def __init__(self, root, fail=False):
        self.qdrant_path = str(root / "qdrant")
        self.fail = fail
        self.ingested, self.resets = [], 0

    def ingest_chunks(self, chunks):
        if self.fail:
            raise RuntimeError("disk full")
        self.ingested.append(chunks)

    def reset_indexes(self):
        self.resets += 1


def use_corpus(monkeypatch, folder):
    monkeypatch.setattr(routes, "DocumentLoader", lambda: DocumentLoader(raw_data_dir=str(folder)))


def test_startup_rebuilds_an_old_index_once(tmp_path, monkeypatch):
    """A fresh Docker volume has no marker and no documents: rebuild once, then never again."""
    (tmp_path / "empty").mkdir()
    use_corpus(monkeypatch, tmp_path / "empty")
    manager = _FakeManager(tmp_path)
    monkeypatch.setattr(routes, "get_retriever", lambda: type("R", (), {"store": type("S", (), {"db_manager": manager})})())

    routes.ensure_index_format()
    routes.ensure_index_format()

    assert manager.resets == 1
    assert manager.index_format() == INDEX_FORMAT_VERSION


def test_failed_rebuild_leaves_no_marker(raw_dir, tmp_path, monkeypatch):
    use_corpus(monkeypatch, raw_dir)
    manager = _FakeManager(tmp_path, fail=True)
    manager.mark_index_format()                  # a healthy index from before

    assert routes.rebuild_index(manager) is False
    assert manager.index_format() == 1, "The next startup must try again"
