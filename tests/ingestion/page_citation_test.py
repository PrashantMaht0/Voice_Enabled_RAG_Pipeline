"""Page level citations: PDFs cite a page, text files keep their chunk."""
import pytest

from src.api import routes
from src.ingestion.chunking import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.verification.citations import strip_ungrounded_citations
from src.vector_db.vector_store import INDEX_FORMAT_VERSION
from tests.helpers import FakeManager, use_corpus, write_pdf


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

def test_startup_rebuilds_an_old_index_once(tmp_path, monkeypatch):
    """A fresh Docker volume has no marker and no documents: rebuild once, then never again."""
    (tmp_path / "empty").mkdir()
    use_corpus(monkeypatch, tmp_path / "empty")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr(routes, "get_retriever", lambda: type("R", (), {"store": type("S", (), {"db_manager": manager})})())

    routes.ensure_index_format()
    routes.ensure_index_format()

    assert manager.ingested == [[]], "one build, of an empty corpus"
    assert manager.index_format() == INDEX_FORMAT_VERSION and manager.active.number == 1


def test_a_failed_build_leaves_the_previous_marker_and_generation(raw_dir, tmp_path, monkeypatch):
    """the previous generation keeps answering after a failed build."""
    use_corpus(monkeypatch, raw_dir)
    manager = FakeManager(tmp_path, fail=True)
    manager.mark_index_format("none")            # a healthy index from before

    assert routes.rebuild_index(manager) is False
    assert manager.index_format() == INDEX_FORMAT_VERSION and manager.active.number == 0
    assert manager.read_marker()["generation"] == 0
