from src.ingestion.chunking import DocumentChunker
from src.ingestion.loader import DocumentLoader


def test_chunks_respect_size_and_overlap():
    chunker = DocumentChunker(chunk_size=100, overlap=20)
    text = " ".join(f"word{i}" for i in range(600))

    chunks = chunker.split_text(text)

    assert len(chunks) > 1, "Long text should split into several chunks."
    for chunk in chunks:
        assert len(chunker.tokenizer.encode(chunk)) <= 100

    # The window advances by size - overlap, so consecutive chunks must share tokens
    first_tail = chunker.tokenizer.encode(chunks[0])[-20:]
    second_head = chunker.tokenizer.encode(chunks[1])[:20]
    assert first_tail == second_head


def test_chunk_metadata_carries_citation_header():
    """The [Source: ... | Chunk: n] header is what the LLM cites and the UI parses."""
    documents = [{
        "content": "Azure Storage stores blobs.",
        "metadata": {"source": "azure.pdf", "doc_hash": "abc123"}
    }]

    chunks = DocumentChunker().chunk_documents(documents)

    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("[Source: azure.pdf | Chunk: 0]")
    assert chunks[0]["metadata"]["source"] == "azure.pdf"
    assert chunks[0]["metadata"]["doc_hash"] == "abc123"


def test_normalization_makes_equivalent_unicode_identical():
    """Composed and decomposed accents must collapse, or BM25 silently misses terms."""
    loader = DocumentLoader()

    composed = loader._normalize_text("caf\u00e9")      # e-acute as a single codepoint
    decomposed = loader._normalize_text("cafe\u0301")   # e + combining acute accent

    assert composed == decomposed


def test_normalization_strips_zero_width_and_collapses_spaces():
    loader = DocumentLoader()

    assert loader._normalize_text("ERR\u200b_SSL") == "ERR_SSL"
    assert loader._normalize_text("too    many   spaces") == "too many spaces"
