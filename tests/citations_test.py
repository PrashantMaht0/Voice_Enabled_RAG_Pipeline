"""Citation clean up: an answer only cites sources that were retrieved for it."""
from src.verification.citations import strip_ungrounded_citations

CHUNKS = [
    "[Source: azure.pdf | Chunk: 12]\nBlobs are stored in containers.",
    "[Source: azure.pdf | Chunk: 13]\nA container holds many blobs.",
]


def test_valid_citation_survives():
    answer = "Blobs live in containers. [Source: azure.pdf | Chunk: 12]"

    cleaned, removed = strip_ungrounded_citations(answer, CHUNKS)

    assert removed == []
    assert cleaned == answer


def test_citation_to_unretrieved_chunk_is_removed():
    answer = "Blobs live in containers. [Source: azure.pdf | Chunk: 99]"

    cleaned, removed = strip_ungrounded_citations(answer, CHUNKS)

    assert removed == ["[Source: azure.pdf | Chunk: 99]"]
    assert "Chunk: 99" not in cleaned
    assert cleaned.startswith("Blobs live in containers.")


def test_citation_to_unretrieved_source_is_removed():
    answer = "Blobs live in containers. [Source: other.pdf | Chunk: 12]"

    _, removed = strip_ungrounded_citations(answer, CHUNKS)

    assert removed == ["[Source: other.pdf | Chunk: 12]"]
