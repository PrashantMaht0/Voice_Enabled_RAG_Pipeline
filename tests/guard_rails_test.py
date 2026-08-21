import pytest

from src.verification.guard_rails import HallucinationGuard


@pytest.fixture(scope="module")
def guard():
    g = HallucinationGuard()
    if g.nlp is None:
        pytest.skip("spacy model en_core_web_sm not installed")
    return g


CONTEXT = ["The router TX-9000 requires port 443 open. Error ERR_SSL_PROTOCOL means a bad certificate."]


def test_grounded_answer_passes(guard):
    assert guard.verify_assertions("Open port 443 on the TX-9000.", CONTEXT) is True


def test_invented_error_code_is_flagged(guard):
    assert guard.verify_assertions("You will see ERR_TOTALLY_MADE_UP instead.", CONTEXT) is False


def test_invented_part_number_is_flagged(guard):
    assert guard.verify_assertions("Use the TX-1234 router.", CONTEXT) is False


def test_answer_without_hard_claims_passes(guard):
    assert guard.verify_assertions("Please check your configuration.", CONTEXT) is True


# --- Citation grounding (pure regex: runs without the spacy model) ---

CHUNKS = [
    "[Source: azure.pdf | Chunk: 12]\nBlobs are stored in containers.",
    "[Source: azure.pdf | Chunk: 13]\nA container holds many blobs.",
]


@pytest.fixture(scope="module")
def bare_guard():
    return HallucinationGuard()


def test_valid_citation_survives(bare_guard):
    answer = "Blobs live in containers. [Source: azure.pdf | Chunk: 12]"

    cleaned, removed = bare_guard.strip_ungrounded_citations(answer, CHUNKS)

    assert removed == []
    assert cleaned == answer


def test_citation_to_unretrieved_chunk_is_removed(bare_guard):
    answer = "Blobs live in containers. [Source: azure.pdf | Chunk: 99]"

    cleaned, removed = bare_guard.strip_ungrounded_citations(answer, CHUNKS)

    assert removed == ["[Source: azure.pdf | Chunk: 99]"]
    assert "Chunk: 99" not in cleaned
    assert cleaned.startswith("Blobs live in containers.")


def test_citation_to_unretrieved_source_is_removed(bare_guard):
    answer = "Blobs live in containers. [Source: other.pdf | Chunk: 12]"

    _, removed = bare_guard.strip_ungrounded_citations(answer, CHUNKS)

    assert removed == ["[Source: other.pdf | Chunk: 12]"]
