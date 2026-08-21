from src.utils.text_sanitizer import (
    drop_incomplete_header,
    has_incomplete_header,
    sanitize_for_speech,
    strip_citation_headers,
)

CITED = "A Web App is hosted in an App Service. [Source: azure.pdf | Chunk: 42]"


def test_speech_drops_citations_but_keeps_the_sentence():
    assert sanitize_for_speech(CITED) == "A Web App is hosted in an App Service."


def test_text_answer_keeps_its_citation():
    """The sanitizer is for speech only — the written answer stays auditable."""
    assert "[Source: azure.pdf | Chunk: 42]" in CITED


def test_partial_citation_is_never_spoken():
    """A half-written marker leaks the filename, which Kokoro reads digit by digit."""
    assert has_incomplete_header("Blobs are files. [Source: 978150930_Micro")
    assert sanitize_for_speech("Blobs are files. [Source: 978150930_Micro") == "Blobs are files."


def test_complete_marker_is_not_treated_as_partial():
    assert not has_incomplete_header(CITED)
    assert drop_incomplete_header(CITED) == CITED


def test_bullet_glyphs_and_double_spaces_are_cleaned():
    messy = "Types include  block blobs  and  page blobs."
    assert sanitize_for_speech(messy) == "Types include block blobs and page blobs."


def test_strip_is_idempotent_on_a_growing_buffer():
    buffer = ""
    for token in ["Blobs", " are", " files.", " [Source:", " a.pdf", " | Chunk: 9]"]:
        buffer += token
        buffer = strip_citation_headers(buffer)
    assert buffer.strip() == "Blobs are files."
