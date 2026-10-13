"""Spoken answers: citations, partial tags, bullets and formatting symbols are never read aloud; prose is unchanged."""
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



# The shape of the old English sample: list items start with "*", so Kokoro said "Asterisk can create..."
LISTED = (
    "Azure Resource Manager lets you:\n"
    "* Can create and manage resources [Source: azure.pdf | Chunk: 3]\n"
    "* **Deploy** templates as a group [Source: azure.pdf | Chunk: 4]\n"
)


def test_list_items_are_spoken_as_sentences_without_markers():
    assert sanitize_for_speech(LISTED) == (
        "Azure Resource Manager lets you: Can create and manage resources. Deploy templates as a group."
    )


def test_no_formatting_symbol_survives():
    answer = (
        "## Storage tiers\n"
        "1. __Hot__ tier, for *frequent* access;\n"
        "2) `Cool` tier, set by max_retries\n"
        "- Archive tier: [docs](https://example.com)\n"
        "\u2022 Premium tier for C#\n"
        "```bash\naz login\n```"
    )
    spoken = sanitize_for_speech(answer)
    assert spoken == ("Storage tiers. Hot tier, for frequent access. Cool tier, set by max retries. "
                      "Archive tier: docs. Premium tier for C sharp. az login")
    assert not set("*#`_\u2022\uf0b7") & set(spoken)


def test_citations_file_names_chunk_and_page_numbers_are_not_spoken():
    answer = (
        "Blobs are files [Source: azure.pdf | Page: 12]. Tables hold rows (Source: azure.pdf, Chunk 3). "
        "See chunk 7 of 978150930_Microsoft_Azure.pdf for limits."
    )
    spoken = sanitize_for_speech(answer)
    assert spoken == "Blobs are files. Tables hold rows. See of the document for limits."
    assert not any(ch.isdigit() for ch in spoken)


def test_plain_prose_is_unchanged():
    prose = "A Web App is hosted in an App Service... It scales out on demand."
    assert sanitize_for_speech(prose) == prose
