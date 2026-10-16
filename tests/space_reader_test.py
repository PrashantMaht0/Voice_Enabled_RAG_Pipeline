"""The live demo's PDF reading Space, without Gradio or a GPU (spec 0009, build task 7): checks, limits, the daily
cap per address, the client address, and that nothing of an upload is kept."""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "space"))
import reader                                            # noqa: E402
from src.ingestion.upload_checks import UploadRefused    # noqa: E402


def pdf_bytes(pages: int) -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.fixture
def upload(tmp_path):
    def make(name: str, content: bytes):
        path = tmp_path / "gradio-cache" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        return str(path)
    return make


@pytest.mark.parametrize("chain, peer, expected", [
    ("81.2.69.160", "10.0.0.1", "81.2.69.160"),
    ("1.2.3.4, 81.2.69.160", "10.0.0.1", "81.2.69.160"),          # the leftmost hop is whatever the client claimed
    ("81.2.69.160, 10.16.0.5", "10.0.0.1", "81.2.69.160"),         # internal proxies after the visitor are skipped
    ("not-an-ip, 10.1.1.1", "81.2.69.161", "81.2.69.161"),
    (None, None, "unknown"),
])
def test_the_client_address_is_the_rightmost_public_hop(chain, peer, expected):
    """covers: AC-14"""
    assert reader.client_ip(chain, peer) == expected


def test_the_daily_cap_counts_successful_reads_per_address_and_resets_after_a_day(monkeypatch):
    """covers: AC-14"""
    monkeypatch.setenv("DAILY_READS_PER_IP", "2")
    cap = reader.DailyCap()
    cap.record("a", now=0)
    cap.record("a", now=10)

    assert not cap.allowed("a", now=20)
    assert cap.allowed("b", now=20)
    assert cap.allowed("a", now=reader.DAY_S + 1)


@pytest.mark.parametrize("name, content, code", [
    ("x.exe", b"MZ", "bad_type"),
    (".hidden.md", b"# hi", "bad_name"),
    ("fake.pdf", b"not a pdf", "not_pdf"),
    ("latin.txt", "café".encode("latin-1"), "not_utf8"),
])
def test_refused_uploads_say_why_and_leave_nothing_behind(upload, name, content, code):
    """covers: AC-7, AC-14"""
    with pytest.raises(UploadRefused) as refused:
        reader.prepare(upload(name, content), name)

    assert refused.value.code == code


def test_page_and_size_limits_come_from_the_space_variables(upload, monkeypatch):
    """covers: AC-14"""
    monkeypatch.setenv("MAX_PAGES", "2")
    monkeypatch.setenv("MAX_MB", "1")

    with pytest.raises(UploadRefused) as pages:
        reader.prepare(upload("long.pdf", pdf_bytes(3)), "long.pdf")
    with pytest.raises(UploadRefused) as size:
        reader.prepare(upload("big.txt", b"x" * (1024 * 1024 + 1)), "big.txt")

    assert (pages.value.code, pages.value.status) == ("too_many_pages", 413)
    assert (size.value.code, size.value.status) == ("too_large", 413)


def test_a_text_upload_is_read_into_chunk_records(upload, monkeypatch):
    """covers: AC-7"""
    folder, pages = reader.prepare(upload("notes.md", b"# Notes\n\nThe pump runs at 40 litres a minute.\n"), "notes.md")
    try:
        chunks = reader.read(folder)
    finally:
        reader.discard(folder)

    assert pages is None
    assert set(chunks[0]) == {"id", "doc_hash", "chunk_index", "page", "kind", "content"}
    assert chunks[0]["content"].startswith("[Source: notes.md |") and "40 litres" in chunks[0]["content"]


def test_too_many_chunks_is_refused(upload, monkeypatch):
    """covers: AC-14"""
    monkeypatch.setenv("MAX_CHUNKS", "1")
    folder, _ = reader.prepare(upload("long.md", ("word " * 3000).encode()), "long.md")
    try:
        with pytest.raises(UploadRefused) as refused:
            reader.read(folder)
    finally:
        reader.discard(folder)

    assert refused.value.code == "too_many_chunks"


def test_discard_removes_the_private_copy_and_the_gradio_upload(upload):
    """covers: AC-9"""
    source = upload("notes.md", b"# Notes\n")
    folder, _ = reader.prepare(source, "notes.md")

    reader.discard(folder, source, None)

    assert not os.path.exists(folder) and not os.path.exists(source)
