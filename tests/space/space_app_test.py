"""The PDF reader Space's request handler, with Gradio, ZeroGPU and the models replaced by fakes: the daily cap,
error mapping, the log line and that the upload is always discarded."""
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

SPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "space")
sys.path.insert(0, SPACE)


class FakeError(Exception):
    """Stands in for gr.Error, the message Gradio shows the visitor."""


def fake_gradio():
    """Just enough of Gradio for space/app.py to build its page at import time."""
    class Blocks:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def queue(self, *args, **kwargs):
            return self

    file = SimpleNamespace(upload=lambda *args, **kwargs: None)
    return types.SimpleNamespace(Blocks=Blocks, Markdown=lambda *a, **k: None, File=lambda *a, **k: file,
                                 JSON=lambda *a, **k: None, Error=FakeError, Request=object)


@pytest.fixture
def space(monkeypatch, tmp_path):
    from src.ingestion import figures, tables
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio())
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace(GPU=lambda **kwargs: (lambda fn: fn)))
    monkeypatch.setenv("VRAG_CONFIG", os.path.join(SPACE, "..", "config", "config.yaml"))
    loader = SimpleNamespace(load=lambda: None)
    monkeypatch.setattr(tables, "check_table_parser", lambda: None)
    monkeypatch.setattr(tables, "get_detector", lambda: loader)
    monkeypatch.setattr(tables, "get_layout_model", lambda: loader)
    monkeypatch.setattr(figures, "check_figure_parser", lambda: None)
    spec = importlib.util.spec_from_file_location("space_app", os.path.join(SPACE, "app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    discarded = []
    monkeypatch.setattr(app.reader, "prepare", lambda file, name: (str(tmp_path / "folder"), 2))
    monkeypatch.setattr(app.reader, "discard", lambda *paths: discarded.append(paths))
    monkeypatch.setattr(app, "read_pdf", lambda folder, pages: [{"id": "a_0"}])
    app.discarded = discarded
    return app


def visitor(ip="81.2.69.160"):
    return SimpleNamespace(headers={"x-forwarded-for": ip}, client=SimpleNamespace(host="10.0.0.1"))


def test_a_read_returns_the_chunks_counts_against_the_cap_and_discards_the_upload(space, capsys):
    reply = space.parse_document("/tmp/gradio/Private Report.pdf", visitor())

    assert reply["chunks"] == [{"id": "a_0"}] and reply["pages"] == 2
    assert len(space.discarded) == 1 and space.discarded[0][1] == "/tmp/gradio/Private Report.pdf"
    log = capsys.readouterr().out
    assert "Private Report" not in log and space.reader.log_name("Private Report.pdf") in log


def test_an_address_over_its_daily_cap_is_refused_and_the_upload_still_discarded(space, monkeypatch):
    monkeypatch.setattr(space.cap, "allowed", lambda ip: False)

    with pytest.raises(FakeError, match="daily_cap"):
        space.parse_document("/tmp/gradio/a.pdf", visitor())

    assert space.discarded == [(None, "/tmp/gradio/a.pdf")]


def test_an_unexpected_failure_still_discards_the_upload(space, monkeypatch):
    monkeypatch.setattr(space, "read_pdf", lambda folder, pages: (_ for _ in ()).throw(RuntimeError("GPU lost")))

    with pytest.raises(RuntimeError):
        space.parse_document("/tmp/gradio/a.pdf", visitor())

    assert len(space.discarded) == 1 and space.discarded[0][1] == "/tmp/gradio/a.pdf"


def test_a_failed_read_does_not_use_up_the_visitor_s_daily_reads(space, monkeypatch):
    recorded = []
    monkeypatch.setattr(space.cap, "record", recorded.append)
    monkeypatch.setattr(space, "read_pdf", lambda folder, pages: (_ for _ in ()).throw(RuntimeError("GPU lost")))

    with pytest.raises(RuntimeError):
        space.parse_document("/tmp/gradio/a.pdf", visitor())

    assert recorded == []
