"""Startup cleanup of index generations: a lost marker never costs a good index."""
import threading
from types import SimpleNamespace

import pytest

from src.vector_db.vector_store import Generation, VectorDBManager


class FakeQdrant:
    """Just the collection listing and deletion that cleanup uses."""

    def __init__(self, names):
        self.names = set(names)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in sorted(self.names)])

    def delete_collection(self, collection_name):
        self.names.discard(collection_name)


def manager(tmp_path, active, collections, opened_with_marker):
    """A manager over real folders in tmp_path and a fake Qdrant, with `active` as its generation."""
    m = VectorDBManager.__new__(VectorDBManager)
    m.qdrant_path = str(tmp_path / "qdrant")
    m.base_tantivy_path = str(tmp_path / "tantivy_index")
    m.base_collection = "docs"
    m.qdrant_client = FakeQdrant(collections)
    m._lock = threading.Lock()
    m._active = Generation(active, m.collection_for(active), m.tantivy_path_for(active), None)
    m.opened_with_marker = opened_with_marker
    for name in collections:
        number = 0 if name == "docs" else int(name.split("__g")[1])
        (tmp_path / ("tantivy_index" if number == 0 else f"tantivy_index__g{number}")).mkdir()
    return m


def test_a_marker_written_by_the_startup_rebuild_does_not_unlock_cleanup(tmp_path):
    """The marker was missing when the index opened; the startup rebuild then wrote one. The old generation
    must survive this startup."""
    m = manager(tmp_path, active=1, collections=["docs", "docs__g1", "docs__g2"], opened_with_marker=False)
    m.mark_index_format("none", generation=1, corpus=[])
    m.cleanup_orphans()
    assert m.qdrant_client.names == {"docs", "docs__g1", "docs__g2"}
    assert (tmp_path / "tantivy_index__g2").exists()


def test_a_marker_present_at_open_lets_cleanup_remove_other_generations(tmp_path):
    m = manager(tmp_path, active=2, collections=["docs", "docs__g1", "docs__g2"], opened_with_marker=True)
    m.mark_index_format("none", generation=2, corpus=[])
    m.cleanup_orphans()
    assert m.qdrant_client.names == {"docs__g2"}
    assert not (tmp_path / "tantivy_index__g1").exists() and not (tmp_path / "tantivy_index").exists()
    assert (tmp_path / "tantivy_index__g2").exists()


def test_a_marker_lost_while_running_still_stops_cleanup(tmp_path):
    m = manager(tmp_path, active=2, collections=["docs__g1", "docs__g2"], opened_with_marker=True)
    m.cleanup_orphans()                                  # no marker on disk now
    assert m.qdrant_client.names == {"docs__g1", "docs__g2"}


def test_a_startup_rebuild_after_a_lost_marker_never_overwrites_a_generation_on_disk(tmp_path):
    """The marker is gone, so the API opened generation 0, but the real index is __g1. The rebuild must build
    a fresh number: reusing 1 would delete the good index first, and a failed build would leave nothing."""
    m = manager(tmp_path, active=0, collections=["docs", "docs__g1"], opened_with_marker=False)
    built = []

    def fail(number):
        built.append(number)
        raise RuntimeError("embedding failed")

    m._open_generation = fail
    try:
        m.build_generation([])
    except RuntimeError:
        pass
    assert built == [2]
    assert m.qdrant_client.names == {"docs", "docs__g1"}
    assert (tmp_path / "tantivy_index__g1").exists()


# --- startup drift check and the corpus fingerprint ---

from src.api import routes


class _StartupManager:
    """What start_index reads from the index: the marker's corpus, and cleanup."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.cleaned = False

    def index_corpus(self):
        return self.corpus

    def cleanup_orphans(self):
        self.cleaned = True


@pytest.fixture
def startup(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("alpha")
    monkeypatch.setattr(routes, "RAW_DIR", str(raw))
    requests = []
    worker = SimpleNamespace(request_rebuild=requests.append)
    monkeypatch.setattr(routes, "get_index_worker", lambda: worker)
    return raw, requests


def test_a_document_added_while_the_api_was_off_queues_a_rebuild_without_waiting(startup):
    raw, requests = startup
    manager = _StartupManager(routes.corpus_files(str(raw)))
    (raw / "b.md").write_text("beta")

    routes.start_index(manager)

    assert manager.cleaned
    assert requests == ["documents changed while the API was off"]


def test_an_unchanged_corpus_queues_nothing(startup):
    raw, requests = startup

    routes.start_index(_StartupManager(routes.corpus_files(str(raw))))

    assert requests == []


def test_an_old_marker_without_a_corpus_counts_as_matching(startup):
    raw, requests = startup
    (raw / "b.md").write_text("beta")

    routes.start_index(_StartupManager(None))

    assert requests == []


def test_the_corpus_fingerprint_is_name_size_and_mtime_of_documents_only(tmp_path):
    """Files in .incoming, other file types and folders never count."""
    (tmp_path / "b.txt").write_text("beta!")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-")
    (tmp_path / "notes.docx").write_text("ignored")
    (tmp_path / routes.INCOMING).mkdir()
    (tmp_path / routes.INCOMING / "x.part").write_text("half an upload")
    mtime = (tmp_path / "a.pdf").stat().st_mtime_ns

    entries = routes.corpus_files(str(tmp_path))

    assert [e.rsplit(":", 2)[0] for e in entries] == ["a.pdf", "b.txt"]
    assert entries[0] == f"a.pdf:5:{mtime}"


def test_the_corpus_fingerprint_of_a_missing_folder_is_empty(tmp_path):
    assert routes.corpus_files(str(tmp_path / "missing")) == []


# --- marker and build behavior ---

def test_the_marker_is_written_whole_and_names_the_generation_and_corpus(tmp_path):
    m = manager(tmp_path, active=1, collections=["docs__g1"], opened_with_marker=True)

    m.mark_index_format("tables-v1", generation=2, corpus=["a.md:5:1"])

    assert m.read_marker()["generation"] == 2 and m.read_marker()["corpus"] == ["a.md:5:1"]
    assert not (tmp_path / "index_version.json.tmp").exists()


def test_an_unreadable_marker_reads_as_none(tmp_path):
    m = manager(tmp_path, active=0, collections=[], opened_with_marker=False)
    (tmp_path / "index_version.json").write_text("{half a marker")

    assert m.read_marker() is None and m.index_format() == 1


def test_commit_switches_the_active_generation_only_after_the_marker_names_it(tmp_path):
    m = manager(tmp_path, active=1, collections=["docs__g1", "docs__g2"], opened_with_marker=True)
    new = Generation(2, "docs__g2", str(tmp_path / "tantivy_index__g2"), None)

    previous = m.commit_generation(new, "none", ["a.md:1:1"])

    assert previous.number == 1
    assert m.active.number == 2 and m.read_marker()["generation"] == 2


def test_a_failed_build_removes_its_own_partial_generation_and_keeps_the_active_one(tmp_path):
    m = manager(tmp_path, active=1, collections=["docs__g1"], opened_with_marker=True)

    def half_build(number):
        m.qdrant_client.names.add(f"docs__g{number}")
        (tmp_path / f"tantivy_index__g{number}").mkdir()
        raise RuntimeError("embedding failed halfway")

    m._open_generation = half_build

    with pytest.raises(RuntimeError):
        m.build_generation([])

    assert m.qdrant_client.names == {"docs__g1"}
    assert not (tmp_path / "tantivy_index__g2").exists()
    assert m.active.number == 1


def test_the_active_generation_is_never_dropped(tmp_path):
    m = manager(tmp_path, active=2, collections=["docs__g2"], opened_with_marker=True)

    m.drop_generation(2)

    assert m.qdrant_client.names == {"docs__g2"}


def test_a_crash_leftover_is_skipped_by_the_next_build_and_removed_by_startup_cleanup(tmp_path):
    m = manager(tmp_path, active=1, collections=["docs__g1", "docs__g3"], opened_with_marker=True)
    m.mark_index_format("none", generation=1, corpus=[])
    built = []
    m._open_generation = lambda number: built.append(number) or (_ for _ in ()).throw(RuntimeError("stop"))

    with pytest.raises(RuntimeError):
        m.build_generation([])
    m.cleanup_orphans()

    assert built == [4]
    assert m.qdrant_client.names == {"docs__g1"}
