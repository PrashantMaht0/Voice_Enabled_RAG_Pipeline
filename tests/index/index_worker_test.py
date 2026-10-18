"""The single index worker: one build at a time, requests during a build merge into one follow up,
and its status reports idle, building or failed with a short reason."""
import threading

import pytest

from src.vector_db.index_worker import ERROR_CHARS, IndexWorker, short_reason


class Builds:
    """A stand in build that waits on a gate, so tests decide when each build finishes."""

    def __init__(self, fail_on=()):
        self.count = 0
        self.running = 0
        self.overlapped = False
        self.started = threading.Event()
        self.gate = threading.Event()
        self.fail_on = set(fail_on)
        self._lock = threading.Lock()

    def __call__(self, cancel):
        with self._lock:
            self.count += 1
            number = self.count
            self.running += 1
            self.overlapped = self.overlapped or self.running > 1
        self.started.set()
        self.gate.wait(5)
        with self._lock:
            self.running -= 1
        if number in self.fail_on:
            raise ValueError("cannot read broken.pdf\nTraceback line that must not show")


@pytest.fixture
def worker():
    made = []

    def make(build):
        made.append(IndexWorker(build))
        return made[-1]

    yield make
    for w in made:
        w.stop(timeout=2)


def test_a_new_worker_is_idle_with_nothing_pending(worker):
    status = worker(Builds()).status()

    assert status == {"state": "idle", "pending": False, "started_at": None, "finished_at": None, "last_error": None}


def test_a_request_returns_at_once_and_the_status_shows_building(worker):
    builds = Builds()
    w = worker(builds)

    w.request_rebuild("a.md uploaded")
    assert builds.started.wait(2)
    status = w.status()

    assert status["state"] == "building" and status["started_at"] and status["finished_at"] is None
    builds.gate.set()
    assert w.wait_idle(2)
    assert w.status()["state"] == "idle" and w.status()["finished_at"]


def test_requests_during_a_build_merge_into_exactly_one_follow_up(worker):
    builds = Builds()
    w = worker(builds)
    w.request_rebuild("first")
    assert builds.started.wait(2)

    for i in range(5):
        w.request_rebuild(f"upload {i}")
    assert w.status()["pending"] is True
    builds.gate.set()

    assert w.wait_idle(3)
    assert builds.count == 2
    assert w.builds == 2


def test_two_builds_never_run_at_once(worker):
    builds = Builds()
    w = worker(builds)
    w.request_rebuild("first")
    assert builds.started.wait(2)
    w.request_rebuild("second")

    builds.gate.set()

    assert w.wait_idle(3)
    assert not builds.overlapped


def test_a_failed_build_reports_failed_with_a_short_reason_and_does_not_retry(worker, capsys):
    builds = Builds(fail_on={1})
    builds.gate.set()
    w = worker(builds)

    w.request_rebuild("broken upload")
    assert w.wait_idle(2)
    status = w.status()

    assert status["state"] == "failed"
    assert status["last_error"] == "ValueError: cannot read broken.pdf"
    assert "Traceback" not in status["last_error"]
    assert builds.count == 1, "A failure must never retry by itself."
    assert "Traceback" in capsys.readouterr().err, "The full trace goes to the log."


def test_a_follow_up_requested_during_a_failing_build_still_runs(worker):
    builds = Builds(fail_on={1})
    w = worker(builds)
    w.request_rebuild("first")
    assert builds.started.wait(2)
    w.request_rebuild("second")

    builds.gate.set()

    assert w.wait_idle(3)
    assert builds.count == 2
    assert w.status()["state"] == "idle" and w.status()["last_error"] is None


def test_the_next_request_after_a_failure_builds_again(worker):
    """failed -> building on the next request."""
    builds = Builds(fail_on={1})
    builds.gate.set()
    w = worker(builds)
    w.request_rebuild("first")
    assert w.wait_idle(2) and w.status()["state"] == "failed"

    w.request_rebuild("fixed the file")

    assert w.wait_idle(2)
    assert w.status()["state"] == "idle"


def test_short_reason_is_the_class_and_first_line_capped_at_200_characters():
    assert short_reason(RuntimeError("")) == "RuntimeError"
    assert short_reason(OSError("disk full\nmore")) == "OSError: disk full"
    assert len(short_reason(ValueError("x" * 500))) == ERROR_CHARS == 200


def test_stop_ends_the_thread_and_signals_a_build_in_progress_to_cancel(worker):
    builds = Builds()
    w = worker(builds)
    w.request_rebuild("first")
    assert builds.started.wait(2)

    builds.gate.set()
    w.stop(timeout=2)

    assert w.cancel.is_set()
    assert not w._thread.is_alive()
