"""Latency benchmark (spec 0003), with the API faked and the speech models stubbed."""
import json
from types import SimpleNamespace

import pytest

from src.eval import latency_benchmark as lb


# --- fakes ---

class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _StreamResp(_Resp):
    """/api/query/stream: meta, one sentence, then done, built from the same answer the old fake returned."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        d = self._data
        yield json.dumps({"type": "meta", "cache": d.get("cache")}).encode()
        yield json.dumps({"type": "sentence", "index": 0, "text": d.get("answer", "")}).encode()
        yield json.dumps({"type": "done", "answer": d.get("answer", ""), "trace": d.get("trace", []),
                          "cache": d.get("cache"), "total_ms": d.get("total_latency_ms")}).encode()


def answered(cache=None):
    trace = [{"layer": "bm25", "elapsed_ms": 10.0}, {"layer": "vector", "elapsed_ms": 20.0},
             {"layer": "confidence", "elapsed_ms": 0.0}, {"layer": "generation", "elapsed_ms": 300.0}]
    if cache:
        trace = [{"layer": "cache", "elapsed_ms": 0.0}]
    return {"answer": "Blobs hold files. [Source: a.pdf | Page: 3]", "trace": trace,
            "total_latency_ms": 5.0 if cache else 400.0, "cache": cache}


GATED = {"answer": "Insufficient context.", "cache": None, "total_latency_ms": 50.0,
         "trace": [{"layer": "bm25", "elapsed_ms": 10.0}, {"layer": "confidence", "elapsed_ms": 0.0}]}


class FakeSession:
    """Answers in scope questions, refuses the rest, and caches like the real API."""

    def __init__(self, ollama="online", fail=(), model_error=(), cache_enabled=True):
        self.ollama, self.fail, self.model_error = ollama, set(fail), set(model_error)
        self.cache_enabled, self.cached, self.deletes = cache_enabled, {}, 0

    def get(self, url, timeout=None):
        if url.endswith("/health"):
            return _Resp({"ollama_status": self.ollama, "generation_model": "llama3.2:3b"})
        if url.endswith("/cache"):
            return _Resp({"enabled": self.cache_enabled})
        if url.endswith("/corpus"):
            return _Resp([{"file": "a.pdf", "chunks": 3}])
        if url.endswith("/api/ps"):
            return _Resp({"models": [{"name": "llama3.2:3b", "size": 100, "size_vram": 100}]})
        raise AssertionError(url)

    def delete(self, url, timeout=None):
        self.deletes += 1
        self.cached.clear()
        return _Resp({})

    def post(self, url, json=None, timeout=None, stream=False):
        assert url.endswith("/query/stream"), "The benchmark asks through the stream, as the dashboard does"
        q = json["query"]
        if q in self.fail:
            return _StreamResp({}, status=500)
        if q in self.model_error:
            return _StreamResp({"answer": " Error processing response: connection refused", "trace": [],
                                "total_latency_ms": 1.0, "cache": None})
        if self.cache_enabled and q in self.cached:
            return _StreamResp({**self.cached[q], "cache": "exact", "total_latency_ms": 5.0,
                                "trace": [{"layer": "cache", "elapsed_ms": 0.0}]})
        self.cached[q] = answered() if q.startswith("in") else GATED
        return _StreamResp(self.cached[q])


@pytest.fixture
def golden(tmp_path):
    items = [{"question": f"in {i}", "in_scope": True} for i in range(4)] + \
            [{"question": f"out {i}", "in_scope": False} for i in range(2)]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(items))
    return path


def bench(tmp_path, golden, session, **kw):
    kw.setdefault("skip_voice", True)
    return lb.Benchmark("http://api/api", "localhost:11434", session=session, golden_path=golden,
                        out_dir=tmp_path / "runs", **kw)


# --- the numbers ---

def test_nearest_rank_on_odd_even_and_empty_lists():
    assert lb.nearest_rank([5, 1, 3], 0.5) == 3
    assert lb.nearest_rank([4, 1, 3, 2], 0.5) == 2          # lower middle for an even n
    assert lb.nearest_rank(list(range(1, 21)), 0.95) == 19
    assert lb.summarize([]) == {"n": 0, "p50": None, "p95": None}


def test_a_first_pass_cache_hit_is_kept_out_of_the_cold_numbers():
    text = [lb.Benchmark.text_record("first", {"id": 0, "in_scope": True}, answered(), 410.0, None),
            lb.Benchmark.text_record("first", {"id": 1, "in_scope": True}, answered(cache="semantic"), 6.0, None)]

    summary = lb.Benchmark.summarize_run(text, [])

    assert summary["first"]["answered"]["n"] == 1
    assert summary["first"]["cache_hit"]["n"] == 1


# --- full runs against a faked API ---

def test_full_text_run_groups_by_path_and_saves(tmp_path, golden):
    session = FakeSession()
    path, run = bench(tmp_path, golden, session, label="base").run()

    saved = json.loads(path.read_text())
    assert path.name.endswith("-base.json")
    assert session.deletes >= 1, "Pass 1 must start with a cleared cache"
    first, repeat = saved["summary"]["first"], saved["summary"]["repeat"]
    assert (first["answered"]["n"], first["refused"]["n"], first["cache_hit"]["n"]) == (4, 2, 0)
    assert repeat["cache_hit"]["n"] == 6
    assert first["answered"]["metrics"]["generation"]["p50"] == 300.0
    assert "generation" not in first["refused"]["metrics"], "A refusal stops at the confidence gate"
    assert saved["meta"]["ollama"]["placement_start"] == "gpu"


def test_errors_are_left_out_and_three_in_a_row_stop_the_run(tmp_path, golden):
    _, run = bench(tmp_path, golden, FakeSession(fail={"in 1"}, model_error={"in 2"})).run()
    assert run["summary"]["errors"] == 4                    # two questions, two passes
    assert run["summary"]["first"]["answered"]["n"] == 2

    path, run = bench(tmp_path, golden, FakeSession(fail={"in 1", "in 2", "in 3"})).run()
    assert run["meta"]["aborted"] is True and path.exists()


def test_voice_in_mock_mode_is_never_reported(tmp_path, golden):
    class MockSpeech:
        model = None

    b = bench(tmp_path, golden, FakeSession(), skip_voice=False, speech_factory=lambda: (MockSpeech(), MockSpeech()))
    _, run = b.run()

    assert run["meta"]["voice_valid"] is False and run["voice"] == []


# --- power state and checker in every run (spec 0006, AC-23) ---

def fake_pmset(batt, settings, fail=False):
    def run(cmd, **kw):
        if fail:
            raise OSError("no pmset")
        return SimpleNamespace(stdout=batt if cmd[-1] == "batt" else settings)
    return run


@pytest.mark.parametrize("batt, settings, expected", [
    ("Now drawing from 'AC Power'\n -InternalBattery-0", " lowpowermode         0\n", ("AC", False)),
    ("Now drawing from 'Battery Power'\n", " lowpowermode         1\n", ("Battery", True)),
    ("Now drawing from 'UPS Power'\n", " lowpowermode         0\n", (None, False)),
    ("Now drawing from 'AC Power'\n", " sleep 1\n", ("AC", None)),
    ("", "", (None, None)),
])
def test_power_state_parses_pmset(batt, settings, expected):
    state = lb.power_state(run=fake_pmset(batt, settings))
    assert (state["source"], state["low_power_mode"]) == expected


def test_power_state_is_null_when_pmset_fails():
    assert lb.power_state(run=fake_pmset("", "", fail=True)) == {"source": None, "low_power_mode": None}


def test_power_that_changes_during_the_run_is_unknown():
    ac, batt = {"source": "AC", "low_power_mode": False}, {"source": "Battery", "low_power_mode": True}
    assert lb.power_record(ac, ac) == {**ac, "changed": False}
    assert lb.power_record(ac, batt) == {"source": None, "low_power_mode": None, "changed": True}
    assert lb.power_line(lb.power_record(ac, batt)).startswith("power unknown")
    assert lb.power_line(lb.power_record(ac, ac)) == "power AC, Low Power Mode off"


def test_run_records_power_and_checker(tmp_path, golden):
    session = FakeSession()
    original = session.get
    session.get = lambda url, timeout=None: _Resp({"ollama_status": "online", "generation_model": "llama3.2:3b",
                                                   "grounding_checker_effective": "judge:llama3.2:1b"}) \
        if url.endswith("/health") else original(url, timeout)
    readings = iter([{"source": "AC", "low_power_mode": False}] * 2)
    path, _ = bench(tmp_path, golden, session, power_reader=lambda: next(readings)).run()
    meta = json.loads(path.read_text())["meta"]
    assert meta["machine"]["power"] == {"source": "AC", "low_power_mode": False, "changed": False}
    assert meta["grounding_checker_effective"] == "judge:llama3.2:1b"
    assert "drafts" not in meta
