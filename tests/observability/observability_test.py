"""The trace store, the metrics math and the trace endpoints."""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes
from src.api.guards import install_guards
from src.observability import metrics
from src.observability.store import TraceStore


@pytest.fixture
def store(tmp_path):
    return TraceStore(str(tmp_path / "traces.sqlite3"), max_traces=100)


def add(store, path="answered", spans=(), **fields):
    trace_id = store.start("typed", fields.pop("question", "What is DNS?"))
    store.finish(trace_id, spans=list(spans), path=path, **fields)
    return trace_id


# --- writes and reads ---

def test_a_finished_trace_reads_back_with_its_spans_and_json_fields(store):
    trace_id = add(store, spans=[{"layer": "rerank", "elapsed_ms": 12.5, "details": "12 chunks"}],
                   answer="An answer.", dropped_sentences=["A dropped one."], retrieved=[{"page": 3}],
                   config={"generation_model": "m"}, total_ms=900.0, grounding_checker="judge:x")
    trace = store.get(trace_id)
    assert trace["path"] == "answered" and trace["answer"] == "An answer."
    assert trace["dropped_sentences"] == ["A dropped one."] and trace["retrieved"] == [{"page": 3}]
    assert trace["config"] == {"generation_model": "m"}
    assert trace["spans"] == [{"layer": "rerank", "elapsed_ms": 12.5, "details": "12 chunks",
                               "prompt_tokens": None, "output_tokens": None}]


def test_a_started_trace_is_in_progress_and_left_out_of_recent(store):
    store.start("voice", "still answering")
    assert store.recent() == []


def test_recent_lists_newest_first_with_the_summary_fields(store):
    first, second = add(store, question="one"), add(store, question="two")
    rows = store.recent(limit=50)
    assert [r["id"] for r in rows] == [second, first]
    assert set(rows[0]) == {"id", "created_at", "channel", "question", "path", "total_ms", "answer",
                            "prompt_tokens", "output_tokens"}


def test_unknown_trace_is_none(store):
    assert store.get(999) is None


def test_voice_timings_update_an_existing_trace_only(store):
    trace_id = add(store)
    assert store.add_voice(trace_id, stt_ms=500, first_audio_ms=6000, spoken_total_ms=None) is True
    trace = store.get(trace_id)
    assert (trace["stt_ms"], trace["first_audio_ms"], trace["spoken_total_ms"]) == (500, 6000, None)
    assert store.add_voice(999, stt_ms=1) is False


def test_finish_ignores_fields_it_does_not_know(store):
    trace_id = store.start("typed", "q")
    store.finish(trace_id, path="answered", not_a_column="x")
    assert store.get(trace_id)["path"] == "answered"


# --- retention and failures ---

def test_past_the_limit_the_oldest_traces_go_with_their_spans(tmp_path):
    store = TraceStore(str(tmp_path / "t.sqlite3"), max_traces=3)
    ids = [add(store, spans=[{"layer": "bm25", "elapsed_ms": 1}]) for _ in range(4)]
    assert [r["id"] for r in store.recent()] == ids[:0:-1]
    orphans = store.query("SELECT COUNT(*) AS n FROM spans WHERE trace_id NOT IN (SELECT id FROM traces)")[0]["n"]
    assert orphans == 0


def test_a_failed_write_is_one_warning_a_counter_and_no_exception(tmp_path, capsys):
    path = tmp_path / "t.sqlite3"
    store = TraceStore(str(path))
    add(store)
    os.chmod(path, 0o444)
    try:
        assert store.start("typed", "q") is None
    finally:
        os.chmod(path, 0o644)
    assert store.failed_writes == 1
    assert capsys.readouterr().out.count("[Traces] WARNING") == 1


def test_finish_without_a_trace_id_does_nothing(store):
    assert store.finish(None, path="answered") is None and store.failed_writes == 0


# --- metrics math ---

def test_nearest_rank_percentiles():
    assert metrics.nearest_rank([], 0.5) is None
    assert metrics.nearest_rank([5.0], 0.95) == 5.0
    assert metrics.percentiles([1, 2, 3, 4, 100]) == {"p50": 3, "p95": 100, "p99": 100, "n": 5}


def test_counts_and_rates_over_every_path(store, tmp_path):
    add(store, path="answered", grounding_checker="judge:x", sentences_dropped=0, total_ms=100)
    add(store, path="answered", grounding_checker="judge:x", sentences_dropped=1, total_ms=300)
    add(store, path="refused_grounding", grounding_checker="judge:x", sentences_dropped=2)
    add(store, path="gated", answer="Insufficient context.")
    add(store, path="cache", answer="A cached answer.")
    add(store, path="cache", answer="Insufficient context.")
    add(store, path="empty_query", answer="Please ask a full question.")
    add(store, path="error", error="boom")
    m = metrics.compute(store, "all", benchmarks_dir=tmp_path)
    assert m["counts"] == {"asked": 8, "answered": 3, "refused_gate": 2, "refused_grounding": 1, "refused_model": 0,
                           "refused_empty": 1, "cache_hits": 2, "aborted": 0, "errors": 1}
    assert m["rates"]["cache_hit_rate"] == round(2 / 7, 3), "errors are not served"
    assert m["rates"]["refusal_rate"] == round(4 / 7, 3)
    assert m["rates"]["grounding_pass_rate"] == round(1 / 3, 3), "only checked answers count"


def test_unchecked_answers_are_left_out_of_the_grounding_pass_rate(store, tmp_path):
    add(store, path="answered", grounding_checker=None, sentences_dropped=0)
    assert metrics.compute(store, "all", benchmarks_dir=tmp_path)["rates"]["grounding_pass_rate"] is None


def test_stage_times_come_from_answered_requests_only(store, tmp_path):
    add(store, path="answered", spans=[{"layer": "rerank", "elapsed_ms": 10}], total_ms=100, first_sentence_ms=50)
    add(store, path="gated", spans=[{"layer": "rerank", "elapsed_ms": 999}], total_ms=5)
    stages = metrics.compute(store, "all", benchmarks_dir=tmp_path)["stages"]
    assert stages["rerank"] == {"p50": 10, "p95": 10, "p99": 10, "n": 1}
    assert stages["total_ms"]["n"] == 1 and stages["first_sentence_ms"]["p50"] == 50


def test_voice_percentiles_use_traces_that_have_voice_timings(store, tmp_path):
    store.add_voice(add(store), stt_ms=400, first_audio_ms=6000, spoken_total_ms=9000)
    add(store)
    voice = metrics.compute(store, "all", benchmarks_dir=tmp_path)["voice"]
    assert voice["first_audio_ms"] == {"p50": 6000, "p95": 6000, "p99": 6000, "n": 1}


def test_the_window_leaves_out_older_traces(store, tmp_path):
    old = add(store)
    add(store)
    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="milliseconds")
    store._write("age a trace", lambda c: c.execute("UPDATE traces SET created_at = ? WHERE id = ?", (two_days_ago, old)))
    assert metrics.compute(store, "24h", benchmarks_dir=tmp_path)["counts"]["asked"] == 1
    assert metrics.compute(store, "7d", benchmarks_dir=tmp_path)["counts"]["asked"] == 2


def test_empty_store_has_no_rates(store, tmp_path):
    m = metrics.compute(store, "24h", benchmarks_dir=tmp_path)
    assert m["counts"]["asked"] == 0 and m["rates"] == {"cache_hit_rate": None, "refusal_rate": None,
                                                        "grounding_pass_rate": None, "retrieval_pass_rate": None, "error_rate": None}


def test_latest_quality_reads_the_newest_sweep_winner(tmp_path):
    def sweep(name, winner, tables):
        (tmp_path / name).write_text(json.dumps({"winner": winner, "finished": "now", "configs": {winner: {
            "settings": {"x": 1}, "golden": {"answered": 44, "in_scope": 47, "refused": 20, "out_of_scope": 20},
            "ragas": {"faithfulness": 0.88}, "tables": {"right_value_and_page": tables},
            "figures": {"right_value_and_page": 11}}}}))
    sweep("sweep-20260101-000000.json", "A0", 10)
    sweep("sweep-20260102-000000.json", "A7", 20)
    (tmp_path / "sweep-20260103-000000.json").write_text(json.dumps({"winner": None}))
    q = metrics.latest_quality(tmp_path)
    assert (q["file"], q["winner"], q["tables"], q["faithfulness"]) == ("sweep-20260102-000000.json", "A7", 20, 0.88)
    assert metrics.latest_quality(tmp_path / "missing") is None


# --- endpoints ---

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_every_query_writes_one_finished_trace(client, monkeypatch):
    response = client.post("/api/query", json={"query": "You"}).json()
    traces = routes.get_traces().recent()
    assert [t["id"] for t in traces] == [response["trace_id"]] and traces[0]["path"] == "empty_query"
    assert routes.get_traces().get(response["trace_id"])["spans"][0]["layer"] == "empty_query"


@pytest.mark.parametrize("url, status", [("/api/metrics?window=2d", 422), ("/api/traces?limit=0", 422),
                                         ("/api/traces?limit=201", 422), ("/api/traces/999999", 404),
                                         ("/api/metrics?window=all", 200), ("/api/traces", 200)])
def test_endpoint_validation(client, url, status):
    assert client.get(url).status_code == status


def test_trace_endpoint_returns_the_full_trace(client):
    trace_id = client.post("/api/query", json={"query": "You"}).json()["trace_id"]
    trace = client.get(f"/api/traces/{trace_id}").json()
    assert {"spans", "retrieved", "dropped_sentences", "answer", "stt_ms", "first_audio_ms"} <= set(trace)


def test_voice_timings_endpoint(client):
    trace_id = client.post("/api/query", json={"query": "You"}).json()["trace_id"]
    assert client.post(f"/api/traces/{trace_id}/voice", json={"stt_ms": 300}).status_code == 200
    assert client.get(f"/api/traces/{trace_id}").json()["stt_ms"] == 300
    assert client.post("/api/traces/999999/voice", json={"stt_ms": 300}).status_code == 404


@pytest.mark.parametrize("body", ['{"stt_ms": -1}', '{"stt_ms": 1e400}', '{"first_audio_ms": 700000}'])
def test_voice_timings_must_be_finite_and_in_range(body):
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    client = TestClient(install_guards(app), base_url="http://localhost")
    trace_id = client.post("/api/query", json={"query": "You"}).json()["trace_id"]
    reply = client.post(f"/api/traces/{trace_id}/voice", content=body, headers={"Content-Type": "application/json"})
    assert reply.status_code == 422
    assert client.get("/api/metrics").status_code == 200


def test_metrics_endpoint_counts_the_traffic(client):
    client.post("/api/query", json={"query": "You"})
    m = client.get("/api/metrics?window=24h").json()
    assert m["counts"]["refused_empty"] == 1 and m["failed_writes"] == 0


# --- token columns, the upgrade, windowed recent ---

def old_schema_file(path, rows=3):
    """A trace file made by the first trace schema (no token columns), with a few finished traces and spans."""
    import sqlite3
    from src.observability import store as store_module
    schema = store_module.SCHEMA.replace(",\n    prompt_tokens INTEGER,\n    output_tokens INTEGER", "")
    assert "prompt_tokens" not in schema
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    for i in range(rows):
        conn.execute("INSERT INTO traces (created_at, channel, question, search_query, path) VALUES (?, ?, ?, ?, ?)",
                     ("2026-10-01T00:00:00.000+00:00", "typed", f"Q{i}", f"Q{i}", "answered"))
        conn.execute("INSERT INTO spans (trace_id, layer, elapsed_ms) VALUES (?, ?, ?)", (i + 1, "rerank", 5.0))
    conn.commit()
    conn.close()


def columns(path, table):
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_an_old_trace_file_gains_the_token_columns_and_keeps_every_row(tmp_path):
    path = str(tmp_path / "old.sqlite3")
    old_schema_file(path, rows=3)
    store = TraceStore(path)
    rows = store.query("SELECT id, prompt_tokens, output_tokens FROM traces ORDER BY id")
    assert {"prompt_tokens", "output_tokens"} <= columns(path, "traces")
    assert {"prompt_tokens", "output_tokens"} <= columns(path, "spans")
    assert [tuple(r) for r in rows] == [(1, None, None), (2, None, None), (3, None, None)]
    assert store.get(1)["spans"] == [{"layer": "rerank", "elapsed_ms": 5.0, "details": None,
                                      "prompt_tokens": None, "output_tokens": None}]


def test_a_new_trace_file_has_the_token_columns_from_the_start(tmp_path):
    path = str(tmp_path / "new.sqlite3")
    TraceStore(path).recent()
    assert {"prompt_tokens", "output_tokens"} <= columns(path, "traces")


def test_an_upgrade_that_fails_warns_once_counts_no_write_and_retries_on_the_next_connection(tmp_path, monkeypatch,
                                                                                               capsys):
    from src.observability import store as store_module
    path = str(tmp_path / "old.sqlite3")
    old_schema_file(path, rows=1)
    good = store_module.ADDED_COLUMNS
    monkeypatch.setattr(store_module, "ADDED_COLUMNS", {"traces": ("broken column (",)})
    store = TraceStore(path)
    store.query("SELECT 1")
    store.query("SELECT 1")
    assert capsys.readouterr().out.count("could not upgrade the trace store") == 1
    assert store.failed_writes == 0 and not store._ready
    monkeypatch.setattr(store_module, "ADDED_COLUMNS", good)
    store.query("SELECT 1")
    assert store._ready and "prompt_tokens" in columns(path, "traces")


def test_finish_stores_the_token_totals_and_each_span_pair(store):
    trace_id = add(store, spans=[{"layer": "generation", "elapsed_ms": 300.0, "details": "LLM call",
                                  "prompt_tokens": 900, "output_tokens": 40},
                                 {"layer": "rerank", "elapsed_ms": 5.0, "details": "12 chunks"}],
                   prompt_tokens=900, output_tokens=40)
    trace = store.get(trace_id)
    assert (trace["prompt_tokens"], trace["output_tokens"]) == (900, 40)
    assert [(s["layer"], s["prompt_tokens"], s["output_tokens"]) for s in trace["spans"]] == \
        [("generation", 900, 40), ("rerank", None, None)]


def age(store, trace_id, days):
    when = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds")
    store._write("age a trace", lambda c: c.execute("UPDATE traces SET created_at = ? WHERE id = ?", (when, trace_id)))


def test_recent_keeps_only_traces_since_the_window_start(store):
    old = add(store, question="old")
    new = add(store, question="new")
    age(store, old, 2)
    assert [r["id"] for r in store.recent(since=metrics.since("24h"))] == [new]
    assert [r["id"] for r in store.recent(since=None)] == [new, old]


# --- the new metrics ---

@pytest.mark.parametrize("path, answer, expected", [
    ("answered", "An answer.", True),
    ("cache", "A cached answer.", True),
    ("cache", "Insufficient context.", False),
    ("cache", None, True),
    ("gated", "Insufficient context.", False),
    ("refused_grounding", "Insufficient context.", False),
    ("error", None, False),
    ("aborted", None, False),
])
def test_one_answered_rule(path, answer, expected):
    assert metrics.is_answered(path, answer) is expected


def test_retrieval_counts_traces_that_reached_the_gate_and_passes_those_not_gated(store, tmp_path):
    add(store, path="answered", confidence=0.9)
    add(store, path="refused_grounding", confidence=0.8)
    add(store, path="gated", confidence=0.1)
    add(store, path="cache", confidence=0.9, answer="A cached answer.")
    add(store, path="error", confidence=0.9)
    add(store, path="empty_query")
    m = metrics.compute(store, "all", benchmarks_dir=tmp_path)
    assert m["retrieval"] == {"reached": 3, "passed": 2}
    assert m["rates"]["retrieval_pass_rate"] == round(2 / 3, 3)


def test_the_error_rate_is_errors_over_asked_to_four_places(store, tmp_path):
    for _ in range(2):
        add(store, path="answered")
    add(store, path="error", error="boom")
    assert metrics.compute(store, "all", benchmarks_dir=tmp_path)["rates"]["error_rate"] == round(1 / 3, 4)


def test_latency_is_nearest_rank_p50_and_p99_of_answered_total_time(store, tmp_path):
    for ms in (100, 200, 300, 400, 5000):
        add(store, path="answered", total_ms=ms)
    add(store, path="gated", total_ms=99999)
    add(store, path="cache", total_ms=1, answer="A cached answer.")
    assert metrics.compute(store, "all", benchmarks_dir=tmp_path)["latency"] == {"p50": 300, "p99": 5000, "n": 5}


def test_tokens_leave_out_traces_stored_before_counting_began(store, tmp_path):
    add(store, prompt_tokens=1000, output_tokens=50)
    add(store, path="cache", answer="A cached answer.", prompt_tokens=0, output_tokens=0)
    add(store)                                          # an old row: tokens null
    tokens = metrics.compute(store, "all", benchmarks_dir=tmp_path)["tokens"]
    assert tokens == {"prompt": 1000, "output": 50, "total": 1050, "per_question": 525, "counted": 2, "asked": 3}


def test_with_no_counted_trace_there_is_no_per_question_figure(store, tmp_path):
    add(store)
    tokens = metrics.compute(store, "all", benchmarks_dir=tmp_path)["tokens"]
    assert tokens["counted"] == 0 and tokens["total"] == 0 and tokens["per_question"] is None


def test_the_grounding_block_counts_checked_answers_and_those_with_nothing_removed(store, tmp_path):
    add(store, path="answered", grounding_checker="judge:x", sentences_dropped=0)
    add(store, path="answered", grounding_checker="judge:x", sentences_dropped=None)
    add(store, path="answered", grounding_checker="judge:x", sentences_dropped=2)
    add(store, path="answered", grounding_checker=None, sentences_dropped=0)
    assert metrics.compute(store, "all", benchmarks_dir=tmp_path)["grounding"] == {"checked": 3, "passed": 2}


# --- the endpoints ---

def test_traces_endpoint_filters_by_window_and_adds_answered_and_tokens(client):
    store = routes.get_traces()
    old = add(store, path="answered", answer="An answer.", question="old")
    gated = add(store, path="gated", answer="Insufficient context.", prompt_tokens=160, output_tokens=12)
    cached = add(store, path="cache", answer="Insufficient context.")
    age(store, old, 2)
    items = client.get("/api/traces?window=24h").json()
    assert [t["id"] for t in items] == [cached, gated]
    assert [t["answered"] for t in items] == [False, False]
    assert "answer" not in items[0] and (items[1]["prompt_tokens"], items[1]["output_tokens"]) == (160, 12)
    assert [t["id"] for t in client.get("/api/traces").json()] == [cached, gated, old], "default window is all"


@pytest.mark.parametrize("url", ["/api/traces?window=2d", "/api/traces?window=", "/api/metrics?window=1d"])
def test_a_bad_window_is_422(client, url):
    assert client.get(url).status_code == 422


def test_metrics_endpoint_has_the_new_blocks_and_keeps_the_old_ones(client):
    client.post("/api/query", json={"query": "You"})
    m = client.get("/api/metrics?window=24h").json()
    assert {"retrieval", "grounding", "latency", "tokens", "quality", "counts", "stages", "voice"} <= set(m)
    assert {"retrieval_pass_rate", "error_rate", "grounding_pass_rate"} <= set(m["rates"])
    assert m["tokens"]["counted"] == 1 and m["tokens"]["total"] == 0
    assert all({"p50", "p95", "p99", "n"} == set(v) for v in m["voice"].values())


def test_trace_endpoint_returns_token_fields(client):
    trace_id = client.post("/api/query", json={"query": "You"}).json()["trace_id"]
    trace = client.get(f"/api/traces/{trace_id}").json()
    assert (trace["prompt_tokens"], trace["output_tokens"]) == (0, 0)
    assert all({"prompt_tokens", "output_tokens"} <= set(s) for s in trace["spans"])
