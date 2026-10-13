"""Configuration sweep (spec 0005): gates, the winner rule, Ragas order, apply, resume. No API or Ollama runs."""
import json
from pathlib import Path

import pytest
import yaml

from src.eval import system_sweep as ss
from src.eval.table_benchmark import BenchmarkError


def record(tables=19, figures=4, answered=44, refused=20, p50=4700.0, status="ran"):
    return {"status": status, "tables": {"right_value_and_page": tables}, "figures": {"right_value_and_page": figures},
            "golden": {"answered": answered, "in_scope": 47, "refused": refused, "out_of_scope": 20, "leaked": []},
            "latency": {"typed_p50_ms": p50}, "settings": {"figure_parser": "none"}, "ragas": None}


def gated(a0, **kw):
    r = record(**kw)
    r["gates"] = ss.base_gates(r, a0)
    return r


@pytest.fixture
def a0():
    r = record()
    r["gates"] = ss.base_gates(r, r)
    return r


# --- the gates that need no Ragas (covers: AC-13) ---

def test_today_setup_passes_its_own_gates(a0):
    assert a0["gates"] == {"all_trick_questions_refused": True, "answered_90_percent": True,
                           "tables_within_2_of_a0": True}


def test_one_answered_trick_question_fails_the_refusal_gate(a0):
    assert gated(a0, refused=19)["gates"]["all_trick_questions_refused"] is False


@pytest.mark.parametrize("answered, passes", [(43, True), (42, False)])
def test_at_least_90_percent_of_in_scope_questions_must_be_answered(a0, answered, passes):
    assert gated(a0, answered=answered)["gates"]["answered_90_percent"] is passes   # 43/47 = 91.5%, 42/47 = 89.4%


@pytest.mark.parametrize("tables, passes", [(17, True), (16, False)])
def test_tables_may_fall_at_most_two_below_today(a0, tables, passes):
    assert gated(a0, tables=tables)["gates"]["tables_within_2_of_a0"] is passes


def test_a_record_that_did_not_run_never_passes(a0):
    assert ss.passes({**a0, "status": "failed"}) is False


# --- ranking and the stage A rule (covers: AC-13) ---

def test_scores_within_two_of_the_top_tie_and_the_tie_goes_to_the_lowest_p50():
    records = {"slow": record(figures=11, p50=5200), "fast": record(figures=9, p50=4700), "far": record(figures=5)}
    assert ss.rank(records, lambda r: ss.right(r, "figures")) == ["fast", "slow", "far"]


def test_a_figure_reader_must_beat_today_by_three_to_win_stage_a(a0):
    run = {"configs": {"A0": a0, "A3": gated(a0, figures=6, p50=4600), "A7": gated(a0, figures=5, p50=5000)}}
    winner, reason = ss.stage_a_winner(run)
    assert winner == "A0" and "fewer than 3" in reason


def test_a_clear_figure_gain_wins_stage_a(a0):
    run = {"configs": {"A0": a0, "A3": gated(a0, figures=9, p50=4700), "A7": gated(a0, figures=11, p50=5200)}}
    assert ss.stage_a_winner(run) == ("A3", "A3 scores 9 on figures, 5 above A0")


def test_a_figure_reader_that_fails_a_gate_cannot_win_stage_a(a0):
    run = {"configs": {"A0": a0, "A7": gated(a0, figures=15, refused=19)}}
    assert ss.stage_a_winner(run)[0] == "A0"


def test_the_reused_stage_b_record_is_ranked_only_once(a0):
    a3 = gated(a0, figures=9)
    run = {"configs": {"A0": a0, "A3": a3, "B-llama": {**a3, "reused_from": "A3"}}}
    assert ss.overall_order(run) == ["A3", "A0"]


def test_the_overall_order_uses_tables_plus_figures(a0):
    run = {"configs": {"A0": a0, "A7": gated(a0, tables=20, figures=11, p50=5200),
                       "B-qwen": gated(a0, tables=17, figures=11, p50=4500)}}
    assert ss.overall_order(run) == ["A7", "B-qwen", "A0"]


# --- Ragas for finalists (covers: AC-12, AC-13) ---

def ragas_run(tmp_path, a0, faith):
    run = {"configs": {"A0": a0, "A7": gated(a0, figures=11), "B-qwen": gated(a0, figures=10),
                       "A3": gated(a0, figures=9), "X": gated(a0, figures=8)}}
    asked = []

    def fake(cid, rec, stamp):
        asked.append(cid)
        value = faith[cid]
        return value if isinstance(value, str) else {"faithfulness": value, "answer_relevancy": 0.95, "file": "f"}
    return run, asked, fake


def test_ragas_runs_for_today_and_the_top_configuration(tmp_path, a0, monkeypatch):
    run, asked, fake = ragas_run(tmp_path, a0, {"A0": 0.88, "A7": 0.86})
    monkeypatch.setattr(ss, "ragas", fake)
    winner = ss.run_ragas(run, ["A7", "B-qwen", "A3", "X", "A0"], "ts", tmp_path / "run.json")
    assert winner == "A7" and asked == ["A0", "A7"]
    assert run["configs"]["A7"]["gates"]["faithfulness_085"] is True


def test_a_failing_top_hands_ragas_to_the_next_at_most_twice_more(tmp_path, a0, monkeypatch):
    run, asked, fake = ragas_run(tmp_path, a0, {"A0": 0.88, "A7": 0.80, "B-qwen": 0.70, "A3": "inconclusive", "X": 0.9})
    monkeypatch.setattr(ss, "ragas", fake)
    assert ss.run_ragas(run, ["A7", "B-qwen", "A3", "X", "A0"], "ts", tmp_path / "run.json") is None
    assert asked == ["A0", "A7", "B-qwen", "A3"]


def test_today_wins_when_it_ranks_first_and_passes_ragas(tmp_path, a0, monkeypatch):
    run, asked, fake = ragas_run(tmp_path, a0, {"A0": 0.88})
    monkeypatch.setattr(ss, "ragas", fake)
    assert ss.run_ragas(run, ["A0", "A7"], "ts", tmp_path / "run.json") == "A0"
    assert asked == ["A0"]


def test_a_finished_ragas_score_is_not_run_again(tmp_path, a0, monkeypatch):
    run, asked, fake = ragas_run(tmp_path, a0, {"A0": 0.88, "A7": 0.86})
    run["configs"]["A0"]["ragas"] = {"faithfulness": 0.88, "answer_relevancy": 0.95, "file": "f"}
    monkeypatch.setattr(ss, "ragas", fake)
    ss.run_ragas(run, ["A7", "A0"], "ts", tmp_path / "run.json")
    assert asked == ["A7"]


# --- configuration files (covers: AC-10, AC-13) ---

def test_each_configuration_gets_its_own_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "CONFIG_PATH", Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    (tmp_path / "w").mkdir()
    settings = ss.config_settings("B-qwen", "qwen2.5vl:3b")
    config = yaml.safe_load(ss.write_config(settings, tmp_path / "corpus", tmp_path / "w").read_text())
    assert config["paths"]["raw_data_dir"] == str(tmp_path / "corpus")
    assert config["paths"]["qdrant_storage"] == str(tmp_path / "w" / "index" / "qdrant_storage")
    assert config["cache"]["enabled"] is False and config["ingestion"]["table_cache"] is False
    assert (config["ingestion"]["table_parser"], config["ingestion"]["figure_parser"]) == ("layoutlmv3", "qwen2.5vl:3b")
    assert config["llm"]["generation_model"] == config["llm"]["rewriter_model"] == "qwen2.5vl:3b"


def test_apply_writes_the_three_model_values_and_keeps_the_comments(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text('ingestion:\n  figure_parser: "none"   # keep me\nllm:\n  generation_model: "llama3.2:3b"\n'
                      '  rewriter_model: "llama3.2:1b"    # rewrites\n')
    monkeypatch.setattr(ss, "CONFIG_PATH", config)
    ss.apply_winner({"figure_parser": "qwen2.5vl:7b", "generation_model": "qwen2.5vl:3b",
                     "rewriter_model": "qwen2.5vl:3b", "table_parser": "layoutlmv3"})
    assert config.read_text() == ('ingestion:\n  figure_parser: "qwen2.5vl:7b"   # keep me\nllm:\n'
                                  '  generation_model: "qwen2.5vl:3b"\n  rewriter_model: "qwen2.5vl:3b"    # rewrites\n')


def test_apply_refuses_a_config_without_the_keys(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  generation_model: llama3.2:3b\n")
    monkeypatch.setattr(ss, "CONFIG_PATH", config)
    with pytest.raises(BenchmarkError, match="figure_parser"):
        ss.apply_winner({"figure_parser": "none", "generation_model": "x", "rewriter_model": "y"})


# --- measurements from the reader log and the latency run (covers: AC-11) ---

def test_reader_seconds_per_pdf_add_detection_and_every_figure_call(tmp_path):
    log = tmp_path / "reader.jsonl"
    lines = [{"file": "a.pdf", "detected_pages": [], "detect_failed_pages": [], "detect_seconds": 10.0, "cached": False},
             {"file": "a.pdf", "figure_candidates": [{"page": 1, "box": [0, 0, 9, 9]}], "candidate_seconds": 1.0},
             {"file": "a.pdf", "page": 1, "box": [0, 0, 9, 9], "reader": "figure:m", "stage": "classify",
              "seconds": 9.0, "outcome": "ok", "kind": "chart"},
             {"file": "a.pdf", "page": 1, "box": [0, 0, 9, 9], "reader": "figure:m", "stage": "extract",
              "seconds": 240.0, "outcome": "timeout"}]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    metrics = ss.ingestion_metrics(log, {"figure_pages": [{"file": "a.pdf", "page": 1, "bbox": [0, 0, 9, 9]}]},
                                   100.0, [{"model": "m", "size": 6, "size_vram": 6}])
    assert metrics["reader_seconds_per_pdf"] == {"a.pdf": 259.0}
    assert metrics["figure_counts"]["timeouts"] == 1 and metrics["figure_recall"] == 1.0
    assert metrics["wall_seconds"] == 100.0 and metrics["peak_during_ingestion"][0]["model"] == "m"


def test_golden_counts_come_from_the_latency_run_first_pass(monkeypatch):
    from src.eval import latency_benchmark
    text = [{"pass": "first", "in_scope": True, "refused": False, "error": None, "id": 1},
            {"pass": "first", "in_scope": True, "refused": True, "error": None, "id": 2},
            {"pass": "first", "in_scope": False, "refused": True, "error": None, "id": 3},
            {"pass": "first", "in_scope": False, "refused": False, "error": None, "id": 4},
            {"pass": "repeat", "in_scope": False, "refused": False, "error": None, "id": 3}]
    summary = {"first": {"answered": {"metrics": {"client_ms": {"p50": 4000, "p95": 6000}}},
                         "refused": {"metrics": {"client_ms": {"p50": 800}}},
                         "voice": {"metrics": {"total_ms": {"p50": 7000}}}}}
    run = {"text": text, "summary": summary, "meta": {"voice_valid": False, "aborted": False}}
    monkeypatch.setattr(latency_benchmark, "Benchmark", lambda **k: type("B", (), {
        "run": lambda self: (ss.REPO_ROOT / "data" / "benchmarks" / "x.json", run)})())
    result = ss.latency(type("Api", (), {"base": "http://x/api"})(), "A0")
    assert result["golden"] == {"answered": 1, "in_scope": 2, "refused": 1, "out_of_scope": 2, "leaked": [4]}
    assert result["latency"]["typed_p50_ms"] == 4000 and result["latency"]["spoken_p50_ms"] is None


# --- the sweep flow (covers: AC-10, AC-13) ---

@pytest.fixture
def offline_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ss, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ss, "load_sets", lambda: ({"questions": []}, {"questions": []}))
    monkeypatch.setattr(ss.tb, "corpus_dir", lambda sets, folder: folder)
    monkeypatch.setattr(ss, "ollama_ps", lambda: [])
    monkeypatch.setattr(ss, "new_run", lambda stamp: {"stamp": stamp, "configs": {}, "notes": []})
    ran = []

    def runner(results):
        def fake(cid, settings, sets, corpus, workdir, copy_from=None):
            ran.append((cid, settings["figure_parser"], copy_from is not None))
            return {**results[cid], "settings": settings, "workdir": f"w/{cid}"}
        monkeypatch.setattr(ss, "run_config", fake)
        return ran
    return runner


def test_a_failing_calibration_stops_the_sweep_with_no_winner(offline_sweep):
    ran = offline_sweep({"A0": record(refused=18)})
    run, path = ss.sweep("all", skip_ragas=True)
    assert [r[0] for r in ran] == ["A0"]
    assert run["winner"] is None and "A0 (today's setup) failed a gate" in run["reason"]
    assert json.loads(path.read_text())["reason"] == run["reason"]


def test_stage_b_copies_the_stage_a_winner_and_reuses_its_record(offline_sweep):
    ran = offline_sweep({"A0": record(), "A3": record(figures=9), "A7": record(figures=5),
                         "B-qwen": record(figures=9, tables=18)})
    run, _ = ss.sweep("all", skip_ragas=True)
    assert ran[-1] == ("B-qwen", "qwen2.5vl:3b", True)
    assert run["stage_a_winner"] == "A3" and run["configs"]["B-llama"]["reused_from"] == "A3"
    assert run["winner"] is None and run["candidate"] in ("A3", "B-qwen")


def test_resume_runs_only_the_configurations_still_missing(offline_sweep, tmp_path):
    ran = offline_sweep({"A3": record(figures=9), "A7": record(figures=5), "B-qwen": record(figures=9)})
    done = record()
    done["gates"] = ss.base_gates(done, done)
    saved = tmp_path / "sweep-old.json"
    saved.write_text(json.dumps({"stamp": "old", "configs": {"A0": done}, "notes": []}))
    run, path = ss.sweep("A", skip_ragas=True, resume=saved)
    assert [r[0] for r in ran] == ["A3", "A7"] and path == saved.resolve()


def test_the_sweep_refuses_to_start_while_ollama_has_models_loaded(offline_sweep, monkeypatch):
    offline_sweep({})
    monkeypatch.setattr(ss, "ollama_ps", lambda: [{"model": "llama3.2:3b", "size": 1, "size_vram": 1}])
    with pytest.raises(BenchmarkError, match="llama3.2:3b"):
        ss.sweep("all", skip_ragas=True)
