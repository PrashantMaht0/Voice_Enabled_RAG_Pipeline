"""Figure benchmark: timings, recall, the draft checks and the run plumbing. No API or Ollama runs."""
import pytest
import yaml

from src.eval import figure_benchmark as fb
from src.eval import table_benchmark as tb

BOX = [100.0, 100.0, 300.0, 300.0]


def call(stage, seconds, outcome, box=BOX, page=1, **extra):
    return {"file": "r.pdf", "page": page, "box": box, "reader": "figure:qwen2.5vl:7b", "stage": stage,
            "seconds": seconds, "outcome": outcome, **extra}


# --- seconds per figure and outcome counts ---

def test_extract_seconds_add_up_every_attempt_for_a_figure():
    lines = [call("classify", 9.0, "ok", kind="chart"), call("extract", 40.0, "retry"), call("extract", 20.0, "ok")]
    timing = fb.figure_timings(lines)
    assert timing["extract"] == {"p50_s": 60.0, "max_s": 60.0, "timed": 1}
    assert timing["classify"]["p50_s"] == 9.0 and timing["retries"] == 1


def test_warm_up_calls_are_left_out_of_the_timings():
    lines = [call("classify", 30.0, "ok", kind="chart", warmup=True), call("extract", 50.0, "ok"),
             call("classify", 8.0, "empty", box=[1, 1, 2, 2], kind="photo")]
    timing = fb.figure_timings(lines)
    assert timing["classify"] == {"p50_s": 8.0, "max_s": 8.0, "timed": 1}
    assert timing["extract"]["p50_s"] == 50.0


def test_timeouts_and_errors_are_counted_but_not_timed():
    lines = [call("classify", 9.0, "ok", kind="chart"), call("extract", 240.0, "timeout"),
             call("classify", 1.0, "error", box=[1, 1, 2, 2])]
    timing = fb.figure_timings(lines)
    assert (timing["timeouts"], timing["errors"]) == (1, 1)
    assert timing["extract"]["timed"] == 0 and timing["classify"]["timed"] == 0


def test_final_failures_and_missed_tables_are_counted():
    lines = [call("classify", 9.0, "ok", kind="chart"), call("extract", 30.0, "retry"),
             call("extract", 15.0, "failed"), call("classify", 9.0, "empty", box=[1, 1, 2, 2], kind="table")]
    timing = fb.figure_timings(lines)
    assert timing["failed"] == 1 and timing["missed_tables"] == 1 and timing["classified_figures"] == 1


def test_no_figure_lines_give_empty_timings():
    timing = fb.figure_timings([])
    assert timing["classify"] == {"p50_s": None, "max_s": None, "timed": 0} and timing["failed"] == 0


# --- recall and classification against the listed figures ---

def candidates_line(*boxes, page=1):
    return {"file": "r.pdf", "figure_candidates": [{"page": page, "box": b} for b in boxes], "candidate_seconds": 1}


def test_a_listed_figure_is_found_when_a_candidate_overlaps_it_enough():
    listed = {"figure_pages": [{"file": "r.pdf", "page": 1, "bbox": BOX},
                               {"file": "r.pdf", "page": 1, "bbox": [400, 400, 500, 500]},
                               {"file": "r.pdf", "page": 2, "bbox": BOX}]}
    lines = [candidates_line([110, 110, 310, 310], [400, 400, 420, 420]),
             call("classify", 9.0, "ok", box=[92, 92, 328, 328], kind="chart")]
    report = fb.figure_recall(lines, listed)
    assert (report["listed"], report["found"], report["recall"]) == (3, 1, 0.333)
    assert report["classified_chart_or_diagram"] == 1
    assert {(m["page"], m["found"]) for m in report["missed"]} == {(1, False), (2, False)}


def test_a_found_figure_classified_as_a_table_is_a_miss():
    listed = {"figure_pages": [{"file": "r.pdf", "page": 1, "bbox": BOX}]}
    lines = [candidates_line(BOX), call("classify", 9.0, "empty", kind="table")]
    report = fb.figure_recall(lines, listed)
    assert report["found"] == 1 and report["classified_chart_or_diagram"] == 0
    assert report["missed"] == [{"file": "r.pdf", "page": 1, "found": True, "kind": "table"}]


@pytest.mark.parametrize("a, b, expected", [([0, 0, 10, 10], [0, 0, 10, 10], 1.0), ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),
                                            ([0, 0, 10, 10], [5, 0, 15, 10], 1 / 3)])
def test_iou_of_two_boxes(a, b, expected):
    assert fb._iou(a, b) == pytest.approx(expected)


# --- drafted question checks ---

def test_a_value_printed_outside_the_figure_is_rejected():
    inside = "Units 2019 2020 12 15 sold per year".split()
    assert fb.candidate_problem(["15"], inside, "Sales were 15 units last year") == "'15' also appears outside the figure"


def test_a_value_missing_from_the_figure_words_is_rejected():
    inside = "Units 2019 2020 12 15 sold per year".split()
    assert "not among the figure's words" in fb.candidate_problem(["99"], inside, "")


def test_a_figure_without_a_text_layer_accepts_values_read_from_the_image():
    assert fb.candidate_problem(["99"], ["Units"], "") == ""


@pytest.mark.parametrize("expected", [[], [""], [None]])
def test_a_draft_without_an_expected_value_is_rejected(expected):
    assert fb.candidate_problem(expected, [], "") == "no expected value"


def test_the_first_draft_pass_lists_candidates_from_the_chart_reports_only(monkeypatch):
    data = {"documents": [{"file": "census.pdf", "dir": "data/eval/figure_pdfs"},
                          {"file": "azure.pdf", "dir": "data/eval/table_pdfs"}], "figure_pages": [], "questions": []}
    asked = []
    monkeypatch.setattr(fb, "page_candidates", lambda d, f: asked.append(f) or {
        3: [{"box": BOX, "caption": "Figure 1."}]})
    assert fb.fill_boxes(data) == 1
    assert asked == ["census.pdf"]
    assert data["figure_pages"] == [{"file": "census.pdf", "page": 3, "bbox": BOX, "bbox_source": "candidates",
                                     "caption": "Figure 1."}]


def test_a_listed_page_without_a_box_gets_one_per_candidate_and_kept_boxes_stay(monkeypatch):
    kept = {"file": "a.pdf", "page": 1, "bbox": BOX}
    data = {"documents": [], "figure_pages": [kept, {"file": "a.pdf", "page": 5}], "questions": []}
    monkeypatch.setattr(fb, "page_candidates", lambda d, f: {5: [{"box": [1, 2, 3, 4], "caption": None},
                                                                  {"box": [5, 6, 7, 8], "caption": None}]})
    assert fb.fill_boxes(data) == 2
    assert data["figure_pages"][0] == kept and [e["bbox"] for e in data["figure_pages"][1:]] == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_crop_names_identify_the_file_page_and_figure():
    assert fb.crop_name("The Report.pdf", 12, 2) == "The Report-p12-2.png"


# --- run plumbing ---

@pytest.fixture
def figure_set(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "r.pdf").write_bytes(b"%PDF")
    return {"documents": [{"file": "r.pdf", "dir": "pdfs", "url": "https://example.com/r.pdf", "sha256": "x"}],
            "figure_pages": [{"file": "r.pdf", "page": 1, "bbox": BOX}],
            "questions": [{"id": "f001", "question": "How many units in 2020?", "expected": ["15"], "file": "r.pdf",
                           "page": 1, "reviewed": True}]}


def test_the_figure_config_isolates_the_run(figure_set, tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(fb, "CONFIG_PATH", Path(__file__).resolve().parents[2] / "config" / "config.yaml")
    config = yaml.safe_load(fb.figure_config(figure_set, "qwen2.5vl:3b", tmp_path).read_text())
    assert (tmp_path / "corpus" / "r.pdf").is_symlink()
    assert config["paths"]["raw_data_dir"] == str(tmp_path / "corpus")
    assert config["paths"]["qdrant_storage"].startswith(str(tmp_path))
    assert config["cache"]["enabled"] is False and config["ingestion"]["table_cache"] is False
    assert config["ingestion"]["figure_parser"] == "qwen2.5vl:3b"


def test_a_reader_that_fell_back_is_reported_not_scored(figure_set, monkeypatch):
    monkeypatch.setattr(fb, "figure_config", lambda data, reader, workdir: workdir / "config.yaml")
    monkeypatch.setattr(fb, "serve_and_ask", lambda *a, **k: {"status": "skipped",
                                                               "reason": "figure_parser_effective is 'none'"})
    result = fb.run_reader(figure_set, "qwen2.5vl:7b", figure_set["questions"])
    assert result == {"status": "skipped", "reason": "figure_parser_effective is 'none'"}


def test_a_ran_reader_gets_timings_and_recall_from_its_log(figure_set, tmp_path, monkeypatch):
    log = tmp_path / "reader.jsonl"
    import json
    log.write_text("\n".join(json.dumps(line) for line in [
        candidates_line(BOX), call("classify", 9.0, "ok", kind="chart"), call("extract", 20.0, "ok")]) + "\n")
    expected = {}
    monkeypatch.setattr(fb, "figure_config", lambda data, reader, workdir: workdir / "config.yaml")

    def fake_serve(config, workdir, expect, questions, label, timeout):
        expected.update(expect)
        return {"status": "ran", "ingest_seconds": 12.0, "scores": {"right_value_and_page": 1}, "results": [],
                "table_log": log}
    monkeypatch.setattr(fb, "serve_and_ask", fake_serve)
    result = fb.run_reader(figure_set, "qwen2.5vl:7b", figure_set["questions"])
    assert expected == {"figure_parser_effective": "qwen2.5vl:7b"}
    assert result["timing"]["extract"]["p50_s"] == 20.0 and result["recall"]["recall"] == 1.0
    assert "table_log" not in result


def test_run_refuses_a_set_without_reviewed_questions(figure_set, monkeypatch):
    monkeypatch.setattr(fb, "check_pdfs", lambda data: None)
    figure_set["questions"][0]["reviewed"] = False
    with pytest.raises(tb.BenchmarkError, match="No reviewed questions"):
        fb.run(figure_set, ["none"])
