"""Table benchmark helpers (spec 0004): value matching, draft checks, the PDF guard."""
import hashlib

import pytest

from src.eval import table_benchmark as tb


@pytest.mark.parametrize("text, value, found", [
    ("The quota is 195 GB per account.", "195 GB", True),
    ("The quota is 195GB per account.", "195 GB", True),
    ("Level P11 offers 1,750 DTUs.", "1750", True),
    ("S3 gives 15 DTUs", "5", False),
    ("Basic tier has 5 DTUs", "5", True),
    ("RPO < 1 hour", "< 1 hour", True),
    ("GPIO 4 maps to GPCLK0", "GPCLK", False),
])
def test_values_match_as_whole_token_sequences(text, value, found):
    assert tb.contains_value(text, value) is found


def test_candidate_checks_inside_and_outside_the_table():
    inside, outside = "S2 50 S3 100", "Standard tier databases"
    assert tb.candidate_problem(["50"], inside, outside) == ""
    assert "not in the table" in tb.candidate_problem(["75"], inside, outside)
    assert "outside" in tb.candidate_problem(["Standard"], inside + " Standard", outside)
    assert tb.candidate_problem([], inside, outside) == "no expected value"


def test_a_changed_pdf_stops_with_its_url(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "a.pdf").write_bytes(b"original")
    data = {"pdf_dir": "pdfs", "documents": [{"file": "a.pdf", "url": "https://example.com/a.pdf",
                                              "sha256": hashlib.sha256(b"original").hexdigest()}]}
    tb.check_pdfs(data)

    (tmp_path / "pdfs" / "a.pdf").write_bytes(b"edited")
    with pytest.raises(tb.BenchmarkError, match="https://example.com/a.pdf"):
        tb.check_pdfs(data)
    (tmp_path / "pdfs" / "a.pdf").unlink()
    with pytest.raises(tb.BenchmarkError, match="Missing"):
        tb.check_pdfs(data)


def test_the_test_set_file_is_well_formed():
    data = tb.load_set()
    files = {d["file"] for d in data["documents"]}
    assert all(set(d) >= {"file", "url", "sha256"} for d in data["documents"])
    assert all(e["file"] in files and len(e["bbox"]) == 4 for e in data["table_pages"])
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids))
    listed = {(e["file"], e["page"]) for e in data["table_pages"]}
    assert all((q["file"], q["page"]) in listed and q["expected"] for q in data["questions"])


# --- scoring and the winner rule ---

Q = {"id": "t1", "question": "?", "expected": ["195 GB"], "file": "a.pdf", "page": 7}


def test_scoring_needs_the_value_outside_citation_tags_and_the_page_in_the_same_file():
    tag = "[Source: a.pdf | Page: 195]"
    right = tb.score_answer(Q, {"answer": "The limit is 195 GB. [Source: a.pdf | Page: 7]",
                                "citations": [{"source": "a.pdf", "page": 7}]})
    tag_only = tb.score_answer({**Q, "expected": ["195"]}, {"answer": f"Not stated. {tag}",
                                                           "citations": [{"source": "a.pdf", "page": 195}]})
    other_file = tb.score_answer(Q, {"answer": "195 GB", "citations": [{"source": "b.pdf", "page": 7}]})

    assert right == {"right_value": True, "right_page": True, "refused": False}
    assert tag_only["right_value"] is False
    assert other_file == {"right_value": True, "right_page": False, "refused": False}


def _reader(score, p50):
    return {"status": "ran", "scores": {"right_value_and_page": score}, "timing": {"p50_s": p50}}


def test_near_ties_go_to_the_fastest_reader():
    readers = {"none": _reader(10, None), "pdfplumber": _reader(19, 0.1),
               "qwen2.5vl:3b": _reader(21, 20.0), "qwen2.5vl:7b": _reader(20, 35.0)}
    winner = tb.pick_winner(readers)
    assert winner["outcome"] == "pdfplumber"
    assert set(winner["near_ties"]) == {"pdfplumber", "qwen2.5vl:3b", "qwen2.5vl:7b"}


def test_a_winner_only_two_above_none_leaves_the_default_at_none():
    winner = tb.pick_winner({"none": _reader(10, None), "qwen2.5vl:3b": _reader(12, 20.0)})
    assert winner["outcome"] == "none" and winner["gain_over_none"] == 2


def test_skipped_readers_never_win():
    readers = {"none": _reader(10, None), "pdfplumber": _reader(14, 0.1),
               "layoutlmv3": {"status": "skipped", "reason": "table_parser_effective is 'none'"}}
    assert tb.pick_winner(readers)["outcome"] == "pdfplumber"


def test_reader_timings_skip_warmup_and_failures():
    calls = [{"outcome": "ok", "seconds": 30.0, "warmup": True}, {"outcome": "ok", "seconds": 2.0},
             {"outcome": "empty", "seconds": 4.0}, {"outcome": "failed", "seconds": 9.0},
             {"outcome": "timeout", "seconds": 120.0}]
    timing = tb.reader_timings(calls)
    assert (timing["p50_s"], timing["max_s"], timing["timed_calls"]) == (2.0, 4.0, 2)


def test_detector_report_counts_recall_and_false_hits():
    data = {"table_pages": [{"file": "a.pdf", "page": 2}, {"file": "a.pdf", "page": 5}, {"file": "a.pdf", "page": 5}]}
    detections = [{"file": "a.pdf", "detected_pages": [2, 9], "detect_failed_pages": [], "detect_seconds": 1.5}]
    report = tb.detector_report(detections, data)
    assert report["recall"] == 0.5 and report["missed"] == [["a.pdf", 5]] and report["false_hits"] == [["a.pdf", 9]]


# --- /test pass 2026-10-08: draft, run plumbing and winner edges (spec 0004) ---

from pathlib import Path

from tests.page_citation_test import write_pdf


@pytest.fixture
def table_set(tmp_path, monkeypatch):
    """A tiny test set on a real one page PDF, with REPO_ROOT pointed at tmp_path."""
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    (tmp_path / "pdfs").mkdir()
    write_pdf(tmp_path / "pdfs" / "a.pdf", ["Tier Basic size 2 GB and Premium size 500 GB."])
    sha = hashlib.sha256((tmp_path / "pdfs" / "a.pdf").read_bytes()).hexdigest()
    return {"pdf_dir": "pdfs", "documents": [{"file": "a.pdf", "url": "https://example.com/a.pdf", "sha256": sha}],
            "table_pages": [{"file": "a.pdf", "page": 1, "bbox": [0, 0, 612, 792]}],
            "questions": [{"id": "t001", "question": "Old?", "expected": ["2 GB"], "file": "a.pdf", "page": 1,
                           "reviewed": True}]}


# draft (covers: AC-1)

def test_draft_appends_only_new_unreviewed_candidates_whose_values_sit_in_the_table(table_set, monkeypatch):
    monkeypatch.setattr(tb, "require_model", lambda model: None)
    monkeypatch.setattr(tb, "ask_model", lambda page_text, table_text: [
        {"question": "What size is Premium?", "expected": ["500 GB"]},
        {"question": "old?", "expected": ["2 GB"]},                  # duplicate of t001 (case ignored)
        {"question": "What is the Ultra size?", "expected": ["9 TB"]},   # not in the table
        {"question": "", "expected": ["2 GB"]},
    ])
    data = tb.draft(table_set)
    new = [q for q in data["questions"] if q["id"] != "t001"]
    assert new == [{"id": "t002", "question": "What size is Premium?", "expected": ["500 GB"], "file": "a.pdf",
                    "page": 1, "reviewed": False}]


def test_draft_stops_before_asking_when_the_model_is_missing(table_set, monkeypatch):
    def missing(model):
        raise tb.BenchmarkError("Ollama model llama3.1:8b is not pulled. Run `ollama pull llama3.1:8b`.")
    monkeypatch.setattr(tb, "require_model", missing)
    monkeypatch.setattr(tb, "ask_model", lambda *a: pytest.fail("asked the model"))
    with pytest.raises(tb.BenchmarkError, match="ollama pull"):
        tb.draft(table_set)


def test_fill_boxes_adds_one_entry_per_detected_table_and_keeps_given_boxes(table_set, monkeypatch, capsys):
    from src.ingestion import tables
    boxes = [{"x0": 10.0, "top": 20.0, "x1": 300.0, "bottom": 100.0}, {"x0": 10.0, "top": 400.0, "x1": 300.0, "bottom": 500.0}]
    monkeypatch.setattr(tables, "get_detector", lambda: type("D", (), {"detect": lambda self, a, b: boxes})())
    table_set["table_pages"].append({"file": "a.pdf", "page": 1})
    assert tb.fill_boxes(table_set) == 2
    assert table_set["table_pages"][0]["bbox"] == [0, 0, 612, 792]
    assert [e["bbox"] for e in table_set["table_pages"][1:]] == [[10.0, 20.0, 300.0, 100.0], [10.0, 400.0, 300.0, 500.0]]
    assert all(e["bbox_source"] == "detector" for e in table_set["table_pages"][1:])


def test_fill_boxes_keeps_a_page_the_detector_misses_and_says_so(table_set, monkeypatch, capsys):
    from src.ingestion import tables
    monkeypatch.setattr(tables, "get_detector", lambda: type("D", (), {"detect": lambda self, a, b: []})())
    table_set["table_pages"] = [{"file": "a.pdf", "page": 1}]
    assert tb.fill_boxes(table_set) == 0
    assert table_set["table_pages"] == [{"file": "a.pdf", "page": 1}]
    assert "add its bbox by hand" in capsys.readouterr().out


def test_ask_model_treats_a_broken_json_reply_as_no_candidates(monkeypatch):
    import ollama
    reply = type("R", (), {"message": type("M", (), {"content": "not json"})()})()
    monkeypatch.setattr(ollama, "chat", lambda **kwargs: reply)
    assert tb.ask_model("page", "table") == []


# run plumbing (covers: AC-5)

def test_the_reader_config_isolates_the_run(table_set, tmp_path, monkeypatch):
    import yaml
    monkeypatch.setattr(tb, "CONFIG_PATH", Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    config = yaml.safe_load(tb.reader_config(table_set, "qwen2.5vl:7b", tmp_path).read_text())
    assert config["paths"]["raw_data_dir"] == str(tmp_path / "pdfs")
    assert config["paths"]["qdrant_storage"].startswith(str(tmp_path))
    assert config["cache"]["enabled"] is False
    assert config["ingestion"]["table_parser"] == "qwen2.5vl:7b" and config["ingestion"]["table_cache"] is False


def test_a_reader_that_fell_back_to_none_is_skipped_not_scored(table_set, monkeypatch):
    class Proc:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0
    monkeypatch.setattr(tb.subprocess, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(tb.requests, "get", lambda *a, **k: type("H", (), {"json": lambda self: {"table_parser_effective": "none"}})())
    monkeypatch.setattr(tb.requests, "post", lambda *a, **k: pytest.fail("asked a question"))
    result = tb.run_reader(table_set, "layoutlmv3", table_set["questions"])
    assert result["status"] == "skipped" and "'none', not 'layoutlmv3'" in result["reason"]


def test_run_refuses_a_set_without_reviewed_questions(table_set):
    table_set["questions"][0]["reviewed"] = False
    with pytest.raises(tb.BenchmarkError, match="No reviewed questions"):
        tb.run(table_set, ["none"])


def test_a_refusal_needs_no_citations_and_refusal_wording():
    refused = tb.score_answer(Q, {"answer": "Insufficient context.", "citations": []})
    cited = tb.score_answer(Q, {"answer": "Insufficient context.", "citations": [{"source": "a.pdf", "page": 7}]})
    assert refused["refused"] is True and cited["refused"] is False


def test_summaries_count_each_outcome_once():
    results = [{"right_value": True, "right_page": True, "refused": False},
               {"right_value": True, "right_page": False, "refused": False},
               {"right_value": False, "right_page": True, "refused": False},
               {"right_value": False, "right_page": False, "refused": True},
               {"right_value": False, "right_page": False, "refused": False, "error": "timeout"}]
    assert tb.summarize(results) == {"right_value_and_page": 1, "right_value_only": 1, "right_page_only": 1,
                                     "refusals": 1, "errors": 1, "questions": 5}


def test_the_table_log_splits_reader_calls_from_detection_lines(tmp_path):
    log = tmp_path / "t.jsonl"
    log.write_text('{"file": "a.pdf", "detected_pages": [1], "detect_failed_pages": [], "detect_seconds": 1.0}\n'
                   '{"file": "a.pdf", "page": 1, "reader": "pdfplumber", "seconds": 0.1, "outcome": "ok"}\n')
    calls, detections, _ = tb.read_table_log(log)
    assert len(calls) == 1 and len(detections) == 1
    assert tb.read_table_log(tmp_path / "missing.jsonl") == ([], [], [])


# winner edges (covers: AC-6)

def test_without_the_none_baseline_the_outcome_is_undecided():
    winner = tb.pick_winner({"pdfplumber": _reader(15, 0.1), "none": {"status": "skipped", "reason": "x"}})
    assert winner["outcome"] == "undecided" and winner["candidate"] == "pdfplumber"


def test_with_no_contenders_the_default_stays_none():
    assert tb.pick_winner({"none": _reader(9, None)})["outcome"] == "none"


def test_a_tied_reader_without_timings_loses_the_tie_break():
    readers = {"none": _reader(5, None), "qwen2.5vl:3b": _reader(15, None), "qwen2.5vl:7b": _reader(14, 20.0)}
    assert tb.pick_winner(readers)["outcome"] == "qwen2.5vl:7b"


def test_exactly_three_above_none_is_enough():
    assert tb.pick_winner({"none": _reader(9, None), "pdfplumber": _reader(12, 0.1)})["outcome"] == "pdfplumber"


# Shared with the figure benchmark and the sweep (spec 0005)

def test_a_document_with_its_own_folder_resolves_there(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    data = {"pdf_dir": "tables", "documents": [{"file": "a.pdf", "dir": "figures"}, {"file": "b.pdf"}]}
    assert tb.pdf_path(data, "a.pdf") == tmp_path / "figures" / "a.pdf"
    assert tb.pdf_path(data, "b.pdf") == tmp_path / "tables" / "b.pdf"


def test_the_corpus_folder_links_every_listed_pdf_once(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    for folder, name in (("t", "azure.pdf"), ("f", "census.pdf")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_bytes(b"%PDF")
    tables_set = {"pdf_dir": "t", "documents": [{"file": "azure.pdf", "sha256": "s1"}]}
    figures_set = {"documents": [{"file": "census.pdf", "dir": "f", "sha256": "s2"},
                                 {"file": "azure.pdf", "dir": "t", "sha256": "s1"}]}
    corpus = tb.corpus_dir([tables_set, figures_set], tmp_path / "corpus")
    assert sorted(p.name for p in corpus.iterdir()) == ["azure.pdf", "census.pdf"]
    assert (corpus / "census.pdf").resolve() == (tmp_path / "f" / "census.pdf").resolve()


def test_the_same_pdf_listed_with_two_hashes_stops_the_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    sets = [{"pdf_dir": "t", "documents": [{"file": "azure.pdf", "sha256": "s1"}]},
            {"documents": [{"file": "azure.pdf", "dir": "t", "sha256": "other"}]}]
    with pytest.raises(tb.BenchmarkError, match="two different SHA256"):
        tb.corpus_dir(sets, tmp_path / "corpus")


def test_figure_lines_go_to_their_own_bucket(tmp_path):
    log = tmp_path / "t.jsonl"
    log.write_text('{"file": "a.pdf", "page": 1, "reader": "layoutlmv3", "seconds": 0.1, "outcome": "ok"}\n'
                   '{"file": "a.pdf", "figure_candidates": [], "candidate_seconds": 0.0}\n'
                   '{"file": "a.pdf", "page": 23, "figure_skip": "dense", "drawn_objects": 6000}\n'
                   '{"file": "a.pdf", "page": 2, "box": [0, 0, 1, 1], "reader": "figure:qwen2.5vl:7b", '
                   '"stage": "classify", "seconds": 9.0, "outcome": "empty", "kind": "photo"}\n')
    calls, detections, figure_lines = tb.read_table_log(log)
    assert [c["reader"] for c in calls] == ["layoutlmv3"] and detections == []
    assert len(figure_lines) == 3


def test_serve_and_ask_skips_when_health_differs_from_what_was_asked(tmp_path, monkeypatch):
    class Proc:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0
    monkeypatch.setattr(tb.subprocess, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(tb.requests, "get", lambda *a, **k: type("H", (), {"json": lambda self: {
        "table_parser_effective": "layoutlmv3", "figure_parser_effective": "none"}})())
    monkeypatch.setattr(tb.requests, "post", lambda *a, **k: pytest.fail("asked a question"))
    result = tb.serve_and_ask(tmp_path / "config.yaml", tmp_path, {"figure_parser_effective": "qwen2.5vl:7b"}, [], "x")
    assert result["status"] == "skipped" and "'none', not 'qwen2.5vl:7b'" in result["reason"]


def test_ask_all_scores_each_answer_and_records_errors(monkeypatch):
    import requests
    question = {"id": "f001", "question": "Units in 2020?", "expected": ["15"], "file": "r.pdf", "page": 3}
    replies = iter([type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: {
        "answer": "15 units [Source: r.pdf | Page: 3]", "citations": [{"source": "r.pdf", "page": 3}]}})(),
        requests.ConnectionError("down")])

    def post(*a, **k):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(tb.requests, "post", post)
    first, second = tb.ask_all("http://x/api", [question, dict(question, id="f002")], "test")
    assert (first["right_value"], first["right_page"], first["cited"]) == (True, True, [["r.pdf", 3]])
    assert second["error"] == "down" and second["right_value"] is False
