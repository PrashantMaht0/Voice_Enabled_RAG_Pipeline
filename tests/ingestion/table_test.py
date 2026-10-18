"""Tables in PDFs: chunks, gating, failures and the startup check. The detector is mocked."""
import json

import pytest

from src.api import routes
from src.ingestion import tables
from src.ingestion.chunking import DocumentChunker
from src.ingestion.loader import DocumentLoader
from tests.helpers import FakeManager, use_corpus, write_pdf

TABLE = "| Tier | Size |\n|---|---|\n| Basic | 2 GB |"


class FakeDetector:
    """Flags the given pages with one full width box."""
    settings = {"model": "fake", "revision": "0", "threshold": 0.99, "render_px": 1000}

    def __init__(self, pages=()):
        self.pages = set(pages)

    def detect(self, pdfium_page, plumber_page):
        tables.detector_calls += 1
        number = plumber_page.page_number
        if number not in self.pages:
            return []
        return [{"x0": 50.0, "top": 100.0, "x1": 550.0, "bottom": 300.0, "label": "table", "score": 1.0}]


@pytest.fixture(autouse=True)
def fresh_counters():
    tables.reset_counters()


@pytest.fixture
def pdf_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_pdf(raw / "guide.pdf", ["Blob storage holds files and folders.", "Queues carry messages between apps."])
    return raw


def load(raw, extractor):
    return DocumentChunker().chunk_documents(DocumentLoader(raw_data_dir=str(raw), table_extractor=extractor)
                                             .load_documents())


# --- VLM output rules ---

def test_vlm_reply_with_code_fence_and_preamble_keeps_only_the_table():
    reply = "Here is the table:\n\n```markdown\n" + TABLE + "\n```"
    assert tables.parse_vlm_reply(reply) == [TABLE]


@pytest.mark.parametrize("reply", ["None.", "NONE", "", "no table here"])
def test_vlm_reply_without_a_table_is_empty(reply):
    assert tables.parse_vlm_reply(reply) == []


def test_ragged_and_looping_tables_are_dropped():
    ragged = "| a | b |\n|---|---|\n| 1 | 2 | 3 |"
    looping = "| a | b |\n|---|---|\n" + "| 1 | 2 |\n" * 7
    assert tables.parse_vlm_reply(ragged) == []
    assert tables.parse_vlm_reply(looping) == []


def test_two_tables_separated_by_a_blank_line():
    assert tables.parse_vlm_reply(TABLE + "\n\n" + TABLE) == [TABLE, TABLE]


# --- Markdown and chunks ---

def test_rows_to_markdown_drops_empty_rows_and_columns_and_escapes_pipes():
    rows = [["Tier", None, "Size"], ["", "", ""], ["Basic", "", "2\nGB"], ["a|b", None, "1"]]
    assert tables.rows_to_markdown(rows) == "| Tier | Size |\n|---|---|\n| Basic | 2 GB |\n| a\\|b | 1 |"
    assert tables.rows_to_markdown([["only a header"]]) is None


def test_long_table_splits_under_400_tokens_each_with_the_header():
    chunker = DocumentChunker()
    rows = [["Name", "Value"]] + [[f"setting number {i} with a longer label", f"{i} GB"] for i in range(60)]
    parts = chunker.split_table(tables.rows_to_markdown(rows))
    assert len(parts) > 1
    for part in parts:
        assert part.startswith("| Name | Value |\n|---|---|")
        assert len(chunker.tokenizer.encode(part)) <= tables.MAX_TABLE_TOKENS
    assert sum(part.count("GB") for part in parts) == 60


def test_table_chunks_come_after_text_and_text_ids_do_not_change():
    doc = {"content": "x", "metadata": {"source": "a.pdf", "doc_hash": "h" * 64},
           "pages": [(1, "Blob storage holds files."), (2, "Queues carry messages.")]}
    plain = DocumentChunker().chunk_documents([dict(doc)])
    with_tables = DocumentChunker().chunk_documents([dict(doc, tables=[(1, 100.0, TABLE)], table_parser="pdfplumber")])

    assert with_tables[:len(plain)] == plain
    assert all(c["metadata"]["kind"] == "text" for c in plain)
    table = with_tables[-1]
    assert table["content"] == f"[Source: a.pdf | Page: 1]\n{TABLE}"
    assert table["metadata"] == {"source": "a.pdf", "doc_hash": "h" * 64, "chunk_index": 2, "page": 1,
                                 "kind": "table", "table_parser": "pdfplumber"}


# --- gating, failures and the reader log ---

def test_none_makes_no_detector_or_reader_calls(pdf_dir):
    chunks = load(pdf_dir, tables.TableExtractor(parser="none", detector=FakeDetector([1])))
    assert tables.detector_calls == 0 and tables.reader_calls == 0
    assert all(c["metadata"]["kind"] == "text" for c in chunks)


def test_a_table_free_pdf_makes_zero_reader_calls_and_matches_none(pdf_dir):
    none = load(pdf_dir, tables.TableExtractor(parser="none"))
    on = load(pdf_dir, tables.TableExtractor(parser="pdfplumber", detector=FakeDetector(), use_cache=False))
    assert tables.detector_calls == 2 and tables.reader_calls == 0
    assert on == none


def test_a_failing_reader_keeps_the_text_and_logs_failed(pdf_dir, tmp_path, monkeypatch):
    log = tmp_path / "tables.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))

    def broken(parser, page):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(tables, "read_page", broken)

    none = load(pdf_dir, tables.TableExtractor(parser="none"))
    on = load(pdf_dir, tables.TableExtractor(parser="pdfplumber", detector=FakeDetector([2]), use_cache=False))

    assert on == none
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[0]["detected_pages"] == [2]
    assert lines[1]["page"] == 2 and lines[1]["outcome"] == "failed"


def test_detection_is_cached_per_file_and_page(pdf_dir, tmp_path):
    extractor = tables.TableExtractor(parser="pdfplumber", detector=FakeDetector([1]),
                                      cache_dir=str(tmp_path / "cache"), use_cache=True)
    load(pdf_dir, extractor)
    first = tables.detector_calls
    load(pdf_dir, extractor)
    assert first == 2 and tables.detector_calls == first


# --- geometry ---

def test_pixels_to_points_respects_the_page_origin():
    box = tables.pixels_to_points((100, 200, 300, 400), (10, 20, 622, 812), 612, 792, (612 * 2, 792 * 2))
    assert box == (60, 120, 160, 220)


def test_overlapping_boxes_merge_into_their_union():
    a = {"x0": 0, "top": 0, "x1": 100, "bottom": 100, "label": "table", "score": 0.99}
    b = {"x0": 10, "top": 10, "x1": 110, "bottom": 110, "label": "table", "score": 1.0}
    far = {"x0": 0, "top": 500, "x1": 100, "bottom": 600, "label": "table", "score": 1.0}
    merged = tables.merge_boxes([a, b, far])
    assert [(m["x0"], m["top"], m["x1"], m["bottom"]) for m in merged] == [(0, 0, 110, 110), (0, 500, 100, 600)]


# --- startup check and the fingerprint rebuild ---

def test_a_missing_vlm_is_one_error_and_falls_back_to_none(monkeypatch, capsys):
    monkeypatch.setattr(tables, "_effective_parser", None)   # restored after the test
    monkeypatch.setattr(tables, "configured_parser", lambda: "qwen2.5vl:7b")
    monkeypatch.setattr(tables, "get_detector", lambda: type("D", (), {"load": lambda self: self})())

    def missing(model):
        raise RuntimeError(f"Ollama model {model!r} is not pulled")
    monkeypatch.setattr(tables, "vlm_digest", missing)

    assert tables.check_table_parser() == "none"
    assert tables.table_fingerprint() == "none"
    assert capsys.readouterr().out.count("ERROR") == 1


def test_changing_the_table_settings_rebuilds_once(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    use_corpus(monkeypatch, tmp_path / "empty")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr(routes, "get_retriever", lambda: type("R", (), {"store": type("S", (), {"db_manager": manager})})())
    manager.mark_index_format("none")

    monkeypatch.setattr(tables, "table_fingerprint", lambda parser=None: "pdfplumber|fake|0|0.99")
    routes.ensure_index_format()
    routes.ensure_index_format()

    assert len(manager.ingested) == 1
    assert manager.index_tables() == "pdfplumber|fake|0|0.99"


# --- reranker input (bug: Markdown pipes sank correct table chunks under the confidence gate) ---

def test_the_reranker_reads_tables_without_markdown_syntax():
    from src.pipeline.retrieval import rerank_text
    chunk = "[Source: a.pdf | Page: 3]\n| Tier | DTUs |\n|---|:---:|\n| P11 | 1,750 |\nplain text stays"
    assert rerank_text(chunk) == "[Source: a.pdf | Page: 3]\nTier DTUs\nP11 1,750\nplain text stays"


def test_the_reranker_scores_plain_text_but_returns_the_markdown_chunk():
    from src.retrieval.reranker import CrossEncoderReranker
    seen = []

    class FakeRanker:
        def rerank(self, request):
            seen.extend(p["text"] for p in request.passages)
            return [{"id": p["id"], "text": p["text"], "score": 0.9 - int(p["id"]) / 10} for p in request.passages]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.ranker = FakeRanker()
    candidates = [{"id": "c1", "content": TABLE, "metadata": {"kind": "table"}},
                  {"id": "c2", "content": "Blob storage holds files.", "metadata": {"kind": "text"}}]
    ranked = reranker.rerank("tier size", candidates)

    assert "|" not in seen[0]
    assert [r["chunk"] for r in ranked] == [TABLE, "Blob storage holds files."]
    assert ranked[0]["metadata"] == {"kind": "table"}


# --- edges, errors and the contract ---

# VLM output rules

def test_vlm_tables_with_aligned_separators_and_escaped_pipes_are_kept():
    table = "| Pin | Function |\n|:---|---:|\n| 4 | GPCLK0 \\| alt |"
    assert tables.parse_vlm_reply(table) == [table]


def test_a_vlm_block_without_a_separator_row_is_not_a_table():
    assert tables.parse_vlm_reply("| Tier | Size |\n| Basic | 2 GB |") == []


def test_a_reply_starting_with_none_in_any_case_is_empty_even_if_a_table_follows():
    assert tables.parse_vlm_reply("none of these are tables\n\n" + TABLE) == []


# Markdown and splitting

def test_rows_to_markdown_returns_none_when_every_cell_is_empty():
    assert tables.rows_to_markdown([[None, ""], ["", None]]) is None
    assert tables.rows_to_markdown([]) is None


def test_ragged_pdfplumber_rows_are_padded_to_the_widest_row():
    assert tables.rows_to_markdown([["a", "b", "c"], ["1"]]) == "| a | b | c |\n|---|---|---|\n| 1 |  |  |"


def test_a_single_row_larger_than_the_limit_becomes_its_own_chunk_with_the_header():
    chunker = DocumentChunker()
    huge = "| " + " ".join(["word"] * 500) + " |"
    markdown = "| Name |\n|---|\n| small |\n" + huge + "\n| after |"
    parts = chunker.split_table(markdown)
    assert parts == ["| Name |\n|---|\n| small |", "| Name |\n|---|\n" + huge, "| Name |\n|---|\n| after |"]


def test_a_header_only_table_is_one_chunk():
    assert DocumentChunker().split_table("| Name |\n|---|") == ["| Name |\n|---|"]


def test_text_and_markdown_files_get_kind_text():
    chunks = DocumentChunker().chunk_documents([{"content": "Queues carry messages.",
                                                 "metadata": {"source": "notes.md", "doc_hash": "a" * 64}}])
    assert chunks[0]["metadata"]["kind"] == "text"


def test_tables_are_numbered_after_all_text_in_page_then_box_order():
    doc = {"content": "x", "metadata": {"source": "a.pdf", "doc_hash": "b" * 64},
           "pages": [(1, "Page one text here."), (2, "Page two text here.")],
           "tables": [(1, 50.0, TABLE), (1, 300.0, TABLE.replace("Basic", "Premium")), (2, 10.0, TABLE)],
           "table_parser": "qwen2.5vl:7b"}
    chunks = DocumentChunker().chunk_documents([doc])
    table_chunks = [c for c in chunks if c["metadata"]["kind"] == "table"]
    assert [c["metadata"]["chunk_index"] for c in table_chunks] == [2, 3, 4]
    assert [c["metadata"]["page"] for c in table_chunks] == [1, 1, 2]
    assert "Premium" in table_chunks[1]["content"]
    assert len({c["id"] for c in chunks}) == len(chunks)


# Text layer and loader

def test_a_page_needs_twenty_visible_characters_to_count_as_text(tmp_path):
    write_pdf(tmp_path / "t.pdf", ["abcdefghij klmnopqrs", "abcdefghij klmnopqrst", ""])   # 19 and 20 visible
    pages = DocumentLoader()._load_pdf_pages(str(tmp_path / "t.pdf"))
    assert [number for number, _ in pages] == [2]


def test_a_crashing_table_pass_keeps_the_document_text(pdf_dir, capsys):
    class Broken(tables.TableExtractor):
        def extract(self, file_path, text_pages):
            raise RuntimeError("pdfium exploded")

    broken = load(pdf_dir, Broken(parser="pdfplumber", detector=FakeDetector(), use_cache=False))
    plain = load(pdf_dir, tables.TableExtractor(parser="none"))
    assert broken == plain
    assert "Table extraction failed" in capsys.readouterr().out


def test_readers_only_see_pages_the_detector_flags(pdf_dir, monkeypatch):
    seen = []
    monkeypatch.setattr(tables, "read_page", lambda parser, page: seen.append(page["number"]) or [(0.0, TABLE)])
    chunks = load(pdf_dir, tables.TableExtractor(parser="pdfplumber", detector=FakeDetector([2]), use_cache=False))
    assert seen == [2] and tables.reader_calls == 1
    assert [c["metadata"]["page"] for c in chunks if c["metadata"]["kind"] == "table"] == [2]


# Failures per page

def test_a_reader_timeout_is_logged_as_timeout(pdf_dir, tmp_path, monkeypatch):
    import httpx
    log = tmp_path / "t.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))

    def slow(parser, page):
        raise httpx.ReadTimeout("timed out")
    monkeypatch.setattr(tables, "read_page", slow)
    load(pdf_dir, tables.TableExtractor(parser="qwen2.5vl:3b", detector=FakeDetector([1]), use_cache=False))
    calls = [json.loads(line) for line in log.read_text().splitlines() if "outcome" in line]
    assert [c["outcome"] for c in calls] == ["timeout"]


def test_a_detector_error_on_one_page_marks_it_failed_and_keeps_going(pdf_dir, tmp_path, monkeypatch):
    log = tmp_path / "t.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))

    class FlakyDetector(FakeDetector):
        def detect(self, pdfium_page, plumber_page):
            if plumber_page.page_number == 1:
                raise RuntimeError("render failed")
            return super().detect(pdfium_page, plumber_page)

    chunks = load(pdf_dir, tables.TableExtractor(parser="pdfplumber", detector=FlakyDetector([2]), use_cache=False))
    detection = json.loads(log.read_text().splitlines()[0])
    assert detection["detect_failed_pages"] == [1] and detection["detected_pages"] == [2]
    assert sum(c["metadata"]["kind"] == "text" for c in chunks) == 2


def test_the_first_call_per_reader_is_marked_as_warm_up(tmp_path, monkeypatch):
    log = tmp_path / "t.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))
    monkeypatch.setattr(tables, "_warmed_up", set())
    for _ in range(2):
        tables.log_reader_call("a.pdf", 1, "pdfplumber", 0.1, "ok")
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[0].get("warmup") is True and "warmup" not in lines[1]


def test_nothing_is_logged_without_a_log_path(tmp_path, monkeypatch):
    monkeypatch.delenv("VRAG_TABLE_LOG", raising=False)
    monkeypatch.chdir(tmp_path)
    tables.log_line({"file": "a.pdf"})
    assert list(tmp_path.iterdir()) == []


# Detection cache and box rules

def test_a_new_threshold_misses_the_detection_cache(pdf_dir, tmp_path):
    cache = str(tmp_path / "cache")
    load(pdf_dir, tables.TableExtractor(parser="pdfplumber", detector=FakeDetector([1]), cache_dir=cache, use_cache=True))
    stricter = FakeDetector([1])
    stricter.settings = dict(FakeDetector.settings, threshold=0.5)
    tables.reset_counters()
    load(pdf_dir, tables.TableExtractor(parser="pdfplumber", detector=stricter, cache_dir=cache, use_cache=True))
    assert tables.detector_calls == 2


def test_padded_boxes_never_leave_the_page():
    box = {"x0": 2.0, "top": 1.0, "x1": 610.0, "bottom": 790.0}
    assert tables.padded_box(box, (0, 0, 612, 792)) == (0, 0, 612, 792)


def test_pdfplumber_skips_rotated_boxes_and_pages_without_text():
    class Page:
        bbox = (0, 0, 612, 792)
        def crop(self, box):
            raise AssertionError("a rotated box must not be read")

    rotated = {"x0": 50.0, "top": 50.0, "x1": 300.0, "bottom": 300.0, "label": "table rotated", "score": 1.0}
    assert tables.read_pdfplumber({"has_text": True, "plumber": Page(), "boxes": [rotated]}) == []
    assert tables.read_pdfplumber({"has_text": False, "plumber": Page(), "boxes": [dict(rotated, label="table")]}) == []


# Startup check and fingerprint

@pytest.fixture
def fresh_parser(monkeypatch):
    monkeypatch.setattr(tables, "_effective_parser", None)


def test_an_unknown_parser_is_one_error_and_falls_back_to_none(fresh_parser, monkeypatch, capsys):
    monkeypatch.setattr(tables, "configured_parser", lambda: "docling")
    assert tables.check_table_parser() == "none"
    assert capsys.readouterr().out.count("ERROR") == 1


def test_parser_none_never_loads_the_detector(fresh_parser, monkeypatch):
    monkeypatch.setattr(tables, "configured_parser", lambda: "none")
    monkeypatch.setattr(tables, "get_detector", lambda: pytest.fail("detector loaded"))
    assert tables.check_table_parser() == "none"


def test_a_pulled_vlm_passes_the_startup_check(fresh_parser, monkeypatch):
    monkeypatch.setattr(tables, "configured_parser", lambda: "qwen2.5vl:3b")
    monkeypatch.setattr(tables, "get_detector", lambda: type("D", (), {"load": lambda self: self})())
    monkeypatch.setattr(tables, "vlm_digest", lambda model: "fb90415cde1e")
    assert tables.check_table_parser() == "qwen2.5vl:3b"
    assert tables.effective_parser() == "qwen2.5vl:3b"


def test_vlm_digest_accepts_the_latest_tag_and_names_the_pull_command(monkeypatch):
    import ollama
    listing = type("L", (), {"models": [type("M", (), {"model": "qwen2.5vl:3b:latest", "digest": "abc"})()]})()
    monkeypatch.setattr(ollama, "list", lambda: listing)
    assert tables.vlm_digest("qwen2.5vl:3b") == "abc"
    with pytest.raises(RuntimeError, match="ollama pull qwen2.5vl:7b"):
        tables.vlm_digest("qwen2.5vl:7b")


def test_the_fingerprint_changes_with_parser_threshold_and_model_revision(monkeypatch):
    settings = {"ingestion.table_detect_threshold": 0.99, "ingestion.table_model_revision": "r1"}
    monkeypatch.setattr(tables, "get", lambda key, default=None: settings.get(key, default))
    base = tables.table_fingerprint("pdfplumber")
    assert tables.table_fingerprint("none") == "none"
    assert tables.table_fingerprint("qwen2.5vl:3b") != base
    assert tables.table_fingerprint("layoutlmv3").endswith("|r1")
    settings["ingestion.table_detect_threshold"] = 0.9
    assert tables.table_fingerprint("pdfplumber") != base


def test_the_marker_stores_the_fingerprint_and_an_old_marker_has_none(tmp_path):
    manager = FakeManager(tmp_path)
    assert manager.index_tables() is None
    manager.mark_index_format("pdfplumber|m|r|0.99")
    assert manager.index_tables() == "pdfplumber|m|r|0.99"
    (tmp_path / "index_version.json").write_text('{"format": 3}')
    assert manager.index_tables() is None


def test_matching_settings_do_not_rebuild(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    use_corpus(monkeypatch, tmp_path / "empty")
    manager = FakeManager(tmp_path)
    monkeypatch.setattr(routes, "get_retriever", lambda: type("R", (), {"store": type("S", (), {"db_manager": manager})})())
    monkeypatch.setattr(tables, "table_fingerprint", lambda parser=None: "none")
    manager.mark_index_format("none")
    routes.ensure_index_format()
    assert manager.ingested == []


def test_health_reports_the_effective_table_parser(monkeypatch):
    monkeypatch.setattr(routes, "ollama_status", lambda: "online")
    monkeypatch.setattr(routes, "get_llm", lambda: type("LLM", (), {"generation_model": "llama3.2:3b"})())
    monkeypatch.setattr(tables, "_effective_parser", "pdfplumber")
    body = routes.health_check()
    assert body["table_parser_effective"] == "pdfplumber"


# Reranker input

@pytest.mark.parametrize("line", ["|---|---|", "| :--- | ---: |", "---|---", "|:---:|"])
def test_every_markdown_separator_style_is_dropped(line):
    from src.pipeline.retrieval import rerank_text
    assert rerank_text(f"| a | b |\n{line}\n| 1 | 2 |") == "a b\n1 2"


# --- LayoutLMv3 reader and read cache (build steps 7 and 8) ---

def _word(text, x0, top, label="TABLE_CELL", width=30, height=10):
    return {"text": text, "x0": x0, "top": top, "x1": x0 + width, "bottom": top + height, "label": label}


def test_a_synthetic_three_by_three_table_rebuilds_exactly():
    words = [_word("Tier", 0, 0, "TABLE_HEADER"), _word("Size", 100, 0, "TABLE_HEADER"), _word("DTUs", 200, 0, "TABLE_HEADER"),
             _word("Basic", 0, 20), _word("2", 100, 20), _word("5", 200, 20),
             _word("S0", 0, 40), _word("250", 100, 40), _word("10", 200, 40)]
    assert tables.rebuild_table(words) == ("| Tier | Size | DTUs |\n|---|---|---|\n"
                                           "| Basic | 2 | 5 |\n| S0 | 250 | 10 |")


def test_leading_header_rows_merge_column_by_column():
    words = [_word("Default", 100, 0, "TABLE_HEADER"),
             _word("GPIO", 0, 15, "TABLE_HEADER"), _word("Pull", 100, 15, "TABLE_HEADER"),
             _word("9", 0, 35), _word("Low", 100, 35)]
    assert tables.rebuild_table(words) == "| GPIO | Default Pull |\n|---|---|\n| 9 | Low |"


def test_without_header_words_the_first_row_is_the_header():
    words = [_word("Tier", 0, 0), _word("Size", 100, 0), _word("Basic", 0, 20), _word("2", 100, 20)]
    assert tables.rebuild_table(words) == "| Tier | Size |\n|---|---|\n| Basic | 2 |"


def test_words_in_one_cell_join_in_left_to_right_order():
    words = [_word("Name", 0, 0, "TABLE_HEADER"), _word("Value", 200, 0, "TABLE_HEADER"),
             _word("blob", 32, 20, width=25), _word("Page", 0, 20, width=28), _word("1", 200, 20)]
    assert tables.rebuild_table(words).splitlines()[2] == "| Page blob | 1 |"


def test_an_empty_word_list_gives_no_table():
    assert tables.rebuild_table([]) is None


def test_each_word_takes_its_label_from_the_most_central_window():
    # word 5 sits at the end of window A (positions 1..6) and in the middle of window B (positions 1..9)
    window_a = [None, 0, 1, 2, 3, 4, 5, None]
    window_b = [None, 2, 3, 4, 5, 5, 6, 7, 8, 9, None]
    preds_a = [0] * len(window_a)
    preds_b = [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0]          # word 5's first subword in B is position 4
    merged = tables.merge_windows([window_a, window_b], [preds_a, preds_b])
    assert merged[5] == 1 and merged[0] == 0 and merged[9] == 0


def test_reading_order_goes_line_by_line_then_left_to_right():
    boxes = [[200, 0, 230, 10], [0, 2, 30, 12], [0, 30, 30, 40], [100, 1, 130, 11]]
    assert tables.reading_order(boxes) == [1, 3, 0, 2]


def test_layoutlmv3_falls_back_to_tesseract_when_the_text_layer_is_thin(monkeypatch):
    import pytesseract

    class Plumber:
        bbox = (0, 0, 612, 792)
        def extract_words(self):
            return [{"text": "x", "x0": 0, "top": 0, "x1": 5, "bottom": 5}]

    class Render:
        def to_pil(self):
            from PIL import Image
            return Image.new("RGB", (10, 10))

    pdfium_page = type("P", (), {"render": lambda self, scale: Render()})()
    monkeypatch.setattr(pytesseract, "image_to_data", lambda image, output_type: {
        "text": ["P11", " ", "1,750"], "left": [0, 0, 200], "top": [400, 0, 400], "width": [100, 0, 100], "height": [50, 0, 50]})
    words = tables.page_words({"plumber": Plumber(), "pdfium": pdfium_page, "has_text": True})
    assert [w["text"] for w in words] == ["P11", "1,750"]
    assert words[1]["x0"] == pytest.approx(200 / (200 / 72)) and words[1]["top"] == pytest.approx(400 / (200 / 72))


def test_layoutlmv3_passes_the_startup_check_when_the_hub_model_loads(monkeypatch):
    monkeypatch.setattr(tables, "_effective_parser", None)
    monkeypatch.setattr(tables, "configured_parser", lambda: "layoutlmv3")
    monkeypatch.setattr(tables, "get_detector", lambda: type("D", (), {"load": lambda self: self})())
    monkeypatch.setattr(tables, "get_layout_model", lambda: type("M", (), {"load": lambda self: self})())
    assert tables.check_table_parser() == "layoutlmv3"


# Read cache (build step 8)

def test_a_second_ingest_reads_tables_from_the_cache(pdf_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "read_page", lambda parser, page: [(10.0, TABLE)])
    monkeypatch.setattr(tables, "reader_model_id", lambda parser: "rev1")
    make = lambda: tables.TableExtractor(parser="layoutlmv3", detector=FakeDetector([1]),
                                         cache_dir=str(tmp_path / "cache"), use_cache=True)
    first = load(pdf_dir, make())
    tables.reset_counters()
    second = load(pdf_dir, make())
    assert tables.reader_calls == 0 and tables.detector_calls == 0
    assert second == first


def test_a_new_model_revision_misses_the_read_cache(pdf_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "read_page", lambda parser, page: [(10.0, TABLE)])
    revision = {"id": "rev1"}
    monkeypatch.setattr(tables, "reader_model_id", lambda parser: revision["id"])
    make = lambda: tables.TableExtractor(parser="layoutlmv3", detector=FakeDetector([1]),
                                         cache_dir=str(tmp_path / "cache"), use_cache=True)
    load(pdf_dir, make())
    revision["id"] = "rev2"
    tables.reset_counters()
    load(pdf_dir, make())
    assert tables.reader_calls == 1


def test_failed_reads_are_not_cached(pdf_dir, tmp_path, monkeypatch):
    def broken(parser, page):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(tables, "read_page", broken)
    monkeypatch.setattr(tables, "reader_model_id", lambda parser: "rev1")
    make = lambda: tables.TableExtractor(parser="layoutlmv3", detector=FakeDetector([1]),
                                         cache_dir=str(tmp_path / "cache"), use_cache=True)
    load(pdf_dir, make())
    tables.reset_counters()
    load(pdf_dir, make())
    assert tables.reader_calls == 1


def test_the_read_cache_key_names_each_readers_model(monkeypatch):
    import pdfplumber
    monkeypatch.setattr(tables, "get", lambda key, default=None: {"ingestion.table_model_revision": "abc123"}.get(key, default))
    monkeypatch.setattr(tables, "vlm_digest", lambda model: f"digest-{model}")
    assert tables.reader_model_id("layoutlmv3") == "abc123"
    assert tables.reader_model_id("qwen2.5vl:7b") == "digest-qwen2.5vl:7b"
    assert tables.reader_model_id("pdfplumber") == pdfplumber.__version__


# --- LayoutLMv3 labelling path and reader filtering ---

def test_label_words_merges_window_predictions_into_label_names():
    import torch

    class Encoding(dict):
        def word_ids(self, i):
            return [[None, 0, 1, 2, None], [None, 2, 3, None, None]][i]

    class Tokenizer:
        def __call__(self, words, boxes, **kwargs):
            assert kwargs["stride"] == tables.LAYOUT_STRIDE and kwargs["max_length"] == tables.LAYOUT_MAX_LEN
            return Encoding(input_ids=[[0] * 5, [0] * 5], attention_mask=[[1] * 5, [1] * 5], bbox=[[[0] * 4] * 5] * 2)

    class ImageProcessor:
        def __call__(self, image, return_tensors):
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    class Model:
        def __call__(self, **inputs):
            assert inputs["pixel_values"].shape[0] == 2          # one page image per window
            logits = torch.zeros(2, 5, 3)
            logits[0, 1, 1] = logits[0, 2, 2] = 1                 # window 0: word 0 header, word 1 cell
            logits[1, 1, 2] = logits[1, 2, 1] = 1                 # window 1: word 2 cell, word 3 header
            return type("Out", (), {"logits": logits})()

    layout = tables.LayoutModel.__new__(tables.LayoutModel)
    layout.model, layout.device = Model(), "cpu"
    layout.processor = type("P", (), {"tokenizer": Tokenizer(), "image_processor": ImageProcessor()})()
    labels = layout.label_words(["Tier", "Size", "Basic", "2"], [[0, 0, 1, 1]] * 4, image=None)
    # word 2 sits at the edge of window 0 but in the middle of window 1, so window 1's label wins
    assert labels == ["TABLE_HEADER", "TABLE_CELL", "TABLE_CELL", "TABLE_HEADER"]


def test_the_layoutlmv3_reader_keeps_labeled_words_inside_unrotated_boxes(monkeypatch):
    from PIL import Image
    words = [{"text": t, "x0": x, "top": y, "x1": x + 30, "bottom": y + 10} for t, x, y in [
        ("Tier", 100, 100), ("Size", 200, 100), ("Basic", 100, 120), ("2", 200, 120),
        ("Caption", 100, 60),          # labeled O: never part of a table
        ("Stray", 400, 500)]]          # labeled as a cell but outside every box
    labels = {"Tier": "TABLE_HEADER", "Size": "TABLE_HEADER", "Basic": "TABLE_CELL", "2": "TABLE_CELL",
              "Caption": "O", "Stray": "TABLE_CELL"}
    monkeypatch.setattr(tables, "page_words", lambda page: words)
    monkeypatch.setattr(tables, "render", lambda pdfium_page, px: Image.new("RGB", (100, 100)))
    monkeypatch.setattr(tables, "get_layout_model", lambda: type("M", (), {
        "label_words": lambda self, texts, boxes, image: [labels[t] for t in texts]})())
    plumber = type("Plumber", (), {"bbox": (0, 0, 612, 792), "width": 612, "height": 792})()
    table_box = {"x0": 95.0, "top": 95.0, "x1": 240.0, "bottom": 135.0, "label": "table", "score": 1.0}
    rotated = dict(table_box, label="table rotated", top=480.0, bottom=520.0, x0=380.0, x1=450.0)

    out = tables.read_layoutlmv3({"plumber": plumber, "pdfium": None, "has_text": True, "boxes": [table_box, rotated]})
    assert out == [(95.0, "| Tier | Size |\n|---|---|\n| Basic | 2 |")]


@pytest.mark.integration
def test_the_real_layoutlmv3_reads_the_gpio_table_with_its_merged_header():
    """Real detector and Hub model on the Raspberry Pi 4 datasheet, page 11."""
    from pathlib import Path
    pdf_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "table_pdfs" / "raspberry-pi-4-datasheet.pdf"
    if not pdf_path.exists():
        pytest.skip("test PDF not downloaded (see data/eval/table_questions.json)")
    import pdfplumber
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    with pdfplumber.open(pdf_path) as plumber:
        page = {"number": 11, "has_text": True, "plumber": plumber.pages[10], "pdfium": pdf[10]}
        page["boxes"] = tables.get_detector().detect(pdf[10], plumber.pages[10])
        out = tables.read_page("layoutlmv3", page)
    pdf.close()
    assert len(out) == 1
    lines = out[0][1].splitlines()
    assert lines[0] == "| GPIO | Default Pull | ALT0 | ALT1 | ALT2 | ALT3 | ALT4 | ALT5 |"
    assert "| 9 | Low | SPI0 MISO |" in out[0][1] and "| 14 | Low | TXD0 |" in out[0][1]
