"""Charts and diagrams in PDFs (spec 0005): candidates, classification gating, JSON checks, chunks, cache,
unload and the startup check. Ollama is mocked; geometry runs on real generated PDFs."""
import json

import pytest

from src.api import routes
from src.ingestion import figures, tables
from src.ingestion.chunking import DocumentChunker
from src.ingestion.loader import DocumentLoader
from tests.page_citation_test import _FakeManager, use_corpus, write_pdf

PAGE = (0.0, 0.0, 612.0, 792.0)
SETTINGS = {"min_area": 0.08, "aspect_min": 0.3, "aspect_max": 3.5, "image_px": 400, "series_cap": 4,
            "values_cap": 8}
CHART = {"type": "chart", "title": "Sales by year", "axes": ["Year", "Units"],
         "series": [{"name": "Units", "values": ["10", "12", "15"]}], "summary": "Units grew every year."}


def write_chart_pdf(path, bars=40, caption="Figure 1. Units sold by year"):
    """One real PDF page: a caption line, `bars` filled bars 3 points apart, an x axis and a y axis."""
    ops = [f"BT /F1 11 Tf 72 680 Td ({caption}) Tj ET",
           "BT /F1 10 Tf 72 360 Td (Sales rose steadily across the whole period shown here.) Tj ET"]
    for i in range(bars):
        ops.append(f"{72 + i * 11} 400 8 {40 + (i * 37) % 210} re f")
    ops += ["72 400 m 520 400 l S", "72 400 m 72 660 l S"]
    stream = "\n".join(ops)
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
               "/Contents 5 0 R >>",
               f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


class FakePage:
    """The pdfplumber page surface raw_candidates reads."""
    def __init__(self, images=(), rects=(), curves=(), lines=(), number=1):
        self.bbox, self.width, self.height, self.page_number = PAGE, 612.0, 792.0, number
        self.images, self.rects, self.curves, self.lines = list(images), list(rects), list(curves), list(lines)


def box(x0, top, x1, bottom):
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def bars(n, x=72.0, top=200.0, width=8.0, gap=3.0, height=150.0):
    return [box(x + i * (width + gap), top, x + i * (width + gap) + width, top + height) for i in range(n)]


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    figures.reset_counters()
    monkeypatch.setattr(figures, "_loaded", set())


@pytest.fixture
def chart_pdf(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_chart_pdf(raw / "report.pdf")
    return raw / "report.pdf"


# --- candidate geometry (covers: AC-3) ---

def test_a_large_embedded_image_is_an_image_candidate():
    found = figures.raw_candidates(FakePage(images=[box(100, 100, 500, 450)]), SETTINGS)
    assert [(c["source"], c["x0"], c["bottom"]) for c in found] == [("image", 100, 450)]


@pytest.mark.parametrize("image", [box(100, 100, 160, 160),      # under 8% of the page
                                   box(0, 300, 612, 340)],        # a 15 to 1 strip, outside the aspect limits
                         ids=["too small", "too wide"])
def test_small_or_strip_shaped_images_are_not_candidates(image):
    assert figures.raw_candidates(FakePage(images=[image]), SETTINGS) == []


def test_forty_drawn_bars_and_two_axes_make_one_vector_candidate():
    axes = [box(72, 350, 520, 350.5), box(72, 150, 72.5, 350)]
    found = figures.raw_candidates(FakePage(rects=bars(40), lines=axes), SETTINGS)
    assert len(found) == 1 and found[0]["source"] == "vector"
    assert found[0]["x0"] == 72 and found[0]["top"] == 150


def test_a_cluster_needs_at_least_ten_drawn_objects():
    assert figures.raw_candidates(FakePage(rects=bars(9, width=40, height=300)), SETTINGS) == []


def test_page_rules_and_the_page_frame_never_form_a_figure():
    rule = box(10, 500, 602, 501)                  # wider than 90% of the page and under 2 points tall
    frame = box(5, 5, 607, 787)                    # covers the whole page
    assert figures.raw_candidates(FakePage(rects=[frame] * 6, lines=[rule] * 6), SETTINGS) == []


def test_a_page_with_more_than_5000_drawn_objects_is_skipped_and_logged(tmp_path, monkeypatch, capsys):
    log = tmp_path / "reader.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))
    dense = [box(i % 600, (i // 600) * 2, i % 600 + 1, (i // 600) * 2 + 1) for i in range(5001)]
    found = figures.raw_candidates(FakePage(lines=dense, number=23), SETTINGS, "map.pdf")
    assert found == []
    assert "5001 drawn objects" in capsys.readouterr().out
    assert json.loads(log.read_text()) == {"file": "map.pdf", "page": 23, "figure_skip": "dense",
                                           "drawn_objects": 5001}


def test_each_page_geometry_pass_is_counted():
    figures.raw_candidates(FakePage(), SETTINGS)
    figures.raw_candidates(FakePage(), SETTINGS)
    assert figures.geometry_calls == 2


@pytest.mark.parametrize("gap, clusters", [(12, 1), (30, 2)])
def test_shapes_within_twelve_points_join_one_cluster(gap, clusters):
    shapes = [box(0, 0, 10, 10), box(10 + gap, 0, 20 + gap, 10)]
    assert len(figures.cluster_boxes(shapes)) == clusters


def test_a_candidate_mostly_inside_a_table_box_is_dropped():
    candidate = dict(box(100, 100, 300, 300), source="vector")
    table = box(90, 90, 310, 250)                  # covers 75% of the candidate
    assert figures.clean_candidates([candidate], [table]) == []
    assert figures.clean_candidates([candidate], [box(90, 90, 310, 150)]) == [candidate]   # only 25% inside


def test_overlapping_candidates_merge_into_their_union_in_reading_order():
    low = dict(box(100, 400, 300, 600), source="vector")
    a, b = dict(box(100, 100, 300, 300), source="image"), dict(box(110, 110, 310, 310), source="vector")
    merged = figures.clean_candidates([low, a, b], [])
    assert [(m["x0"], m["top"], m["x1"], m["bottom"]) for m in merged] == [(100, 100, 310, 310), (100, 400, 300, 600)]


def test_padding_adds_eighteen_points_and_stays_on_the_page():
    assert figures.pad(box(100, 100, 200, 200), PAGE) == box(82, 82, 218, 218)
    assert figures.pad(box(5, 5, 607, 787), PAGE) == box(0, 0, 612, 792)


def test_the_caption_is_the_nearest_figure_line_within_72_points():
    chart = box(100, 200, 500, 400)
    lines = [{"text": "Figure 3. Far away", "top": 60, "bottom": 70},       # 130 points above
             {"text": "Chart 2. Close above", "top": 170, "bottom": 180},    # 20 points above
             {"text": "Figure 4. Below", "top": 430, "bottom": 440},         # 30 points below
             {"text": "Sales rose in 2020", "top": 405, "bottom": 410}]      # not a caption
    assert figures.find_caption(lines, chart) == "Chart 2. Close above"
    assert figures.find_caption(lines[:1], chart) is None


def test_a_text_only_page_skips_the_geometry_pass(tmp_path):
    write_pdf(tmp_path / "text.pdf", ["Plain text only."] * 3)
    found = figures.FigureExtractor(parser="qwen2.5vl:7b", cache_dir=str(tmp_path / "c")).candidates(
        str(tmp_path / "text.pdf"), table_boxes={})
    assert found == [] and figures.geometry_calls == 0


def test_a_drawn_chart_in_a_real_pdf_is_found_with_its_caption(chart_pdf, tmp_path, monkeypatch):
    log = tmp_path / "reader.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))
    found = figures.FigureExtractor(parser="qwen2.5vl:7b", cache_dir=str(tmp_path / "c")).candidates(
        str(chart_pdf), table_boxes={})
    assert len(found) == 1
    assert found[0]["source"] == "vector" and found[0]["caption"] == "Figure 1. Units sold by year"
    line = json.loads(log.read_text().splitlines()[-1])
    assert line["file"] == "report.pdf" and line["figure_candidates"][0]["box"] == found[0]["box"]


def test_table_boxes_are_detected_only_on_pages_with_candidates(chart_pdf, tmp_path):
    seen = []
    detect = lambda sha, number, pdfium_page, plumber_page: (seen.append(number) or [], False)
    figures.FigureExtractor(parser="qwen2.5vl:7b", cache_dir=str(tmp_path / "c"), detect=detect).candidates(str(chart_pdf))
    assert seen == [1]


def test_a_table_box_from_the_table_pass_removes_the_chart(chart_pdf, tmp_path):
    whole_page = {1: [box(0, 0, 612, 792)]}
    assert figures.FigureExtractor(parser="qwen2.5vl:7b", cache_dir=str(tmp_path / "c")).candidates(
        str(chart_pdf), table_boxes=whole_page) == []


# --- JSON validation (covers: AC-4) ---

def test_a_valid_reply_is_returned_trimmed():
    reply = json.dumps(dict(CHART, title="  Sales by year ", series=[{"name": " Units ", "values": [" 10 "]}]))
    figure = figures.validate_figure(reply, 4, 8)
    assert figure["title"] == "Sales by year" and figure["series"] == [{"name": "Units", "values": ["10"]}]


@pytest.mark.parametrize("change", [
    {"type": "photo"}, {"title": None}, {"summary": "   "}, {"axes": ["a", "b", "c", "d", "e"]},
    {"axes": "Year"}, {"series": [{"name": "s", "values": []}] * 5},
    {"series": [{"name": "s", "values": [str(i) for i in range(9)]}]},
    {"series": [{"name": "s", "values": [10, 12]}]}, {"series": [{"values": ["1"]}]},
], ids=["wrong type", "no title", "empty summary", "five axes", "axes not a list", "five series",
        "nine values", "numbers not strings", "series without a name"])
def test_replies_outside_the_schema_are_rejected(change):
    assert figures.validate_figure(json.dumps(dict(CHART, **change)), 4, 8) is None


@pytest.mark.parametrize("reply", ['{"type": "chart", "title": "Sales", "axes": [', "", "not json", "[1, 2]"])
def test_cut_off_or_non_object_replies_are_rejected(reply):
    assert figures.validate_figure(reply, 4, 8) is None


def test_the_schema_carries_the_configured_caps():
    schema = figures.figure_schema(2, 4)
    assert schema["properties"]["series"]["maxItems"] == 2
    assert schema["properties"]["series"]["items"]["properties"]["values"]["maxItems"] == 4
    assert schema["properties"]["type"]["enum"] == ["chart", "diagram"]


@pytest.mark.parametrize("reply, kind", [('{"type": "table"}', "table"), ('{"type": "chart"}', "chart"),
                                         ('{"type": "map"}', None), ("{", None), ('"chart"', None)])
def test_only_a_known_kind_is_accepted(reply, kind):
    assert figures.parse_kind(reply) == kind


# --- classification gating, retry and fallback (covers: AC-3, AC-4) ---

class FakeAsk:
    """Stands in for the Ollama call: classify replies first, then extract replies, recording each call."""
    def __init__(self, kind, *extract_replies):
        self.replies = [json.dumps({"type": kind})] + list(extract_replies)
        self.calls = []

    def __call__(self, model, prompt, png, schema, num_predict):
        self.calls.append({"prompt": prompt, "schema": schema, "num_predict": num_predict})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def read(monkeypatch, fake):
    log = []
    monkeypatch.setattr(figures, "ask", fake)
    result = figures.read_figure("qwen2.5vl:7b", b"png", SETTINGS, lambda st, s, o, kind=None: log.append((st, o, kind)))
    return result, log


@pytest.mark.parametrize("kind", ["screenshot", "table", "photo", "other"])
def test_anything_but_a_chart_or_diagram_gets_no_extraction_call(monkeypatch, kind):
    result, log = read(monkeypatch, FakeAsk(kind))
    assert result == {"classification": kind, "figure": None, "outcome": "empty"}
    assert log == [("classify", "empty", kind)]
    assert (figures.figure_classify_calls, figures.figure_extract_calls) == (1, 0)


def test_a_chart_is_classified_then_read(monkeypatch):
    fake = FakeAsk("chart", json.dumps(CHART))
    result, log = read(monkeypatch, fake)
    assert result["outcome"] == "ok" and result["figure"]["title"] == "Sales by year"
    assert log == [("classify", "ok", "chart"), ("extract", "ok", None)]
    assert [c["num_predict"] for c in fake.calls] == [20, 600]
    assert (figures.figure_classify_calls, figures.figure_extract_calls) == (1, 1)


def test_invalid_json_is_retried_once_with_half_the_caps(monkeypatch):
    fake = FakeAsk("diagram", '{"type": "chart", "title": "cut', json.dumps(CHART))
    result, log = read(monkeypatch, fake)
    retry = fake.calls[2]
    assert result["outcome"] == "ok"
    assert retry["num_predict"] == 300
    assert retry["schema"]["properties"]["series"]["maxItems"] == 2
    assert "up to 2 data series with up to 4 values" in retry["prompt"]
    assert log[1:] == [("extract", "retry", None), ("extract", "ok", None)]


def test_two_invalid_replies_end_in_a_failed_read(monkeypatch):
    result, log = read(monkeypatch, FakeAsk("chart", "{", "{"))
    assert result == {"classification": "chart", "figure": None, "outcome": "failed"}
    assert log[-1] == ("extract", "failed", None) and figures.figure_extract_calls == 2


def test_a_classify_timeout_is_a_timeout_and_reads_nothing(monkeypatch):
    import httpx
    fake = FakeAsk("chart")
    fake.replies = [httpx.ReadTimeout("slow")]
    result, log = read(monkeypatch, fake)
    assert result["outcome"] == "timeout" and result["classification"] is None
    assert log == [("classify", "timeout", None)] and figures.figure_extract_calls == 0


def test_an_unknown_kind_is_a_failed_classification(monkeypatch):
    result, log = read(monkeypatch, FakeAsk("map"))
    assert result["outcome"] == "failed" and figures.figure_extract_calls == 0


def test_an_extract_timeout_keeps_the_classification(monkeypatch):
    import httpx
    result, _ = read(monkeypatch, FakeAsk("chart", httpx.ReadTimeout("slow")))
    assert result == {"classification": "chart", "figure": None, "outcome": "timeout"}


# --- figure chunks (covers: AC-5) ---

def test_a_figure_body_is_marked_generated_and_lists_every_field():
    body = figures.figure_body("Figure 1.", CHART)
    assert body.splitlines() == ["Figure description (generated)", "Caption: Figure 1.", "Title: Sales by year",
                                 "Type: chart", "Axes: Year, Units", "Units: 10, 12, 15",
                                 "Summary: Units grew every year."]


def test_a_series_that_repeats_one_value_is_left_out_of_the_chunk():
    """A copied series (the latest value repeated across a time series) would outweigh the true figures for the
    answer model; a short or varying series stays. covers: AC-5"""
    figure = dict(CHART, series=[{"name": "Female", "values": ["$55,240"] * 8},
                                 {"name": "Ratio", "values": ["60%", "60%", "65%"]},
                                 {"name": "Latest", "values": ["$42,110", "$42,110"]}])
    lines = figures.figure_body("Figure 6.", figure).splitlines()

    assert "Ratio: 60%, 60%, 65%" in lines and "Latest: $42,110, $42,110" in lines
    assert not any(line.startswith("Female:") for line in lines)


def test_changing_how_reads_are_written_changes_the_index_fingerprint_but_not_the_read_cache(monkeypatch):
    """Cached vision reads are reused; only the chunk text is rebuilt. covers: AC-5"""
    monkeypatch.setattr(figures, "effective_parser", lambda: "qwen2.5vl:7b")
    before = figures.figure_fingerprint()
    monkeypatch.setattr(figures, "FIGURE_BODY_VERSION", figures.FIGURE_BODY_VERSION + 1)

    assert figures.figure_fingerprint() != before
    assert figures.FIGURE_PROMPT_VERSION == 2


def test_a_failed_read_keeps_the_caption_only():
    assert figures.figure_body(None, None) == ("Figure description (generated)\nCaption: none\n"
                                               "Figure description unavailable")


def test_figure_chunks_come_after_tables_and_never_shift_their_ids():
    doc = {"content": "x", "metadata": {"source": "r.pdf", "doc_hash": "a" * 64},
           "pages": [(1, "Text on page one."), (2, "Text on page two.")],
           "tables": [(2, 100.0, "| A | B |\n|---|---|\n| 1 | 2 |")], "table_parser": "layoutlmv3",
           "figures": [(1, 50.0, "Figure 1.", CHART), (2, 80.0, None, None)], "figure_parser": "qwen2.5vl:7b"}
    plain = DocumentChunker().chunk_documents([{k: v for k, v in doc.items() if k not in ("figures", "figure_parser")}])
    chunks = DocumentChunker().chunk_documents([doc])

    assert [c["id"] for c in chunks[:len(plain)]] == [c["id"] for c in plain]
    figure_chunks = chunks[len(plain):]
    assert [c["id"] for c in figure_chunks] == ["aaaaaaaaaaaaaaaa_3", "aaaaaaaaaaaaaaaa_4"]
    first = figure_chunks[0]
    assert first["content"].startswith("[Source: r.pdf | Page: 1]\nFigure description (generated)")
    assert first["metadata"] == {"source": "r.pdf", "doc_hash": "a" * 64, "chunk_index": 3, "page": 1,
                                 "kind": "figure", "figure_parser": "qwen2.5vl:7b", "figure": CHART}
    assert "Figure description unavailable" in figure_chunks[1]["content"]


def test_a_long_figure_description_stays_one_chunk():
    long = dict(CHART, summary="word " * 2000)
    doc = {"content": "x", "metadata": {"source": "r.pdf", "doc_hash": "b" * 64}, "pages": [(1, "Page text.")],
           "figures": [(1, 0.0, None, long)], "figure_parser": "qwen2.5vl:7b"}
    assert sum(c["metadata"]["kind"] == "figure" for c in DocumentChunker().chunk_documents([doc])) == 1


# --- the extractor: cache, unload, logging (covers: AC-6, AC-4) ---

class FakeRead:
    """Stands in for read_figure; returns the queued result and counts calls."""
    def __init__(self, *results):
        self.results, self.calls = list(results), 0

    def __call__(self, model, png, settings, log):
        self.calls += 1
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        log("classify", 1.0, "ok" if result["classification"] in figures.READ_KINDS else "empty",
            result["classification"])
        return result


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "vlm_digest", lambda model: "digest-1")
    unloads = []
    monkeypatch.setattr(figures, "unload", lambda model: unloads.append(model))

    def make(use_cache=True):
        fx = figures.FigureExtractor(parser="qwen2.5vl:7b", cache_dir=str(tmp_path / "cache"), use_cache=use_cache)
        fx.unloads = unloads
        return fx
    return make


def test_a_read_chart_comes_back_with_its_page_caption_and_description(chart_pdf, extractor, monkeypatch):
    monkeypatch.setattr(figures, "read_figure", FakeRead({"classification": "chart", "figure": CHART, "outcome": "ok"}))
    [(page, top, caption, figure)] = extractor().extract(str(chart_pdf), table_boxes={})
    assert (page, caption, figure) == (1, "Figure 1. Units sold by year", CHART)


def test_a_second_ingest_reads_every_figure_from_the_cache(chart_pdf, extractor, monkeypatch):
    fake = FakeRead({"classification": "chart", "figure": CHART, "outcome": "ok"})
    monkeypatch.setattr(figures, "read_figure", fake)
    first = extractor().extract(str(chart_pdf), table_boxes={})
    second = extractor().extract(str(chart_pdf), table_boxes={})
    assert fake.calls == 1 and first == second


def test_a_cached_invalid_json_failure_is_not_read_again(chart_pdf, extractor, monkeypatch):
    fake = FakeRead({"classification": "chart", "figure": None, "outcome": "failed"})
    monkeypatch.setattr(figures, "read_figure", fake)
    extractor().extract(str(chart_pdf), table_boxes={})
    [(_, _, _, figure)] = extractor().extract(str(chart_pdf), table_boxes={})
    assert fake.calls == 1 and figure is None


def test_a_timeout_is_not_cached_so_the_next_ingest_tries_again(chart_pdf, extractor, monkeypatch):
    fake = FakeRead({"classification": "chart", "figure": None, "outcome": "timeout"},
                    {"classification": "chart", "figure": CHART, "outcome": "ok"})
    monkeypatch.setattr(figures, "read_figure", fake)
    extractor().extract(str(chart_pdf), table_boxes={})
    [(_, _, _, figure)] = extractor().extract(str(chart_pdf), table_boxes={})
    assert fake.calls == 2 and figure == CHART


def test_a_new_model_digest_misses_the_cache(chart_pdf, extractor, monkeypatch):
    fake = FakeRead({"classification": "chart", "figure": CHART, "outcome": "ok"})
    monkeypatch.setattr(figures, "read_figure", fake)
    extractor().extract(str(chart_pdf), table_boxes={})
    monkeypatch.setattr(tables, "vlm_digest", lambda model: "digest-2")
    extractor().extract(str(chart_pdf), table_boxes={})
    assert fake.calls == 2


def test_the_cache_can_be_turned_off(chart_pdf, extractor, monkeypatch, tmp_path):
    fake = FakeRead({"classification": "chart", "figure": CHART, "outcome": "ok"})
    monkeypatch.setattr(figures, "read_figure", fake)
    extractor(use_cache=False).extract(str(chart_pdf), table_boxes={})
    extractor(use_cache=False).extract(str(chart_pdf), table_boxes={})
    assert fake.calls == 2 and not (tmp_path / "cache").exists()


def test_a_non_chart_is_left_out_of_the_index(chart_pdf, extractor, monkeypatch):
    monkeypatch.setattr(figures, "read_figure", FakeRead({"classification": "table", "figure": None, "outcome": "empty"}))
    assert extractor().extract(str(chart_pdf), table_boxes={}) == []


def test_the_model_is_unloaded_after_the_pdf_only_when_it_was_called(chart_pdf, extractor, monkeypatch):
    monkeypatch.setattr(figures, "read_figure", FakeRead({"classification": "chart", "figure": CHART, "outcome": "ok"}))
    fx = extractor()
    fx.extract(str(chart_pdf), table_boxes={})
    assert fx.unloads == ["qwen2.5vl:7b"]
    extractor().extract(str(chart_pdf), table_boxes={})              # all cached, nothing loaded
    assert fx.unloads == ["qwen2.5vl:7b"]


def test_unload_asks_ollama_to_free_the_model_at_once(monkeypatch):
    calls = []
    monkeypatch.setattr(figures, "_client", type("C", (), {"generate": lambda self, **k: calls.append(k)})())
    figures._loaded.add("qwen2.5vl:7b")
    figures.unload("qwen2.5vl:7b")
    assert calls == [{"model": "qwen2.5vl:7b", "prompt": "", "keep_alive": 0}]
    assert "qwen2.5vl:7b" not in figures._loaded


def test_the_first_call_after_each_load_is_marked_as_warm_up(tmp_path, monkeypatch):
    log = tmp_path / "reader.jsonl"
    monkeypatch.setenv("VRAG_TABLE_LOG", str(log))
    figures.log_call("r.pdf", 1, [1, 2, 3, 4], "m", "classify", 9.0, "ok", "chart")
    figures.log_call("r.pdf", 1, [1, 2, 3, 4], "m", "extract", 20.0, "ok")
    figures._loaded.discard("m")                                        # what unload does
    figures.log_call("r.pdf", 2, [1, 2, 3, 4], "m", "classify", 9.0, "empty", "photo")
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line.get("warmup", False) for line in lines] == [True, False, True]
    assert lines[0] == {"file": "r.pdf", "page": 1, "box": [1, 2, 3, 4], "reader": "figure:m", "stage": "classify",
                        "seconds": 9.0, "outcome": "ok", "kind": "chart", "warmup": True}
    assert "kind" not in lines[1]


def test_figure_parser_none_does_nothing(chart_pdf, monkeypatch):
    monkeypatch.setattr(figures, "read_figure", lambda *a: pytest.fail("vision call"))
    assert figures.FigureExtractor(parser="none").extract(str(chart_pdf)) == []
    assert figures.geometry_calls == 0


# --- the loader (covers: AC-2, AC-7) ---

class StubFigures:
    enabled, parser = True, "qwen2.5vl:7b"

    def __init__(self, result=None, error=None):
        self.result, self.error, self.seen = result or [], error, []

    def extract(self, file_path, table_boxes=None):
        self.seen.append(table_boxes)
        if self.error:
            raise self.error
        return self.result


class StubTables:
    enabled, parser = True, "layoutlmv3"
    page_boxes = {1: [box(0, 0, 10, 10)]}

    def extract(self, file_path, text_pages):
        return [(1, 5.0, "| A | B |\n|---|---|\n| 1 | 2 |")]


def test_the_figure_pass_reuses_the_table_pass_boxes(chart_pdf):
    stub = StubFigures([(1, 10.0, "Figure 1.", CHART)])
    [doc] = DocumentLoader(str(chart_pdf.parent), StubTables(), stub).load_documents()
    assert stub.seen == [StubTables.page_boxes]
    assert doc["figures"] == [(1, 10.0, "Figure 1.", CHART)] and doc["figure_parser"] == "qwen2.5vl:7b"


def test_a_crashing_figure_pass_keeps_the_text_and_the_tables(chart_pdf, capsys):
    broken = DocumentLoader(str(chart_pdf.parent), StubTables(), StubFigures(error=RuntimeError("gpu"))).load_documents()
    [doc] = broken
    assert "figures" not in doc and doc["tables"] and doc["pages"]
    assert "[Figures] Figure extraction failed" in capsys.readouterr().out


def test_figure_parser_none_leaves_the_document_as_before(chart_pdf):
    [doc] = DocumentLoader(str(chart_pdf.parent), tables.TableExtractor(parser="none"),
                           figures.FigureExtractor(parser="none")).load_documents()
    assert "figures" not in doc and "figure_parser" not in doc


# --- startup check, fingerprint and health (covers: AC-7) ---

@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(figures, "_effective_parser", None)

    def set_parser(parser):
        monkeypatch.setattr(figures, "configured_parser", lambda: parser)
    return set_parser


def test_a_missing_figure_model_is_one_error_and_falls_back_to_none(configured, monkeypatch, capsys):
    configured("qwen2.5vl:7b")

    def missing(model):
        raise RuntimeError(f"Ollama model {model!r} is not pulled")
    monkeypatch.setattr(tables, "vlm_digest", missing)
    assert figures.check_figure_parser() == "none"
    assert figures.figure_fingerprint() == ""
    assert capsys.readouterr().out.count("ERROR") == 1


def test_an_unknown_figure_parser_is_one_error(configured, capsys):
    configured("llava")
    assert figures.check_figure_parser() == "none"
    assert capsys.readouterr().out.count("ERROR") == 1


def test_a_pulled_figure_model_passes_the_startup_check(configured, monkeypatch):
    configured("qwen2.5vl:3b")
    monkeypatch.setattr(tables, "vlm_digest", lambda model: "fb90")
    assert figures.check_figure_parser() == "qwen2.5vl:3b" == figures.effective_parser()


def test_the_figure_fingerprint_follows_every_setting(monkeypatch):
    settings = {}
    monkeypatch.setattr(figures, "get", lambda key, default=None: settings.get(key, default))
    base = figures.figure_fingerprint("qwen2.5vl:7b")
    assert base == f"|fig:qwen2.5vl:7b|800|0.08|0.3-3.5|4x8|{figures.FIGURE_PROMPT_VERSION}|body{figures.FIGURE_BODY_VERSION}"
    assert figures.figure_fingerprint("none") == ""
    for key, value in (("ingestion.figure_image_px", 640), ("ingestion.figure_min_area", 0.1),
                       ("ingestion.figure_series_cap", 3)):
        settings[key] = value
        assert figures.figure_fingerprint("qwen2.5vl:7b") != base
        settings.clear()


def test_changing_the_figure_settings_rebuilds_the_index_once(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    use_corpus(monkeypatch, tmp_path / "empty")
    manager = _FakeManager(tmp_path)
    monkeypatch.setattr(routes, "get_retriever", lambda: type("R", (), {"store": type("S", (), {"db_manager": manager})})())
    monkeypatch.setattr(tables, "table_fingerprint", lambda parser=None: "none")
    manager.mark_index_format("none")

    monkeypatch.setattr(figures, "figure_fingerprint", lambda parser=None: "|fig:qwen2.5vl:7b|800")
    routes.ensure_index_format()
    routes.ensure_index_format()

    assert len(manager.ingested) == 1
    assert manager.index_tables() == "none|fig:qwen2.5vl:7b|800"


def test_health_reports_the_effective_figure_parser(monkeypatch):
    import asyncio
    import ollama
    monkeypatch.setattr(routes, "ollama_status", lambda: "online")
    monkeypatch.setattr(routes, "get_llm", lambda: type("LLM", (), {"generation_model": "llama3.2:3b"})())
    monkeypatch.setattr(figures, "_effective_parser", "qwen2.5vl:7b")
    assert routes.health_check()["figure_parser_effective"] == "qwen2.5vl:7b"
