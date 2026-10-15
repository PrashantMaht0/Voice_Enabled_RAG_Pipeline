"""The Observability tab (spec 0007): card values and notes, the recent table, row selection and the trace panel,
with the API replaced by fixed replies."""
from types import SimpleNamespace

import pytest

import app as ui


def metrics_reply(asked=9, answered=7, errors=0, failed_writes=0, counted=9):
    """An /api/metrics reply shaped like the real one."""
    empty = not asked
    return {
        "window": "1h",
        "counts": {"asked": asked, "answered": answered, "refused_gate": 0 if empty else 2, "refused_grounding": 0,
                   "refused_model": 0, "refused_empty": 0, "cache_hits": 0 if empty else 3, "aborted": 0,
                   "errors": errors},
        "rates": {"cache_hit_rate": None if empty else 0.333, "refusal_rate": None if empty else 0.222,
                  "grounding_pass_rate": None if empty else 1.0, "retrieval_pass_rate": None if empty else 0.833,
                  "error_rate": None if empty else round(errors / asked, 4)},
        "retrieval": {"reached": 0 if empty else 6, "passed": 0 if empty else 5},
        "grounding": {"checked": 0 if empty else 5, "passed": 0 if empty else 5},
        "latency": {"p50": None, "p99": None, "n": 0} if empty else {"p50": 2723.8, "p99": 4875.2, "n": 5},
        "tokens": {"prompt": 0 if empty else 8453, "output": 0 if empty else 246, "total": 0 if empty else 8699,
                   "per_question": None if empty or not counted else 967, "counted": 0 if empty else counted,
                   "asked": asked},
        "stages": {} if empty else {"rerank": {"p50": 600.0, "p95": 700.0, "p99": 812.0, "n": 5}},
        "voice": {"stt_ms": {"p50": None, "p95": None, "p99": None, "n": 0}},
        "failed_writes": failed_writes,
        "quality": None,
    }


RECENT = [
    {"id": 391, "created_at": "2026-10-13T20:12:17.000+00:00", "channel": "typed", "question": "Q decoder",
     "path": "cache", "total_ms": 1, "answered": True, "prompt_tokens": 0, "output_tokens": 0},
    {"id": 389, "created_at": "2026-10-13T20:12:05.000+00:00", "channel": "typed", "question": "Q cake",
     "path": "cache", "total_ms": 1, "answered": False, "prompt_tokens": 0, "output_tokens": 0},
    {"id": 384, "created_at": "2026-10-13T20:11:58.000+00:00", "channel": "voice", "question": "Q follow up",
     "path": "answered", "total_ms": 1954, "answered": True, "prompt_tokens": 1625, "output_tokens": 42},
]


@pytest.fixture
def api(monkeypatch):
    """Replaces requests.get in the dashboard; records each call's url and params."""
    calls, replies = [], {}

    def get(url, params=None, timeout=None):
        calls.append((url.rsplit("/api", 1)[-1], params))
        for suffix, reply in replies.items():
            if url.endswith(suffix):
                if isinstance(reply, Exception):
                    raise reply
                return SimpleNamespace(json=lambda reply=reply: reply)
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(ui.requests, "get", get)
    return SimpleNamespace(calls=calls, replies=replies)


def card_texts(html):
    """(value, label, note) per card, in order."""
    import re
    return re.findall(r"<div class='metric-value'>(.*?)</div><div class='metric-label'>(.*?)</div>"
                      r"(?:<div class='metric-note'>(.*?)</div>)?</div>", html)


# --- formatting rules (covers: AC-1, AC-2) ---

@pytest.mark.parametrize("ms, shown", [(None, "—"), (0, "0 ms"), (999.4, "999 ms"), (999.6, "1.0 s"),
                                       (1000, "1.0 s"), (2723.8, "2.7 s"), (21922.4, "21.9 s")])
def test_latency_rounds_first_then_picks_ms_or_seconds(ms, shown):
    assert ui.format_ms(ms) == shown


@pytest.mark.parametrize("errors, asked, shown", [(0, 0, "—"), (0, 9, "0.0%"), (1, 8, "12.5%"), (1, 5000, "<0.1%"),
                                                  (1, 2000, "0.1%")])
def test_error_rate_comes_from_counts_and_never_shows_zero_while_errors_exist(errors, asked, shown):
    assert ui.error_rate(errors, asked) == shown


@pytest.mark.parametrize("failed, text", [(1, "1 question could not be saved"),
                                          (2, "2 questions could not be saved"),
                                          (1500, "1,500 questions could not be saved")])
def test_the_failed_writes_line_counts_questions_with_the_right_grammar(failed, text):
    assert text in ui.failed_writes_line(failed)


def test_no_failed_writes_means_no_warning_line():
    assert ui.failed_writes_line(0) == "" and ui.failed_writes_line(None) == ""


# --- the cards (covers: AC-1, AC-2) ---

def test_ten_cards_in_the_wireframe_order_with_their_notes():
    cards = card_texts(ui.observability_cards(metrics_reply()))
    assert [label for _, label, _ in cards] == [
        "Questions", "Answered", "Retrieval quality", "Generation quality", "Refusal rate",
        "p50 latency", "p99 latency", "Error rate", "Tokens used", "Cache hits"]
    assert cards == [
        ("9", "Questions", ""), ("7", "Answered", ""),
        ("83%", "Retrieval quality", "5 of 6 passed the gate"),
        ("100%", "Generation quality", "5 of 5 checked answers had nothing removed"),
        ("22%", "Refusal rate", "2 refused"),
        ("2.7 s", "p50 latency", "over 5 answered"), ("4.9 s", "p99 latency", "over 5 answered"),
        ("0.0%", "Error rate", "0 errors"),
        ("8,699", "Tokens used", "967 per question · 9 of 9 counted"),
        ("3", "Cache hits", "33% hit rate")]


def test_an_empty_window_shows_zero_counts_dashes_for_rates_and_no_notes():
    html = ui.observability_cards(metrics_reply(asked=0, answered=0))
    cards = card_texts(html)
    assert {label: value for value, label, _ in cards} == {
        "Questions": "0", "Answered": "0", "Retrieval quality": "—", "Generation quality": "—",
        "Refusal rate": "—", "p50 latency": "—", "p99 latency": "—", "Error rate": "—", "Tokens used": "0",
        "Cache hits": "0"}
    assert all(note == "" for _, _, note in cards)
    assert "No questions in this window yet." in html


def test_the_empty_window_message_renders_as_text_not_markdown_markers():
    """covers: AC-5. The cards are HTML, so Markdown italics around the message would show as literal underscores."""
    html = ui.observability_cards(metrics_reply(asked=0, answered=0))
    assert "_No questions in this window yet._" not in html


def test_a_window_with_only_old_rows_says_none_were_counted():
    cards = dict((label, note) for _, label, note in card_texts(ui.observability_cards(
        metrics_reply(asked=4, answered=4, counted=0))))
    assert cards["Tokens used"] == "0 of 4 counted"


def test_errors_use_singular_and_plural():
    notes = lambda errors: dict((label, note) for _, label, note in card_texts(
        ui.observability_cards(metrics_reply(errors=errors))))["Error rate"]
    assert notes(1) == "1 error" and notes(3) == "3 errors"


# --- fetching the tab (covers: AC-1, AC-5, AC-7, AC-10) ---

def test_the_tab_asks_both_endpoints_for_the_chosen_window(api):
    api.replies.update({"/metrics": metrics_reply(), "/traces": RECENT})
    ui.fetch_observability("Last hour")
    assert api.calls == [("/metrics", {"window": "1h"}), ("/traces", {"limit": 50, "window": "1h"})]


def test_the_recent_table_shows_time_channel_question_and_answered(api):
    api.replies.update({"/metrics": metrics_reply(), "/traces": RECENT})
    warning, cards, rows, picks, trace, stages = ui.fetch_observability("Last 24 hours")
    assert rows == [["2026-10-13 20:12:17", "typed", "Q decoder", "yes"],
                    ["2026-10-13 20:12:05", "typed", "Q cake", "no"],
                    ["2026-10-13 20:11:58", "voice", "Q follow up", "yes"]]
    assert picks == [[rows[0], 391], [rows[1], 389], [rows[2], 384]]
    assert trace == ui.NO_TRACE, "a refresh clears the trace panel"


def test_the_stage_table_shows_p50_p99_and_n(api):
    api.replies.update({"/metrics": metrics_reply(), "/traces": RECENT})
    stages = ui.fetch_observability("Last 24 hours")[5]
    assert stages == [["Rerank", "600", "812", 5]]


def test_the_warning_line_is_hidden_without_failed_writes_and_shown_with_them(api):
    api.replies.update({"/metrics": metrics_reply(), "/traces": RECENT})
    assert ui.fetch_observability("All")[0]["visible"] is False
    api.replies["/metrics"] = metrics_reply(failed_writes=1)
    warning = ui.fetch_observability("All")[0]
    assert warning["visible"] is True and "1 question could not be saved" in warning["value"]


def test_with_the_api_down_the_tab_shows_the_error_and_empty_tables(api):
    api.replies["/metrics"] = ConnectionError("Connection refused")
    warning, cards, rows, picks, trace, stages = ui.fetch_observability("All")
    assert cards.startswith("API error: Connection refused")
    assert (rows, picks, stages, trace) == ([], [], [], ui.NO_TRACE) and warning["visible"] is False


# --- selecting a row (covers: AC-5) ---

PICKS = [[["2026-10-13 20:12:04", "typed", "Same question", "yes"], 387],
         [["2026-10-13 20:11:53", "typed", "Same question", "yes"], 383],
         [["2026-10-13 20:11:58", "voice", "Q follow up", "yes"], 384]]


def test_a_sorted_table_still_opens_the_selected_rows_own_trace():
    event = SimpleNamespace(index=[0, 2], row_value=["2026-10-13 20:11:53", "typed", "Same question", "yes"])
    assert ui.selected_trace_id(PICKS, event) == 383, "row 0 after sorting is the 20:11:53 trace, not 387"


def test_without_a_row_value_the_position_is_used():
    assert ui.selected_trace_id(PICKS, SimpleNamespace(index=[2, 0], row_value=None)) == 384


def test_a_position_past_the_table_opens_nothing():
    assert ui.selected_trace_id(PICKS, SimpleNamespace(index=[7, 0], row_value=None)) is None


# --- the trace panel (covers: AC-6) ---

def trace_reply(**overrides):
    trace = {"id": 384, "question": "How many layers does it have?", "created_at": "2026-10-13T20:11:58.000+00:00",
             "channel": "typed", "path": "answered", "total_ms": 1954.2, "confidence": 0.947, "answer": "Six.",
             "prompt_tokens": 1625, "output_tokens": 42, "retrieved": [], "dropped_sentences": None,
             "spans": [{"layer": "rewrite", "elapsed_ms": 300, "details": "x", "prompt_tokens": 169, "output_tokens": 16},
                       {"layer": "rerank", "elapsed_ms": 600, "details": "y", "prompt_tokens": None, "output_tokens": None},
                       {"layer": "generation", "elapsed_ms": 900, "details": "z", "prompt_tokens": 883, "output_tokens": 18},
                       {"layer": "grounding", "elapsed_ms": 900, "details": "w", "prompt_tokens": None, "output_tokens": None}]}
    trace.update(overrides)
    return trace


def open_with(api, trace):
    api.replies["/traces/384"] = trace
    return ui.open_trace([[["r"], 384]], SimpleNamespace(index=[0, 0], row_value=None))


def test_the_panel_shows_total_tokens_and_the_split_per_model_call(api):
    panel = open_with(api, trace_reply())
    assert "**Tokens:** prompt 1,625 · output 42 · total 1,667" in panel
    assert "- Rewrite: prompt 169 · output 16 · total 185" in panel
    assert "- Generation: prompt 883 · output 18 · total 901" in panel
    assert "- Grounding check: —" in panel, "a model call layer with no pair shows a dash"
    assert "Rerank:" not in panel, "non model layers are not in the token split"


def test_the_panel_shows_question_time_channel_path_total_and_confidence(api):
    panel = open_with(api, trace_reply())
    assert panel.startswith("### How many layers does it have?")
    assert "2026-10-13 20:11:58 · typed · path **answered** · total 1954 ms · confidence 0.947" in panel


def test_a_trace_stored_before_counting_says_not_recorded(api):
    panel = open_with(api, trace_reply(prompt_tokens=None, output_tokens=None))
    assert "**Tokens:** not recorded" in panel and "- Rewrite:" not in panel


def test_an_unreachable_trace_shows_the_api_error(api):
    api.replies["/traces/384"] = ConnectionError("Connection refused")
    assert ui.open_trace([[["r"], 384]], SimpleNamespace(index=[0, 0], row_value=None)) == \
        "API error: Connection refused"
