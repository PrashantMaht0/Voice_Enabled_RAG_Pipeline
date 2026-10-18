"""The dashboard's side of the request guards: it names the browser in X-VRAG-Client, turns refusals into plain
messages, and reports the index worker's status. HTTP calls are replaced by fakes."""
from types import SimpleNamespace

import pytest

import app as ui


class FakeResponse:
    def __init__(self, status_code, detail=None, headers=None, lines=()):
        self.status_code = status_code
        self._detail = detail
        self.headers = headers or {}
        self._lines = lines

    def json(self):
        if self._detail is None:
            raise ValueError("no JSON body")
        return {"detail": self._detail}

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def browser(host="203.0.113.7", forwarded=None):
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


# --- status codes become plain messages ---

@pytest.mark.parametrize("response, name, message", [
    (FakeResponse(409, "exists"), "a.md", "A document named a.md is already in the corpus. Delete it first to replace it."),
    (FakeResponse(413, "too big"), "big.pdf", "big.pdf is too large (limit 25 MB or 300 pages)."),
    (FakeResponse(413), None, "The request is too large."),
    (FakeResponse(429, "slow down", {"Retry-After": "17"}), None, "Too many requests, please wait 17 seconds."),
    (FakeResponse(503, "busy", {"Retry-After": "5"}), None, "The system is busy, please try again in a few seconds."),
    (FakeResponse(400, "Upload a .pdf, .md or .txt file."), "x.exe", "Upload a .pdf, .md or .txt file."),
    (FakeResponse(500, "Something went wrong (reference 42)."), None, "Something went wrong (reference 42)."),
    (FakeResponse(502), None, "API error 502."),
])
def test_each_status_code_maps_to_its_message(response, name, message):
    assert ui.api_message(response, name) == message


def test_validation_errors_show_the_apis_reasons():
    response = FakeResponse(422, [{"msg": "Value error, the question is longer than 1000 characters"},
                                  {"msg": "Input should be 'typed' or 'voice'"}])

    assert ui.api_message(response) == ("the question is longer than 1000 characters; "
                                        "Input should be 'typed' or 'voice'")


# --- X-VRAG-Client ---

def test_the_client_header_is_the_browsers_address(monkeypatch):
    monkeypatch.setattr(ui, "TRUST_FORWARDED", False)

    assert ui.client_headers(browser(forwarded="198.51.100.1")) == {"X-VRAG-Client": "203.0.113.7"}


def test_behind_a_trusted_proxy_the_first_forwarded_hop_is_used(monkeypatch):
    monkeypatch.setattr(ui, "TRUST_FORWARDED", True)

    headers = ui.client_headers(browser(host="10.0.0.2", forwarded="198.51.100.1, 10.0.0.2"))

    assert headers == {"X-VRAG-Client": "198.51.100.1"}


def test_without_a_request_or_an_address_no_header_is_sent(monkeypatch):
    monkeypatch.setattr(ui, "TRUST_FORWARDED", True)

    assert ui.client_headers(None) == {}
    assert ui.client_headers(SimpleNamespace(client=None, headers={})) == {}


@pytest.fixture
def no_corpus_calls(monkeypatch):
    """The table refresh after an upload or delete is not under test here."""
    monkeypatch.setattr(ui, "fetch_corpus", lambda: [])
    monkeypatch.setattr(ui, "corpus_filenames", lambda: [])


def test_a_question_sends_the_header_and_shows_a_429_in_the_chat(monkeypatch):
    monkeypatch.setattr(ui, "TRUST_FORWARDED", False)
    sent = {}

    def post(url, **kwargs):
        sent.update(kwargs)
        return FakeResponse(429, "Too many requests", {"Retry-After": "12"})

    monkeypatch.setattr(ui.requests, "post", post)

    outputs = list(ui.run_query("what port?", [], request=browser()))

    history, shown = outputs[-1][0], outputs[-1][1]
    assert sent["headers"] == {"X-VRAG-Client": "203.0.113.7"}
    assert shown[-1] == {"role": "assistant", "content": "Too many requests, please wait 12 seconds."}
    assert history == [], "A refused question is not added to the conversation sent next time."


def test_history_turns_sent_with_a_question_are_cut_to_2000_characters(monkeypatch):
    sent = {}

    def post(url, **kwargs):
        sent.update(kwargs)
        return FakeResponse(200, lines=())

    monkeypatch.setattr(ui.requests, "post", post)
    history = [{"role": "user", "content": "x" * 5000}, {"role": "assistant", "content": "short"}]

    list(ui.stream_events("next?", history))

    assert [len(t["content"]) for t in sent["json"]["history"]] == [2000, 5]


def test_delete_sends_the_header_with_a_30_second_timeout_and_starts_polling(monkeypatch, no_corpus_calls):
    monkeypatch.setattr(ui, "TRUST_FORWARDED", False)
    sent = {}

    def delete(url, **kwargs):
        sent.update(kwargs, url=url)
        return FakeResponse(200)

    monkeypatch.setattr(ui.requests, "delete", delete)

    _, _, status, timer, started = ui.delete_document("my notes.md", request=browser())

    assert sent["headers"] == {"X-VRAG-Client": "203.0.113.7"} and sent["timeout"] == 30
    assert sent["url"].endswith("/corpus/my%20notes.md")
    assert status == "Deleted **my notes.md**. Re indexing…"
    assert started > 0


def test_an_upload_refused_as_a_duplicate_says_so(monkeypatch, no_corpus_calls, tmp_path):
    monkeypatch.setattr(ui, "TRUST_FORWARDED", False)
    sent = {}

    def post(url, **kwargs):
        sent.update(kwargs)
        return FakeResponse(409, "exists")

    monkeypatch.setattr(ui.requests, "post", post)
    path = tmp_path / "a.md"
    path.write_text("# A")

    _, _, status, _, _ = ui.upload_document(str(path), request=browser())

    assert sent["headers"] == {"X-VRAG-Client": "203.0.113.7"}
    assert status == "Upload failed: A document named a.md is already in the corpus. Delete it first to replace it."


# --- index status in the Documents tab ---

def test_a_failed_build_shows_when_and_why_and_that_the_old_index_answers():
    status = {"state": "failed", "finished_at": "2026-10-15T21:00:05+00:00", "last_error": "ValueError: bad.pdf"}

    assert ui.failure_line(status) == ("Last indexing failed at 2026-10-15 21:00:05 UTC: ValueError: bad.pdf. "
                                       "The previous index is still answering.")
    assert ui.failure_line({"state": "idle"}) is None and ui.failure_line(None) is None


@pytest.mark.parametrize("status, busy", [
    ({"state": "building", "pending": False}, True),
    ({"state": "idle", "pending": True}, True),
    ({"state": "idle", "pending": False}, False),
    ({"state": "failed", "pending": False}, False),
])
def test_polling_continues_only_while_a_build_runs_or_is_queued(status, busy):
    assert ui.index_busy([], status) is busy


def test_without_a_status_polling_falls_back_to_pending_rows():
    assert ui.index_busy([["a.md", 0, ui.INDEXING]], None) is True
    assert ui.index_busy([["a.md", 12, "Indexed"]], None) is False


def test_the_question_box_limit_matches_the_api():
    assert ui.MAX_QUESTION_CHARS == 1000 and ui.MAX_TURN_CHARS == 2000
