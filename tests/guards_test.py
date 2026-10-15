"""Request guards (spec 0008): trusted hosts, the Origin rule, body size, rate limits, answer slots and quiet
errors, driven through a small app with the real middleware so no model loads."""
import ipaddress

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api import guards
from src.api.guards import AnswerSlots, RateLimiter

TRUSTED = ("127.0.0.1", 50000)
REMOTE = ("203.0.113.9", 50000)


@pytest.fixture(autouse=True)
def fresh_guards(monkeypatch):
    """Each test gets its own limiter, slots and settings, so counts never leak between tests."""
    monkeypatch.setattr(guards, "overrides", {})
    monkeypatch.setattr(guards, "limiter", RateLimiter())
    monkeypatch.setattr(guards, "slots", AnswerSlots())


@pytest.fixture
def calls():
    return []


@pytest.fixture
def app(calls):
    """Stand in routes on the real paths, behind the real guards. `calls` records every route that ran."""
    app = FastAPI()

    @app.get("/api/health")
    def health():
        calls.append("health")
        return {"ok": True}

    @app.post("/api/query")
    def query(body: dict):
        calls.append("query")
        return {"ok": True}

    @app.post("/api/query/stream")
    def refused(body: dict):
        calls.append("refused")
        return {"ok": False} if body.get("bad") else {"ok": True}

    @app.post("/api/upload")
    async def upload(request: Request):
        data = await request.body()
        calls.append("upload")
        return {"bytes": len(data)}

    @app.delete("/api/cache")
    def clear():
        calls.append("clear")
        return {}

    @app.delete("/api/corpus/{name}")
    def delete(name: str):
        calls.append("delete")
        return {}

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("secret database password in this message")

    return guards.install_guards(app)


def make_client(app, peer=TRUSTED, **kwargs):
    return TestClient(app, base_url="http://localhost:8000", client=peer, **kwargs)


# --- AC-1 trusted hosts ---

@pytest.mark.parametrize("host", ["attacker.example", "localhost.attacker.example", "127.0.0.2", ""])
def test_a_host_outside_the_allowed_list_is_400_before_any_route_runs(app, calls, host):
    """covers: AC-1"""
    response = make_client(app).get("/api/health", headers={"Host": host})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid host."}
    assert calls == []


@pytest.mark.parametrize("host", ["localhost:8000", "[::1]:8000", "LOCALHOST.", "127.0.0.1", "api:8000"])
def test_allowed_hosts_pass_with_any_port_case_or_trailing_dot(app, host):
    """covers: AC-1"""
    assert make_client(app).get("/api/health", headers={"Host": host}).status_code == 200


def test_a_host_added_to_the_config_is_accepted(app):
    """covers: AC-1"""
    guards.overrides["api.allowed_hosts"] = ["rag.example.com"]

    assert make_client(app).get("/api/health", headers={"Host": "rag.example.com"}).status_code == 200
    assert make_client(app).get("/api/health", headers={"Host": "localhost"}).status_code == 400


@pytest.mark.parametrize("value, expected", [
    ("LocalHost:8000", "localhost"), ("[::1]:8000", "[::1]"), ("[::1]", "[::1]"), ("example.com.", "example.com"),
    ("", None), (None, None), ("[::1", None),
])
def test_host_name_drops_the_port_and_trailing_dot_and_lowercases(value, expected):
    """covers: AC-1"""
    assert guards.host_name(value) == expected


# --- AC-2 Origin rule ---

@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost:9999", "https://localhost:8000",
                                    "file://", "not a url"])
def test_a_cross_site_write_is_403_and_does_nothing(app, calls, origin):
    """covers: AC-2"""
    response = make_client(app).post("/api/upload", content=b"x", headers={"Origin": origin})

    assert response.status_code == 403
    assert calls == []


def test_a_write_without_an_origin_is_allowed(app, calls):
    """covers: AC-2"""
    assert make_client(app).post("/api/upload", content=b"x").status_code == 200
    assert calls == ["upload"]


@pytest.mark.parametrize("origin, host", [
    ("http://localhost:8000", "localhost:8000"),
    ("HTTP://LOCALHOST:8000", "localhost:8000"),
    ("http://localhost", "localhost"),                # default port 80 on both sides
    ("http://localhost:80", "localhost"),
])
def test_a_same_origin_write_is_allowed(app, origin, host):
    """covers: AC-2"""
    response = make_client(app).post("/api/upload", content=b"x", headers={"Origin": origin, "Host": host})

    assert response.status_code == 200


def test_an_origin_listed_in_the_config_is_allowed(app):
    """covers: AC-2"""
    guards.overrides["api.allowed_origins"] = ["https://dashboard.example"]

    response = make_client(app).post("/api/upload", content=b"x", headers={"Origin": "https://dashboard.example:443"})

    assert response.status_code == 200


@pytest.mark.parametrize("method", ["delete"])
def test_every_unsafe_method_is_checked(app, calls, method):
    """covers: AC-2"""
    response = getattr(make_client(app), method)("/api/cache", headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert calls == []


def test_reads_are_never_origin_checked(app):
    """covers: AC-2"""
    assert make_client(app).get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 200


# --- AC-3 body size ---

def test_a_declared_body_over_the_limit_is_413_without_running_the_route(app, calls):
    """covers: AC-3"""
    big = b"{" + b" " * (64 * 1024) + b"}"

    response = make_client(app).post("/api/query", content=big, headers={"Content-Type": "application/json"})

    assert response.status_code == 413
    assert calls == []


def test_a_chunked_body_over_the_limit_is_413_not_a_parse_error(app):
    """No Content-Length: the body is counted as it arrives, and the reply is rewritten to 413. covers: AC-3"""
    guards.overrides["api.max_body_kb"] = 1

    def chunks():
        for _ in range(4):
            yield b"x" * 600

    response = make_client(app).post("/api/query", content=chunks(), headers={"Content-Type": "application/json"})

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large."}


def test_a_body_at_the_limit_passes(app):
    """covers: AC-3"""
    guards.overrides["api.max_body_kb"] = 1
    body = b'{"a": "' + b"x" * (1024 - 9) + b'"}'

    assert len(body) == 1024
    assert make_client(app).post("/api/query", content=body,
                                 headers={"Content-Type": "application/json"}).status_code == 200


def test_uploads_get_their_own_larger_limit(app):
    """upload.max_mb plus 64 KB on /api/upload, api.max_body_kb everywhere else. covers: AC-3"""
    guards.overrides.update({"api.max_body_kb": 1, "upload.max_mb": 0.5})

    assert guards.body_limit("/api/upload") == 512 * 1024 + 64 * 1024
    assert guards.body_limit("/api/query") == 1024
    assert make_client(app).post("/api/upload", content=b"x" * 100_000).json() == {"bytes": 100_000}
    assert make_client(app).post("/api/upload", content=b"x" * (600 * 1024)).status_code == 413


# --- AC-13 rate limits ---

def ask(client, headers=None):
    return client.post("/api/query", json={"q": "x"}, headers=headers or {})


def test_a_forwarded_client_over_its_question_limit_gets_429_with_retry_after(app):
    """covers: AC-13"""
    guards.overrides["api.rate_limit.questions_per_min"] = 2
    client, browser = make_client(app), {"X-VRAG-Client": "203.0.113.7"}

    codes = [ask(client, browser).status_code for _ in range(2)]
    third = ask(client, browser)

    assert codes == [200, 200]
    assert third.status_code == 429
    assert 1 <= int(third.headers["Retry-After"]) <= 60
    assert third.json()["detail"] == f"Too many requests, please wait {third.headers['Retry-After']} seconds."


def test_one_client_at_its_limit_does_not_slow_down_another(app):
    """covers: AC-13"""
    guards.overrides["api.rate_limit.questions_per_min"] = 1
    client = make_client(app)
    ask(client, {"X-VRAG-Client": "203.0.113.7"})

    assert ask(client, {"X-VRAG-Client": "203.0.113.7"}).status_code == 429
    assert ask(client, {"X-VRAG-Client": "198.51.100.4"}).status_code == 200


def test_a_trusted_caller_without_the_header_is_never_rate_limited(app):
    """Benchmarks, curl and local tools keep full speed. covers: AC-13"""
    guards.overrides["api.rate_limit.questions_per_min"] = 2
    client = make_client(app)

    assert {ask(client).status_code for _ in range(10)} == {200}


def test_an_untrusted_peer_is_counted_by_its_own_address_whatever_it_forwards(app):
    """covers: AC-13"""
    guards.overrides["api.rate_limit.questions_per_min"] = 2
    client = make_client(app, peer=REMOTE)

    codes = [ask(client, {"X-VRAG-Client": f"198.51.100.{i}"}).status_code for i in range(3)]

    assert codes == [200, 200, 429]


def test_the_test_clients_non_ip_peer_is_counted_too(app):
    """covers: AC-13"""
    guards.overrides["api.rate_limit.questions_per_min"] = 1
    client = make_client(app, peer=("testclient", 50000))

    assert [ask(client).status_code for _ in range(2)] == [200, 429]


@pytest.mark.parametrize("value", ["not-an-ip", "203.0.113.7, 10.0.0.1", "x" * 46])
def test_a_forwarded_header_that_is_not_an_ip_is_400(app, calls, value):
    """covers: AC-13"""
    response = ask(make_client(app), {"X-VRAG-Client": value})

    assert response.status_code == 400
    assert response.json() == {"detail": "X-VRAG-Client must be an IP address."}
    assert calls == []


def test_changes_have_their_own_limit_across_upload_delete_and_cache(app):
    """covers: AC-13"""
    guards.overrides["api.rate_limit.changes_per_min"] = 2
    client, browser = make_client(app), {"X-VRAG-Client": "203.0.113.7"}

    first = client.post("/api/upload", content=b"x", headers=browser).status_code
    second = client.delete("/api/corpus/a.md", headers=browser).status_code
    third = client.delete("/api/cache", headers=browser).status_code

    assert (first, second, third) == (200, 200, 429)
    assert ask(client, browser).status_code == 200, "Questions are counted separately from changes."


def test_refused_requests_do_not_use_up_the_limit(app):
    """Only admitted requests count; 4xx replies do not. covers: AC-13"""
    guards.overrides["api.rate_limit.questions_per_min"] = 1
    client, browser = make_client(app), {"X-VRAG-Client": "203.0.113.7"}

    invalid = client.post("/api/query", content=b"not json", headers={**browser, "Content-Type": "application/json"})

    assert invalid.status_code == 422
    assert ask(client, browser).status_code == 200


def test_retry_after_counts_down_to_when_the_oldest_request_leaves_the_window():
    """covers: AC-13"""
    limiter = RateLimiter()
    limiter.record("c", "questions", now=100.0)
    limiter.record("c", "questions", now=130.0)

    assert limiter.retry_after("c", "questions", limit=2, now=140.0) == 20
    assert limiter.retry_after("c", "questions", limit=2, now=159.5) == 1
    assert limiter.retry_after("c", "questions", limit=2, now=160.0) is None, "The oldest request has left."


def test_at_most_max_clients_are_tracked_and_the_least_recently_used_goes_first():
    """covers: AC-13"""
    limiter = RateLimiter(max_clients=3)
    for name in ["a", "b", "c"]:
        limiter.record(name, "questions", now=0.0)
    limiter.record("a", "questions", now=1.0)           # a is now the most recently used

    limiter.record("d", "questions", now=2.0)

    assert len(limiter) == 3
    assert limiter.retry_after("b", "questions", limit=1, now=3.0) is None, "b was dropped."
    assert limiter.retry_after("a", "questions", limit=1, now=3.0) is not None


def test_empty_windows_are_pruned():
    """covers: AC-13"""
    limiter = RateLimiter()
    limiter.record("c", "questions", now=0.0)

    limiter.retry_after("c", "questions", limit=5, now=61.0)

    assert len(limiter) == 0


def test_ipv6_clients_are_grouped_by_their_64_and_mapped_ipv4_reads_as_ipv4():
    """covers: AC-13"""
    one, other = guards.client_identity("127.0.0.1", "2001:db8:1:2::1")[0], \
        guards.client_identity("127.0.0.1", "2001:db8:1:2:ffff::9")[0]

    assert one == other == "2001:db8:1:2::/64"
    assert guards.parse_ip("::ffff:127.0.0.1") == ipaddress.ip_address("127.0.0.1")
    assert guards.is_trusted("::ffff:127.0.0.1")


def test_a_missing_peer_counts_as_unknown():
    """covers: AC-13"""
    assert guards.client_identity(None, None) == ("unknown", None)


@pytest.mark.parametrize("method, path, kind", [
    ("POST", "/api/query", "questions"), ("POST", "/api/query/stream", "questions"),
    ("POST", "/api/upload", "changes"), ("DELETE", "/api/corpus/a.md", "changes"), ("DELETE", "/api/cache", "changes"),
    ("GET", "/api/health", None), ("GET", "/api/cache", None), ("GET", "/api/index", None),
])
def test_only_questions_and_changes_are_rate_limited(method, path, kind):
    """covers: AC-13"""
    assert guards.limit_class(method, path) == kind


# --- AC-12 answer slots ---

def test_slots_stop_at_the_configured_cap_and_free_on_release():
    """covers: AC-12"""
    guards.overrides["api.max_concurrent_answers"] = 2
    slots = AnswerSlots()

    first, second, third = slots.try_acquire(), slots.try_acquire(), slots.try_acquire()
    first.release()

    assert first is not None and second is not None and third is None
    assert slots.busy == 1
    assert slots.try_acquire() is not None


def test_a_slot_released_twice_frees_only_one_place():
    """covers: AC-12"""
    guards.overrides["api.max_concurrent_answers"] = 2
    slots = AnswerSlots()
    token, _ = slots.try_acquire(), slots.try_acquire()

    token.release()
    token.release()

    assert slots.busy == 1


def test_the_busy_reply_is_503_with_retry_after():
    """covers: AC-12"""
    response = guards.busy_response()

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


# --- AC-14 quiet errors ---

def test_an_unexpected_failure_returns_a_reference_and_logs_the_details(app, capsys):
    """covers: AC-14"""
    response = make_client(app, raise_server_exceptions=False).get("/api/boom")

    detail = response.json()["detail"]
    reference = detail.removeprefix("Something went wrong (reference ").removesuffix(").")
    log = capsys.readouterr().out
    assert response.status_code == 500
    assert detail == f"Something went wrong (reference {reference})." and len(reference) == 8
    assert "password" not in response.text
    assert f"reference {reference}: RuntimeError: secret database password" in log


@pytest.mark.parametrize("enabled, code", [(True, 200), (False, 404)])
def test_api_docs_are_served_only_when_enabled(monkeypatch, enabled, code):
    """main.py reads api.docs_enabled when imported; the lifespan (model warmup) never runs here. covers: AC-14"""
    import importlib

    from src.utils import config
    real_get = config.get
    monkeypatch.setattr(config, "get", lambda key, default=None: enabled if key == "api.docs_enabled" else
                        real_get(key, default))
    import main
    main = importlib.reload(main)
    client = TestClient(main.app, base_url="http://localhost:8000")

    assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [code] * 3
