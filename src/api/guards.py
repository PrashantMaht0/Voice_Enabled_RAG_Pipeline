"""Request guards: trusted hosts, the Origin rule, body size, per client rate limits, answer slots
and quiet errors. No login: these checks plus network binding are what protect the API.

Check order for every request: host (400), Origin (403), rate limit (429), body counting (413 while reading),
then the route's own validation (422) and, for questions, an answer slot (503)."""
import ipaddress
import math
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils.config import get

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
UPLOAD_PATH = "/api/upload"
VOICE_TIMINGS = re.compile(r"/api/traces/\d+/voice")
DRAIN_LIMIT = 64 * 1024 * 1024          # bytes read and discarded after a body goes over its limit
MAX_CLIENTS = 10_000
WINDOW_S = 60.0
CLIENT_HEADER = "x-vrag-client"
DEFAULTS = {
    "api.allowed_hosts": ["localhost", "127.0.0.1", "[::1]", "api"],
    "api.allowed_origins": [],
    "api.trusted_callers": ["127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "::1/128", "fc00::/7"],
    "api.max_body_kb": 64,
    "api.max_concurrent_answers": 4,
    "api.busy_retry_after_s": 5,
    "api.rate_limit.questions_per_min": 20,
    "api.rate_limit.changes_per_min": 5,
    "upload.max_mb": 25,
}

# Tests set values here instead of editing config.yaml.
overrides: Dict[str, object] = {}


def setting(key: str):
    """A guard setting: a test override, else config.yaml, else the default above."""
    if key in overrides:
        return overrides[key]
    return get(key, DEFAULTS.get(key))


# --- hosts and origins ---

def host_name(value: Optional[str]) -> Optional[str]:
    """The host part of a Host header or URL netloc, lowercase, without port or trailing dot; brackets kept for IPv6."""
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[:end + 1] if end > 0 else None
    host = value.rsplit(":", 1)[0] if value.count(":") == 1 else value
    return host.rstrip(".") or None


def normalized_origin(scheme: str, netloc: str) -> Optional[str]:
    """`scheme://host:port` in lowercase with the default port filled in."""
    scheme = (scheme or "").lower()
    host = host_name(netloc)
    if not host or scheme not in ("http", "https"):
        return None
    port = None
    rest = netloc.strip().lower()
    if rest.startswith("["):
        tail = rest[rest.find("]") + 1:]
        port = tail[1:] if tail.startswith(":") else None
    elif rest.count(":") == 1:
        port = rest.rsplit(":", 1)[1]
    port = port or ("443" if scheme == "https" else "80")
    return f"{scheme}://{host}:{port}"


def origin_allowed(origin: str, scheme: str, host_header: str) -> bool:
    """Same origin as the request itself, or listed in api.allowed_origins. `null` never is."""
    if origin.strip().lower() == "null":
        return False
    parts = urlsplit(origin.strip())
    wanted = normalized_origin(parts.scheme, parts.netloc)
    if wanted is None:
        return False
    if wanted == normalized_origin(scheme, host_header or ""):
        return True
    listed = {normalized_origin(urlsplit(o).scheme, urlsplit(o).netloc) for o in setting("api.allowed_origins") or []}
    return wanted in listed


# --- client identity ---

def parse_ip(value: Optional[str]):
    """An ip_address (IPv4 mapped IPv6 read as IPv4), or None when the value is not an IP."""
    try:
        ip = ipaddress.ip_address((value or "").strip().strip("[]"))
    except ValueError:
        return None
    return ip.ipv4_mapped if getattr(ip, "ipv4_mapped", None) else ip


def is_trusted(peer: Optional[str]) -> bool:
    """Whether the socket peer is a trusted caller (loopback, private networks by default)."""
    ip = parse_ip(peer)
    if ip is None:
        return False
    for network in setting("api.trusted_callers") or []:
        try:
            if ip in ipaddress.ip_network(network, strict=False):
                return True
        except ValueError:
            continue
    return False


def bucket_of(ip) -> str:
    """IPv4 addresses count alone; IPv6 addresses count by their /64."""
    if ip.version == 6:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def client_identity(peer: Optional[str], forwarded: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(client key or None when exempt, error message or None). A trusted peer is believed about the browser it
    forwards for; a trusted peer forwarding nobody (benchmarks, curl, tests) is exempt."""
    if is_trusted(peer):
        if forwarded is None:
            return None, None
        ip = parse_ip(forwarded) if len(forwarded) <= 45 else None
        if ip is None:
            return None, "X-VRAG-Client must be an IP address."
        return bucket_of(ip), None
    ip = parse_ip(peer)
    return (bucket_of(ip) if ip is not None else (peer or "unknown")), None


def limit_class(method: str, path: str) -> Optional[str]:
    """`questions`, `changes` or None for the routes the rate limits cover."""
    if method == "POST" and (path in ("/api/query", "/api/query/stream") or VOICE_TIMINGS.fullmatch(path)):
        return "questions"
    if (method == "POST" and path == UPLOAD_PATH) or (method == "DELETE" and (
            path.startswith("/api/corpus/") or path == "/api/cache")):
        return "changes"
    return None


class RateLimiter:
    """A sliding 60 second window per (client, class), in memory, for at most MAX_CLIENTS clients."""

    def __init__(self, max_clients: int = MAX_CLIENTS):
        self.max_clients = max_clients
        self._clients: "OrderedDict[str, Dict[str, deque]]" = OrderedDict()
        self._lock = threading.Lock()

    def _window(self, client: str, kind: str, now: float) -> deque:
        windows = self._clients.get(client)
        if windows is None:
            windows = self._clients[client] = {}
            while len(self._clients) > self.max_clients:
                self._clients.popitem(last=False)
        self._clients.move_to_end(client)
        window = windows.setdefault(kind, deque())
        while window and now - window[0] >= WINDOW_S:
            window.popleft()
        return window

    def retry_after(self, client: str, kind: str, limit: int, now: float = None) -> Optional[int]:
        """Seconds to wait when the client is at its limit, else None (the request may go ahead)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            window = self._window(client, kind, now)
            if len(window) < limit:
                if not window:
                    self._prune(client)
                return None
            return max(1, math.ceil(window[0] + WINDOW_S - now))

    def record(self, client: str, kind: str, now: float = None):
        """Counts one admitted request."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._window(client, kind, now).append(now)

    def admit(self, client: str, kind: str, limit: int, now: float = None) -> Tuple[Optional[int], Optional[float]]:
        """Checks and counts in one step, so a burst of parallel requests cannot all see an empty window.
        Returns (seconds to wait, None) when the client is at its limit, else (None, the stamp to refund)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            window = self._window(client, kind, now)
            if len(window) >= limit:
                return max(1, math.ceil(window[0] + WINDOW_S - now)), None
            window.append(now)
            return None, now

    def refund(self, client: str, kind: str, stamp: float):
        """Uncounts an admitted request that was refused (a 4xx or 5xx reply)."""
        with self._lock:
            window = self._clients.get(client, {}).get(kind)
            if window is not None:
                try:
                    window.remove(stamp)
                except ValueError:
                    pass                                 # it already left the window

    def _prune(self, client: str):
        windows = self._clients.get(client, {})
        if all(not w for w in windows.values()):
            self._clients.pop(client, None)

    def __len__(self):
        return len(self._clients)


limiter = RateLimiter()


# --- answer slots ---

class SlotToken:
    """One taken answer slot; release is safe to call more than once."""

    def __init__(self, slots: "AnswerSlots"):
        self._slots, self._released = slots, False
        self._lock = threading.Lock()

    def release(self):
        with self._lock:
            if self._released:
                return
            self._released = True
        self._slots._free()


class AnswerSlots:
    """At most api.max_concurrent_answers questions inside the answer pipeline at once."""

    def __init__(self):
        self._busy = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> Optional[SlotToken]:
        with self._lock:
            if self._busy >= int(setting("api.max_concurrent_answers")):
                return None
            self._busy += 1
        return SlotToken(self)

    def _free(self):
        with self._lock:
            self._busy = max(0, self._busy - 1)

    @property
    def busy(self) -> int:
        return self._busy


slots = AnswerSlots()


def busy_response() -> JSONResponse:
    """503 with Retry-After when every answer slot is taken."""
    retry = int(setting("api.busy_retry_after_s"))
    return JSONResponse({"detail": "The system is busy, please try again in a few seconds."}, status_code=503,
                        headers={"Retry-After": str(retry)})


# --- quiet errors ---

def reference_id() -> str:
    """A short id that ties an error reply to its log line when there is no trace."""
    return uuid.uuid4().hex[:8]


def log_error(reference, error: BaseException):
    """The full exception, with the reference the client was given."""
    print(f"[API] ERROR reference {reference}: {type(error).__name__}: {error}")
    traceback.print_exception(type(error), error, error.__traceback__)


def error_message(reference) -> str:
    return f"Something went wrong (reference {reference})."


async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    """Any exception a route did not handle: a reference for the client, the details in the log."""
    reference = reference_id()
    log_error(reference, error)
    return JSONResponse({"detail": error_message(reference)}, status_code=500)


# --- the middleware ---

def body_limit(path: str) -> int:
    """Bytes a request body may have on this route."""
    if path == UPLOAD_PATH:
        return int(float(setting("upload.max_mb")) * 1024 * 1024) + 64 * 1024
    return int(setting("api.max_body_kb")) * 1024


async def send_json(send, status: int, detail: str, headers: Dict[str, str] = None):
    """A small JSON reply sent straight from the middleware."""
    response = JSONResponse({"detail": detail}, status_code=status, headers=headers)
    await send({"type": "http.response.start", "status": response.status_code,
                "headers": response.raw_headers})
    await send({"type": "http.response.body", "body": response.body})


class RequestGuard:
    """ASGI middleware applying the host, Origin, rate and body size checks to every HTTP request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        method, path = scope["method"], scope["path"]

        host = host_name(headers.get("host"))
        allowed = {host_name(h) for h in setting("api.allowed_hosts") or []}
        if host is None or host not in allowed:
            return await send_json(send, 400, "Invalid host.")

        origin = headers.get("origin")
        if method in UNSAFE_METHODS and origin is not None and \
                not origin_allowed(origin, scope.get("scheme", "http"), headers.get("host", "")):
            return await send_json(send, 403, "Cross site requests are not allowed.")

        kind = limit_class(method, path)
        client, stamp = None, None
        if kind:
            peer = (scope.get("client") or (None, None))[0]
            client, problem = client_identity(peer, headers.get(CLIENT_HEADER))
            if problem:
                return await send_json(send, 400, problem)
            if client is not None:
                limit = int(setting(f"api.rate_limit.{kind}_per_min"))
                wait, stamp = limiter.admit(client, kind, limit)
                if wait is not None:
                    return await send_json(send, 429, f"Too many requests, please wait {wait} seconds.",
                                           {"Retry-After": str(wait)})

        limit = body_limit(path)
        declared = headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return await send_json(send, 413, "Request body too large.")

        state = {"received": 0, "over": False, "started": False, "refunded": False}

        def refund_once():
            if kind and client is not None and not state["refunded"]:
                state["refunded"] = True
                limiter.refund(client, kind, stamp)

        async def counted_receive():
            message = await receive()
            if message["type"] != "http.request" or state["over"]:
                return message
            state["received"] += len(message.get("body", b""))
            if state["received"] > limit:
                state["over"] = True
                drained, more = 0, message.get("more_body", False)
                while more and drained < DRAIN_LIMIT:   # read the rest so the client sees the 413
                    extra = await receive()
                    if extra["type"] != "http.request":
                        break
                    drained += len(extra.get("body", b""))
                    more = extra.get("more_body", False)
                return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message):
            if message["type"] == "http.response.start":
                if state["over"]:
                    state["started"] = True
                    refund_once()
                    return await send_json(send, 413, "Request body too large.")
                if message["status"] >= 400:
                    refund_once()
            elif state["started"]:
                return                                   # the route's own reply is replaced by the 413
            await send(message)

        await self.app(scope, counted_receive, guarded_send)


async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    """422 naming each problem without echoing the submitted value, which may not even be valid JSON (an
    infinite number) and would otherwise turn the reply into a 500."""
    detail = [{k: v for k, v in problem.items() if k not in ("input", "ctx")} for problem in error.errors()]
    return JSONResponse({"detail": jsonable_encoder(detail)}, status_code=422)


def install_guards(app: FastAPI) -> FastAPI:
    """Adds the request guard and the quiet error handlers to an app (main.py, and tests that want them)."""
    app.add_middleware(RequestGuard)
    app.add_exception_handler(RequestValidationError, invalid_request)
    app.add_exception_handler(Exception, unexpected_error)
    return app
