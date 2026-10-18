"""The single index worker: uploads, deletes and startup checks only request a rebuild; this one
thread runs them, never two at once, and requests that arrive during a build collapse into one follow up."""
import threading
import traceback
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

ERROR_CHARS = 200


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_reason(error: BaseException) -> str:
    """The exception class and its first line, at most ERROR_CHARS characters (the full trace goes to the log)."""
    first_line = (str(error).strip().splitlines() or [""])[0]
    reason = f"{type(error).__name__}: {first_line}" if first_line else type(error).__name__
    return reason[:ERROR_CHARS]


class IndexWorker:
    """State machine: idle -> building -> idle | failed; failed -> building on the next request. A request made
    while building (even one that then fails) runs exactly once more; a failure never retries by itself."""

    def __init__(self, build: Callable[[threading.Event], None]):
        self._build = build
        self.cancel = threading.Event()
        self._wake = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self.state = "idle"
        self.pending = False
        self.reason: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.builds = 0

    def start(self) -> "IndexWorker":
        """Starts the thread once; later calls do nothing."""
        with self._wake:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="index-worker", daemon=True)
                self._thread.start()
        return self

    def request_rebuild(self, reason: str):
        """Asks for one rebuild; returns at once."""
        with self._wake:
            self.pending = True
            self.reason = reason
            self._wake.notify()
        self.start()

    def _run(self):
        while True:
            with self._wake:
                while not self.pending and not self.cancel.is_set():
                    self._wake.wait()
                if self.cancel.is_set():
                    return
                self.pending = False
                self.state, self.started_at, self.finished_at = "building", now_iso(), None
                reason = self.reason
            print(f"[Index] Rebuilding ({reason}).")
            try:
                self._build(self.cancel)
            except BaseException as e:                       # noqa: BLE001 - the worker must keep running
                with self._wake:
                    self.state, self.last_error, self.finished_at = "failed", short_reason(e), now_iso()
                print(f"[Index] Rebuild failed; the previous index is still answering. {short_reason(e)}")
                traceback.print_exception(type(e), e, e.__traceback__)
                if self.cancel.is_set():
                    return
            else:
                with self._wake:
                    self.state, self.last_error, self.finished_at = "idle", None, now_iso()
                    self.builds += 1

    def status(self) -> Dict:
        with self._wake:
            return {"state": self.state, "pending": self.pending, "started_at": self.started_at,
                    "finished_at": self.finished_at, "last_error": self.last_error}

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Test helper: True once nothing is building or pending."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._wake:
                if self.state != "building" and not self.pending:
                    return True
            time.sleep(0.02)
        return False

    def stop(self, timeout: float = 5.0):
        """Asks a build in progress to stop and waits up to `timeout` seconds."""
        self.cancel.set()
        with self._wake:
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
