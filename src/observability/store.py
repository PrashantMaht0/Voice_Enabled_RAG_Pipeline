"""The trace store (spec 0006): one SQLite file beside the index, written only by the API.

Tracing is extra: any failure here is one WARNING and a counter, never a changed answer.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.utils.config import get

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    question TEXT NOT NULL,
    search_query TEXT NOT NULL,
    path TEXT NOT NULL,
    cache_level TEXT,
    confidence REAL,
    answer TEXT,
    sentences_kept INTEGER,
    sentences_dropped INTEGER,
    dropped_sentences TEXT,
    grounding_checker TEXT,
    retrieved TEXT,
    total_ms REAL NOT NULL DEFAULT 0,
    first_sentence_ms REAL,
    stt_ms REAL,
    first_audio_ms REAL,
    spoken_total_ms REAL,
    config TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    prompt_tokens INTEGER,
    output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS traces_created_at ON traces (created_at);
CREATE TABLE IF NOT EXISTS spans (
    trace_id INTEGER NOT NULL REFERENCES traces (id) ON DELETE CASCADE,
    layer TEXT NOT NULL,
    elapsed_ms REAL NOT NULL,
    details TEXT,
    prompt_tokens INTEGER,
    output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS spans_layer ON spans (layer);
CREATE INDEX IF NOT EXISTS spans_trace ON spans (trace_id);
"""
FINISH_FIELDS = ("search_query", "path", "cache_level", "confidence", "answer", "sentences_kept",
                 "sentences_dropped", "dropped_sentences", "grounding_checker", "retrieved", "total_ms",
                 "first_sentence_ms", "config", "error", "prompt_tokens", "output_tokens")
JSON_FIELDS = ("dropped_sentences", "retrieved", "config")
# Columns added after the first schema (spec 0007): files made earlier gain them on startup.
ADDED_COLUMNS = {"traces": ("prompt_tokens INTEGER", "output_tokens INTEGER"),
                 "spans": ("prompt_tokens INTEGER", "output_tokens INTEGER")}


def default_db_path() -> str:
    """Beside the index, so the Docker volume keeps it and deleting the index clears it."""
    qdrant = get("paths.qdrant_storage", "./vectordb/qdrant_storage")
    return get("observability.db_path", None) or os.path.join(
        os.path.dirname(os.path.abspath(qdrant)), "traces.sqlite3")


class TraceStore:
    """One SQLite file of traces (one row per request) and their spans, written only by the API.

    Old rows are deleted past `observability.max_traces`; a failed write is counted, never raised."""
    def __init__(self, path: str = None, max_traces: int = None):
        self.path = path or default_db_path()
        self.max_traces = int(max_traces if max_traces is not None else get("observability.max_traces", 10000))
        self.failed_writes = 0
        self._lock = threading.Lock()
        self._ready = False
        self._upgrade_warned = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if not self._ready:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            self._ready = self._upgrade(conn)
        return conn

    def _upgrade(self, conn: sqlite3.Connection) -> bool:
        """Adds the columns a file made by an earlier schema lacks (existing rows keep nulls). Runs under the
        store lock; a failure is one WARNING and is tried again on the next connection, never raised."""
        try:
            for table, columns in ADDED_COLUMNS.items():
                have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column in columns:
                    if column.split()[0] not in have:
                        try:
                            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e):
                                raise
            conn.commit()
            return True
        except Exception as e:
            if not self._upgrade_warned:
                print(f"[Traces] WARNING: could not upgrade the trace store ({e}); will try again.")
                self._upgrade_warned = True
            return False

    def _write(self, action: str, fn):
        """Runs one write; a failure is a WARNING and a counter, never an exception."""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    with conn:
                        return fn(conn)
                finally:
                    conn.close()
        except Exception as e:
            self.failed_writes += 1
            print(f"[Traces] WARNING: could not {action} ({e}); the answer is unaffected.")
            return None

    # --- writes ---

    def start(self, channel: str, question: str) -> Optional[int]:
        """A new trace, path in_progress, so its id can be sent before the answer exists."""
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return self._write("start a trace", lambda c: c.execute(
            "INSERT INTO traces (created_at, channel, question, search_query, path) VALUES (?, ?, ?, ?, ?)",
            (now, channel, question, question, "in_progress")).lastrowid)

    def finish(self, trace_id: Optional[int], spans: List[Dict] = (), **fields):
        """Completes a trace: its fields (unknown keys ignored, lists and dicts stored as JSON) and its spans, then
        trims the table to max_traces. No-op without a trace id."""
        if trace_id is None:
            return None
        values = {k: (json.dumps(v) if k in JSON_FIELDS and v is not None else v)
                  for k, v in fields.items() if k in FINISH_FIELDS}

        def write(c):
            if values:
                c.execute(f"UPDATE traces SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?",
                          (*values.values(), trace_id))
            c.executemany("INSERT INTO spans (trace_id, layer, elapsed_ms, details, prompt_tokens, output_tokens) "
                          "VALUES (?, ?, ?, ?, ?, ?)",
                          [(trace_id, s["layer"], float(s.get("elapsed_ms") or 0.0), s.get("details"),
                            s.get("prompt_tokens"), s.get("output_tokens")) for s in spans])
            # By id threshold, not a NOT IN scan (spec 0008, AC-18); ids only grow, so this keeps the newest
            c.execute("DELETE FROM traces WHERE id <= (SELECT MAX(id) FROM traces) - ?", (self.max_traces,))
            return trace_id
        return self._write("finish a trace", write)

    def add_voice(self, trace_id: int, **timings) -> bool:
        """Voice timings reported by the dashboard. False when the trace does not exist."""
        allowed = {k: v for k, v in timings.items() if k in ("stt_ms", "first_audio_ms", "spoken_total_ms")
                   and v is not None}
        result = self._write("add voice timings", lambda c: c.execute(
            f"UPDATE traces SET {', '.join(f'{k} = ?' for k in allowed) or 'id = id'} WHERE id = ?",
            (*allowed.values(), trace_id)).rowcount)
        return bool(result)

    # --- reads ---

    def query(self, sql: str, params=()) -> List[sqlite3.Row]:
        """Rows for a read only SQL query."""
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                conn.close()

    def recent(self, limit: int = 50, since: Optional[str] = None) -> List[Dict]:
        """The newest finished traces (summary fields, plus `answer` for the answered rule), newest first; only
        those created at or after `since` when given."""
        rows = self.query("SELECT id, created_at, channel, question, path, total_ms, answer, prompt_tokens, "
                          "output_tokens FROM traces WHERE path != 'in_progress'"
                          + (" AND created_at >= ?" if since else "") + " ORDER BY id DESC LIMIT ?",
                          (since, limit) if since else (limit,))
        return [dict(r) for r in rows]

    def get(self, trace_id: int) -> Optional[Dict]:
        """One trace with its JSON fields parsed and its spans in order; None for an unknown id."""
        rows = self.query("SELECT * FROM traces WHERE id = ?", (trace_id,))
        if not rows:
            return None
        trace = dict(rows[0])
        for key in JSON_FIELDS:
            if trace.get(key):
                trace[key] = json.loads(trace[key])
        trace["spans"] = [dict(r) for r in self.query(
            "SELECT layer, elapsed_ms, details, prompt_tokens, output_tokens FROM spans WHERE trace_id = ? "
            "ORDER BY rowid", (trace_id,))]
        return trace
