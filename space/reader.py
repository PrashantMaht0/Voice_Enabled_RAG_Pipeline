"""Reading one upload for the live demo: the shared upload checks, the pipeline's own
ingestion (layoutlmv3 tables, Qwen2.5 VL figures), the limits, and the per address daily cap. No Gradio here, so
it runs and is tested anywhere; app.py wraps it for the Space."""
import contextlib
import hashlib
import io
import ipaddress
import os
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.ingestion import upload_checks
from src.ingestion.upload_checks import UploadRefused

DAY_S = 24 * 3600


def limit(name: str, default: int) -> int:
    """A Space variable (MAX_MB, MAX_PAGES, MAX_CHUNKS, DAILY_READS_PER_IP), else the default."""
    return int(os.environ.get(name, default))


def client_ip(forwarded_for: Optional[str], peer: Optional[str]) -> str:
    """The visitor's address: the rightmost public hop of X-Forwarded-For (what the first proxy we trust saw; the
    leftmost hops are whatever the client claimed), else the socket peer."""
    return client_hop(forwarded_for, peer)[0]


def client_hop(forwarded_for: Optional[str], peer: Optional[str]) -> Tuple[str, str]:
    """(address, where it came from, as "hop k of n" or "peer"); the second part is safe to log, the first is not."""
    hops = [h.strip() for h in (forwarded_for or "").split(",") if h.strip()]
    for k in range(len(hops) - 1, -1, -1):
        try:
            if ipaddress.ip_address(hops[k]).is_global:
                return hops[k], f"hop {k + 1} of {len(hops)}"
        except ValueError:
            continue
    return peer or "unknown", "peer"


class DailyCap:
    """Successful reads per address in a rolling 24 hours, in memory (lost on restart); safe across threads."""

    def __init__(self):
        self._reads: Dict[str, Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allowed(self, ip: str, now: float = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            start, count = self._reads.get(ip, (now, 0))
            return now - start >= DAY_S or count < limit("DAILY_READS_PER_IP", 10)

    def record(self, ip: str, now: float = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            start, count = self._reads.get(ip, (now, 0))
            self._reads[ip] = (now, 1) if now - start >= DAY_S else (start, count + 1)


def prepare(source: str, name: str) -> Tuple[str, Optional[int]]:
    """Checks the upload and copies it, under its own name, into a fresh private folder: (folder, PDF page count)."""
    upload_checks.check_name(name)
    max_mb = limit("MAX_MB", 10)
    if os.path.getsize(source) > max_mb * 1024 * 1024:
        raise UploadRefused("too_large", 413, f"{name} is larger than {max_mb} MB.")
    folder = tempfile.mkdtemp(prefix="vrag-")
    path = os.path.join(folder, name)
    shutil.copyfile(source, path)
    try:
        pages = upload_checks.check_contents(path, name, limit("MAX_PAGES", 5))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder, pages


def log_name(name: str) -> str:
    """A short fingerprint of a visitor's file name, so the log can tell uploads apart without naming them."""
    return hashlib.sha256((name or "").encode("utf-8")).hexdigest()[:8]


def read(folder: str) -> List[Dict]:
    """The pipeline's own reading and chunking of the one file in `folder`, as the demo's chunk records. The
    ingestion's own log lines name the file and quote reader errors, so they are discarded here."""
    from src.ingestion.chunking import DocumentChunker
    from src.ingestion.loader import DocumentLoader
    with contextlib.redirect_stdout(io.StringIO()):
        chunks = DocumentChunker().chunk_documents(DocumentLoader(raw_data_dir=folder).load_documents())
    max_chunks = limit("MAX_CHUNKS", 2000)
    if len(chunks) > max_chunks:
        raise UploadRefused("too_many_chunks", 413, f"The document makes {len(chunks)} chunks; the limit is {max_chunks}.")
    return [{"id": c["id"], "doc_hash": c["metadata"].get("doc_hash"), "chunk_index": c["metadata"].get("chunk_index"),
             "page": c["metadata"].get("page"), "kind": c["metadata"].get("kind", "text"), "content": c["content"]}
            for c in chunks]


def discard(*paths: Optional[str]) -> None:
    """Deletes the private copy and Gradio's cached upload; nothing of the visitor's file is kept."""
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
