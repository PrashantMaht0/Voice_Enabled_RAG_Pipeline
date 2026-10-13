"""Metrics over stored traces (spec 0006, trace store child): counts, rates, p50 and p95, latest quality."""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.text_sanitizer import is_refusal

WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7), "all": None}
STAGE_LAYERS = ("rewrite", "bm25", "vector", "fusion", "rerank", "generation", "grounding")
VOICE_FIELDS = ("stt_ms", "first_audio_ms", "spoken_total_ms")
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "data" / "benchmarks"


def nearest_rank(values: List[float], q: float) -> Optional[float]:
    """Value at rank ceil(q * n) of the sorted list (as spec 0003); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(1, math.ceil(q * len(ordered))) - 1], 1)


def percentiles(values: List[float]) -> Dict:
    """p50, p95 and the count of the values."""
    return {"p50": nearest_rank(values, 0.5), "p95": nearest_rank(values, 0.95), "n": len(values)}


def since(window: str) -> Optional[str]:
    """The UTC timestamp a window starts at; None for `all`."""
    span = WINDOWS[window]
    return None if span is None else (datetime.now(timezone.utc) - span).isoformat(timespec="milliseconds")


def latest_quality(benchmarks_dir: Path = BENCHMARKS_DIR) -> Optional[Dict]:
    """The newest sweep run's outcome: date, winner settings, golden, faithfulness, tables, figures."""
    runs = sorted(p for p in benchmarks_dir.glob("sweep-*.json") if not p.name.endswith("-ragas.json"))
    for path in reversed(runs):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        winner = run.get("winner")
        record = (run.get("configs") or {}).get(winner) if winner else None
        if not record:
            continue
        ragas = record.get("ragas") if isinstance(record.get("ragas"), dict) else {}
        golden = record.get("golden") or {}
        return {"file": path.name, "finished": run.get("finished") or run.get("updated"), "winner": winner,
                "settings": record.get("settings"),
                "golden_answered": golden.get("answered"), "golden_in_scope": golden.get("in_scope"),
                "golden_refused": golden.get("refused"), "golden_out_of_scope": golden.get("out_of_scope"),
                "faithfulness": ragas.get("faithfulness"),
                "tables": (record.get("tables") or {}).get("right_value_and_page"),
                "figures": (record.get("figures") or {}).get("right_value_and_page")}
    return None


def compute(store, window: str = "24h", benchmarks_dir: Path = BENCHMARKS_DIR) -> Dict:
    """The metrics block `GET /api/metrics` returns for a window: counts per path, rates, stage and voice percentiles
    for answered requests, failed writes and the latest quality card.

    A cache hit counts as answered unless the cached answer was a refusal; errors and aborted requests are left out of
    the rates."""
    start = since(window)
    where, params = ("WHERE path != 'in_progress'" + (" AND created_at >= ?" if start else "")), \
        ((start,) if start else ())
    rows = [dict(r) for r in store.query(
        f"SELECT id, path, answer, grounding_checker, sentences_dropped, total_ms, first_sentence_ms, "
        f"stt_ms, first_audio_ms, spoken_total_ms FROM traces {where}", params)]

    cache = [r for r in rows if r["path"] == "cache"]
    cache_refused = [r for r in cache if is_refusal(r["answer"] or "")]
    count = lambda path: sum(r["path"] == path for r in rows)
    errors, aborted = count("error"), count("aborted")
    counts = {
        "asked": len(rows),
        "answered": count("answered") + len(cache) - len(cache_refused),
        "refused_gate": count("gated") + len(cache_refused),
        "refused_grounding": count("refused_grounding"),
        "refused_model": count("refused_model"),
        "refused_empty": count("empty_query"),
        "cache_hits": len(cache),
        "aborted": aborted,
        "errors": errors,
    }
    served = len(rows) - errors - aborted
    refused = sum(counts[k] for k in ("refused_gate", "refused_grounding", "refused_model", "refused_empty"))
    checked = [r for r in rows if r["path"] in ("answered", "refused_grounding") and r["grounding_checker"]
               and "unchecked" not in r["grounding_checker"]]
    rates = {
        "cache_hit_rate": round(len(cache) / served, 3) if served else None,
        "refusal_rate": round(refused / served, 3) if served else None,
        "grounding_pass_rate": round(sum(not r["sentences_dropped"] for r in checked) / len(checked), 3)
        if checked else None,
    }

    answered_ids = [r["id"] for r in rows if r["path"] == "answered"]
    stages = {}
    if answered_ids:
        marks = ",".join("?" * len(answered_ids))
        spans = store.query(f"SELECT layer, elapsed_ms FROM spans WHERE trace_id IN ({marks})", answered_ids)
        for layer in STAGE_LAYERS:
            values = [s["elapsed_ms"] for s in spans if s["layer"] == layer]
            if values:
                stages[layer] = percentiles(values)
    answered_rows = [r for r in rows if r["path"] == "answered"]
    for field in ("first_sentence_ms", "total_ms"):
        stages[field] = percentiles([r[field] for r in answered_rows if r[field] is not None])
    voice = {field: percentiles([r[field] for r in rows if r[field] is not None]) for field in VOICE_FIELDS}

    return {"window": window, "counts": counts, "rates": rates, "stages": stages, "voice": voice,
            "failed_writes": store.failed_writes, "quality": latest_quality(benchmarks_dir)}
