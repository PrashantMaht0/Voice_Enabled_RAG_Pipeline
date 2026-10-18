"""The API (FastAPI, under /api): answering questions, managing documents, and reading traces and metrics.

Every question, typed or spoken, plain or streamed, goes through answer_events: the empty query rule, the
cache, retrieval and the confidence gate, streamed generation with a sentence by sentence grounding check,
and one trace row in the trace store."""
import json
import os
import threading
import time
import uuid
from typing import Dict, List, Literal, Optional

import ollama
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from qdrant_client import models

from src.api import guards
from src.ingestion import figures, tables
from src.ingestion import upload_checks
from src.ingestion.chunking import DocumentChunker
from src.ingestion.upload_checks import upload_name_problem  # noqa: F401  (the API's name check; tests import it here)
from src.ingestion.loader import DocumentLoader
from src.llm.llm_client import LocalLLMClient, generated_tokens
from src.observability.store import TraceStore
from src.pipeline import core
from src.pipeline.limits import check_history, check_query
from src.pipeline.settings import Settings
from src.pipeline.sources import append_citations, build_citations  # noqa: F401  (imported from here by tests)
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import HybridRetriever
from src.utils.config import get
from src.utils.metrics import PipelineTracer
from src.utils.query_cache import QueryCache
from src.vector_db.index_worker import IndexWorker
from src.vector_db.vector_store import INDEX_FORMAT_VERSION
from src.verification import grounding

router = APIRouter()

RAW_DIR = get("paths.raw_data_dir", "data/raw")
DOCUMENT_TYPES = (".pdf", ".md", ".txt")
INCOMING = ".incoming"                 # uploads are checked here before they join the corpus

# Heavy models are loaded once per process, not per request.
_retriever = None
_llm = None
_rewriter = None
_cache = None
_traces = None
_worker = None
_build_lock = threading.Lock()
_retire_timers = []


def get_retriever() -> HybridRetriever:
    """The retriever (and through it the index), created on first use."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def get_llm() -> LocalLLMClient:
    """The answer model client, created on first use."""
    global _llm
    if _llm is None:
        _llm = LocalLLMClient()   # model from llm.generation_model in config.yaml
    return _llm

def get_rewriter() -> QueryRewriter:
    """The follow up rewriter, created on first use."""
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter

def get_cache() -> QueryCache:
    """The answer cache, created on first use."""
    global _cache
    if _cache is None:
        _cache = QueryCache(
            # Resolved lazily so the cache never forces the retrieval stack to load.
            embedder_provider=lambda: get_retriever().store.db_manager.embedder,
            raw_dir=RAW_DIR,
            max_entries=get("cache.max_entries", 256),
            similarity_threshold=get("cache.similarity_threshold", 0.95),
            enabled=get("cache.enabled", True),
        )
    return _cache


def get_traces() -> TraceStore:
    """The trace store, opened on first use."""
    global _traces
    if _traces is None:
        _traces = TraceStore()
    return _traces


def warmup():
    """Loads every model at server boot so the first query is already warm."""
    get_retriever()
    get_llm()
    get_rewriter()
    get_cache()
    tables.check_table_parser()   # one ERROR line here, not one per page
    figures.check_figure_parser()
    grounding.check_grounding_checker()
    ensure_index_format()


class QueryRequest(BaseModel):
    """A question, with the conversation so far (used only to rewrite follow ups into standalone questions).
    Sizes are limited so one request cannot send a huge prompt to the models."""
    query: str
    history: List[Dict[str, str]] = []
    channel: Literal["typed", "voice"] = "typed"     # recorded in the trace

    @field_validator("query")
    @classmethod
    def query_fits(cls, value: str) -> str:
        return check_query(value, int(get("api.max_query_chars", 1000)))

    @field_validator("history")
    @classmethod
    def history_fits(cls, value: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return check_history(value, int(get("api.max_history_turns", 6)), int(get("api.max_turn_chars", 2000)))


def corpus_files(raw_dir: str = None) -> list:
    """The corpus fingerprint: sorted `name:size:mtime_ns` of every document file (not the incoming folder)."""
    raw_dir = raw_dir or RAW_DIR
    if not os.path.isdir(raw_dir):
        return []
    entries = []
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        if os.path.isfile(path) and name.lower().endswith(DOCUMENT_TYPES):
            stat = os.stat(path)
            entries.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return entries


def build_generation(db_manager=None, cancel: threading.Event = None):
    """Builds a new index generation from data/raw and swaps it in. The only way the API builds:
    one at a time under a lock. Raises on failure, leaving the previous generation and its marker serving."""
    with _build_lock:
        # Embedded Qdrant is single-handle, so this reuses the process-wide manager.
        db_manager = db_manager or get_retriever().store.db_manager
        corpus = corpus_files()                          # taken before reading, so a later upload is not missed
        print("[Ingestion] Starting vectorization...")
        documents = DocumentLoader().load_documents()
        chunks = DocumentChunker().chunk_documents(documents)
        generation = db_manager.build_generation(chunks, cancel=cancel)
        previous = db_manager.commit_generation(generation, index_fingerprint(), corpus)
        get_cache().invalidate(f"index generation {generation.number}")
        retire(db_manager, previous.number)
        print(f"[Ingestion] Generation {generation.number} is live with {len(chunks)} chunks.")


def retire(db_manager, number: int):
    """Deletes a replaced generation after index.retire_after_s, once questions in flight are done with it."""
    timer = threading.Timer(float(get("index.retire_after_s", 30)), db_manager.drop_generation, [number])
    timer.daemon = True
    timer.start()
    _retire_timers[:] = [t for t in _retire_timers if t.is_alive()] + [timer]


def rebuild_index(db_manager=None) -> bool:
    """A build at startup (format or reader settings changed): True only on full success."""
    try:
        build_generation(db_manager)
        return True
    except Exception as e:
        print(f"[Ingestion] Pipeline failed; the previous index is still serving: {e}")
        return False


def get_index_worker() -> IndexWorker:
    """The single index worker, started on first use (so apps built without the lifespan get one too)."""
    global _worker
    if _worker is None:
        _worker = IndexWorker(lambda cancel: build_generation(cancel=cancel))
    return _worker.start()


def start_index(db_manager=None):
    """After warmup: remove generations the marker does not name, start the worker, and queue a rebuild when the
    corpus folder changed while the API was off (an old marker without a corpus counts as matching)."""
    db_manager = db_manager or get_retriever().store.db_manager
    db_manager.cleanup_orphans()
    worker = get_index_worker()
    indexed = db_manager.index_corpus()
    if indexed is not None and indexed != corpus_files():
        worker.request_rebuild("documents changed while the API was off")


def stop_index():
    """On shutdown: stop a build in progress and cancel pending retirements."""
    if _worker is not None:
        _worker.stop()
    for timer in _retire_timers:
        timer.cancel()


def index_fingerprint() -> str:
    """Every reader setting that changes the chunks: tables, then figures (empty when off)."""
    return tables.table_fingerprint() + figures.figure_fingerprint()


def ensure_index_format():
    """Rebuilds once at startup when the index predates the current chunk layout."""
    db_manager = get_retriever().store.db_manager
    found = db_manager.index_format()
    if found < INDEX_FORMAT_VERSION:
        print(f"[Ingestion] Index format {found} is older than {INDEX_FORMAT_VERSION}; rebuilding from data/raw.")
        rebuild_index(db_manager)
    elif found == INDEX_FORMAT_VERSION and db_manager.index_tables() != index_fingerprint():
        print(f"[Ingestion] Table or figure settings changed ({db_manager.index_tables()} -> {index_fingerprint()}); "
              "rebuilding from data/raw.")
        rebuild_index(db_manager)
    elif found > INDEX_FORMAT_VERSION:
        print(f"[Ingestion] WARNING: index format {found} is newer than this code ({INDEX_FORMAT_VERSION}); "
              "leaving it as is.")


_ollama_check = {"at": None, "status": None}


def ollama_status() -> str:
    """`online` or `offline`, asked with a short timeout and reused for health.cache_s seconds."""
    now = time.monotonic()
    if _ollama_check["at"] is not None and now - _ollama_check["at"] < float(get("health.cache_s", 10)):
        return _ollama_check["status"]
    try:
        ollama.Client(timeout=float(get("health.ollama_timeout_s", 3))).list()
        status = "online"
    except Exception:
        status = "offline"
    _ollama_check.update(at=now, status=status)
    return status


@router.get("/health")
def health_check():
    """Whether Ollama answers, and which answer model, table reader, figure reader and grounding checker are
    actually in effect (a configured one that failed to load reports `none`). A plain def: the Ollama check
    must never block the event loop."""
    return {
        "status": "healthy",
        "ollama_status": ollama_status(),
        "generation_model": get_llm().generation_model,
        "table_parser_effective": tables.effective_parser(),
        "figure_parser_effective": figures.effective_parser(),
        "grounding_checker_effective": grounding.effective_checker_name(),
        "pipeline": "VocalRAG-Zero"
    }


@router.get("/cache")
async def cache_stats():
    """Hit rates and invalidation counts for the two-level query cache."""
    return get_cache().stats()


@router.delete("/cache")
async def clear_cache():
    """Empties the answer cache; returns the cache stats afterwards."""
    get_cache().invalidate("manual")
    return get_cache().stats()


@router.get("/corpus")
def list_corpus():
    """Each document in data/raw with its indexed chunk count (`Pending` until indexing finishes)."""
    if not os.path.exists(RAW_DIR):
        return []
    manager = get_retriever().store.db_manager
    rows = []
    for name in sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(DOCUMENT_TYPES)):
        chunk_count = manager.qdrant_client.count(
            collection_name=manager.collection_name,
            count_filter=models.Filter(must=[models.FieldCondition(key="metadata.source",
                                                                   match=models.MatchValue(value=name))]),
            exact=True).count
        rows.append({"file": name, "chunks": chunk_count, "status": "Indexed" if chunk_count else "Pending"})
    return rows


@router.delete("/corpus/{filename:path}")
def delete_document(filename: str):
    """Removes a source file and queues a rebuild; returns at once (the previous index answers until the new
    one is ready, and the deleted file is no longer cited)."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(RAW_DIR, safe_name)

    # Refuse anything that resolves outside the corpus directory
    if os.path.realpath(os.path.dirname(file_path)) != os.path.realpath(RAW_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"{safe_name} is not in the corpus.")

    os.remove(file_path)
    get_index_worker().request_rebuild(f"{safe_name} deleted")
    return {"deleted": safe_name, "index": "queued"}


@router.get("/index")
def index_status():
    """The index worker: idle, building or failed (with a short reason), and the generation answering now."""
    status = get_index_worker().status()
    status["generation"] = get_retriever().store.db_manager.active.number
    return status


def current_generation():
    """The generation answering now, or None for a retriever without generations (tests)."""
    manager = getattr(getattr(get_retriever(), "store", None), "db_manager", None)
    active = getattr(manager, "active", None)
    return getattr(active, "number", None)


def retrieve(search_query: str, tracer) -> tuple:
    """(chunks, top score, per chunk details); details are empty for a retriever without them."""
    retriever = get_retriever()
    detailed = getattr(retriever, "search_and_rerank_detailed", None)
    if detailed:
        return detailed(search_query, tracer=tracer)
    chunks, score = retriever.search_and_rerank(search_query, tracer=tracer)
    return chunks, score, []


def generate_tokens(search_query: str, chunks: List[str], tracer=None):
    """The answer model's tokens (counts to the request's tracer); a client without streaming yields its whole
    answer at once and reports no counts."""
    llm = get_llm()
    if hasattr(llm, "stream_generate"):
        yield from llm.stream_generate(search_query, chunks, tracer=tracer)
    else:
        yield llm.generate(search_query, chunks)


def trace_config() -> Dict:
    """What produced an answer: models, the reader fingerprint (when known) and the grounding checker."""
    llm = get_llm()
    config = {"generation_model": getattr(llm, "generation_model", None),
              "rewriter_model": getattr(llm, "rewriter_model", None),
              "answer_style": getattr(llm, "answer_style", None),
              "grounding_checker": grounding.effective_checker_name()}
    if tables.parser_checked() and figures.parser_checked():
        config["index"] = index_fingerprint()
    return config


def server_settings() -> Settings:
    """The core's settings from config.yaml (the gate threshold and the checker come from the live objects)."""
    return Settings.from_config(get)


def answer_events(request: QueryRequest):
    """The one answer pipeline, driven for the server: `core.answer` decides everything,
    and this loop answers its effects through the server's objects and hooks. Both query endpoints use it."""
    started = time.perf_counter()
    tracer = PipelineTracer()
    traces = get_traces()
    trace_id = traces.start(request.channel, request.query)
    record = {"path": "error", "search_query": request.query}
    state = {}                                  # the open answer stream, its chunks and the grounding checker

    def perform(effect: Dict):
        kind = effect["effect"]
        if kind == "Span":
            tracer.add_static_span(effect["layer"], effect["details"], effect.get("ms") or 0.0)
        elif kind == "Rewrite":
            return get_rewriter().resolve_query(effect["query"], effect["history"], tracer=tracer)
        elif kind == "CacheLookup":
            cached, level = get_cache().get(effect["search_query"])
            if cached is None or cached.get("generation") != current_generation():
                return None                     # a miss, or answered from an index that has since been replaced
            return {"payload": cached, "level": level}
        elif kind == "Retrieve":
            generation = current_generation()
            chunks, score, details = retrieve(effect["search_query"], tracer)
            return {"chunks": chunks, "score": score, "details": details, "generation": generation,
                    "threshold": get_retriever().confidence_threshold}
        elif kind == "Generate":
            state.update(checker=grounding.effective_checker(), chunks=effect["chunks"],
                         stream=iter(generate_tokens(effect["search_query"], effect["chunks"], tracer)))
            unavailable = state["checker"].name is None and grounding.checker_unavailable()
            return {"checker": state["checker"].name,
                    "unchecked": "the grounding checker is unavailable" if unavailable else None}
        elif kind == "NextToken":
            try:
                return {"token": next(state["stream"])}
            except StopIteration:
                prompt_tokens = (tracer.layer_tokens("generation") or (None, None))[0]
                return {"done": True, "prompt_tokens": prompt_tokens, "output_tokens": generated_tokens(tracer)}
        elif kind == "CheckSentence":
            return grounding.answer_check(state["checker"], effect, state["chunks"], tracer)
        elif kind == "CacheStore":
            get_cache().put(effect["search_query"], effect["payload"])
        else:
            raise ValueError(f"unknown effect {kind!r}")
        return None

    steps = core.answer({"query": request.query, "history": request.history, "channel": request.channel},
                        server_settings(), record)
    try:
        reply = None
        while True:
            try:
                item = steps.send(reply)
            except StopIteration:
                break
            if "effect" in item:
                reply = perform(item)
                continue
            reply = None
            if item["type"] == "meta":
                item = {"type": "meta", "trace_id": trace_id, **{k: v for k, v in item.items() if k != "type"}}
            elif item["type"] == "done":
                item["trace"] = tracer.as_list()
            yield item
    except GeneratorExit:
        record["path"] = "aborted"
        raise
    except Exception as e:
        reference = trace_id if trace_id is not None else guards.reference_id()
        guards.log_error(reference, e)
        record.update(path="error", error=str(e))
        yield {"type": "error", "detail": guards.error_message(reference)}
    finally:
        steps.close()
        if state.get("stream") is not None and hasattr(state["stream"], "close"):
            state["stream"].close()             # an answer stopped early never counts its tokens
        tracer.print_trace()
        dropped = record.get("dropped") or []
        prompt_tokens, output_tokens = tracer.token_totals()
        traces.finish(trace_id, spans=tracer.as_store_list(), path=record["path"],
                      search_query=record["search_query"], cache_level=record.get("cache_level"),
                      confidence=record.get("confidence"), answer=record.get("answer"),
                      sentences_kept=record.get("kept"), sentences_dropped=len(dropped) if record.get("answer") else None,
                      dropped_sentences=dropped or None, grounding_checker=record.get("grounding_checker"),
                      retrieved=record.get("retrieved"), total_ms=(time.perf_counter() - started) * 1000,
                      first_sentence_ms=record.get("first_sentence_ms"), config=trace_config(),
                      error=record.get("error"), prompt_tokens=prompt_tokens, output_tokens=output_tokens)


@router.post("/query")
def manual_query(request: QueryRequest):
    """The whole answer at once; same pipeline as the stream. Takes an answer slot or answers 503."""
    token = guards.slots.try_acquire()
    if token is None:
        return guards.busy_response()
    meta, done = {}, None
    try:
        for event in answer_events(request):
            if event["type"] == "meta":
                meta = event
            elif event["type"] == "done":
                done = event
            elif event["type"] == "error":
                raise HTTPException(status_code=500, detail=event["detail"])
    finally:
        token.release()
    return {
        "answer": done["answer"],
        "retrieved_context": meta.get("retrieved_context", []),
        "confidence_score": meta.get("confidence_score"),
        "search_query": meta.get("search_query"),
        "citations": done["citations"],
        "grounded": done["grounded"],
        "dropped_citations": done.get("dropped_citations", []),
        "dropped_sentences": done["dropped_sentences"],
        "path": done["path"],
        "trace_id": meta.get("trace_id"),
        "cache": done.get("cache"),
        "trace": done["trace"],
        "total_latency_ms": done["total_ms"],
    }


class SlotStreamingResponse(StreamingResponse):
    """A stream that gives its answer slot back when the response ends, even if its generator never ran."""

    def __init__(self, content, token, **kwargs):
        super().__init__(content, **kwargs)
        self.token = token

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.token.release()


@router.post("/query/stream")
def stream_query(request: QueryRequest):
    """ndjson events: meta, one sentence per approved sentence, then done (or error). The answer slot is taken
    before the stream starts and released exactly once, by whichever ends first."""
    token = guards.slots.try_acquire()
    if token is None:
        return guards.busy_response()

    def events():
        try:
            for event in answer_events(request):
                yield json.dumps(event) + "\n"
        finally:
            token.release()

    return SlotStreamingResponse(events(), token, media_type="application/x-ndjson")


MAX_VOICE_MS = 600_000                 # ten minutes; a real voice turn is far shorter


class VoiceTimings(BaseModel):
    """Speech timings the dashboard measured for a voice turn, in milliseconds (finite, 0 to ten minutes, so
    the metrics stay serializable)."""
    stt_ms: Optional[float] = Field(None, ge=0, le=MAX_VOICE_MS, allow_inf_nan=False)
    first_audio_ms: Optional[float] = Field(None, ge=0, le=MAX_VOICE_MS, allow_inf_nan=False)
    spoken_total_ms: Optional[float] = Field(None, ge=0, le=MAX_VOICE_MS, allow_inf_nan=False)


@router.post("/traces/{trace_id}/voice")
def add_voice_timings(trace_id: int, timings: VoiceTimings):
    """The dashboard reports a voice turn's speech timings; only the API writes traces."""
    if not get_traces().add_voice(trace_id, **timings.model_dump()):
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}.")
    return {"updated": trace_id}


@router.get("/metrics")
def get_metrics(window: Literal["1h", "24h", "7d", "all"] = "24h"):
    """Counts, rates, stage and voice timings and the latest quality scores over a time window."""
    from src.observability import metrics
    return metrics.compute(get_traces(), window)


@router.get("/traces")
def list_traces(limit: int = Query(50, ge=1, le=200), window: Literal["1h", "24h", "7d", "all"] = "all"):
    """The most recent finished traces in the window, newest first: summary fields, tokens and whether each
    counts as answered."""
    from src.observability import metrics
    items = get_traces().recent(limit, since=metrics.since(window))
    for item in items:
        item["answered"] = metrics.is_answered(item["path"], item.pop("answer"))
    return items


@router.get("/traces/{trace_id}")
def get_trace(trace_id: int):
    """One full trace: stages, retrieved sources, dropped sentences, the answer and voice timings."""
    trace = get_traces().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}.")
    return trace


def check_contents(path: str, name: str):
    """The file is what its name says (shared check), as the API's HTTP error."""
    try:
        upload_checks.check_contents(path, name, int(get("upload.max_pdf_pages", 300)))
    except upload_checks.UploadRefused as e:
        raise HTTPException(status_code=e.status, detail=e.message)


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Checks an uploaded PDF, Markdown or text file, adds it to data/raw in one step, and queues a rebuild.
    A plain def: copying and reading the PDF must not block the event loop."""
    name = os.path.basename(file.filename or "")
    problem = upload_name_problem(name)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    os.makedirs(os.path.join(RAW_DIR, INCOMING), exist_ok=True)
    if name.lower() in {existing.lower() for existing in os.listdir(RAW_DIR)}:
        raise HTTPException(status_code=409, detail=f"A document named {name} is already in the corpus.")

    limit = int(float(get("upload.max_mb", 25)) * 1024 * 1024)
    part = os.path.join(RAW_DIR, INCOMING, f"{uuid.uuid4().hex}.part")
    try:
        written = 0
        with open(part, "wb") as out:
            while True:
                block = file.file.read(upload_checks.CHUNK)
                if not block:
                    break
                written += len(block)
                if written > limit:
                    raise HTTPException(status_code=413, detail=f"{name} is larger than "
                                                                f"{get('upload.max_mb', 25)} MB.")
                out.write(block)
        check_contents(part, name)
        try:
            os.link(part, os.path.join(RAW_DIR, name))    # fails if the name appeared meanwhile
        except FileExistsError:
            raise HTTPException(status_code=409, detail=f"A document named {name} is already in the corpus.")
    finally:
        if os.path.exists(part):
            os.remove(part)
    get_index_worker().request_rebuild(f"{name} uploaded")
    return {"filename": name, "status": "queued"}
