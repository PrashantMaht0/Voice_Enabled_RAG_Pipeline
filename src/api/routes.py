"""The API (FastAPI, under /api): answering questions, managing documents, and reading traces and metrics.

Every question, typed or spoken, plain or streamed, goes through answer_events: the empty query rule, the
cache, retrieval and the confidence gate, streamed generation with a sentence by sentence grounding check,
and one trace row in the trace store."""
import json
import os
import time
from typing import Dict, List, Literal, Optional

import ollama
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import models

from src.ingestion import figures, tables
from src.ingestion.chunking import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.llm.llm_client import LocalLLMClient
from src.observability.store import TraceStore
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import HybridRetriever
from src.utils.config import get
from src.utils.metrics import PipelineTracer
from src.utils.query_cache import QueryCache
from src.utils.text_sanitizer import CITATION_TAG, SOURCE_HEADER, is_refusal, parse_tag, usable_history
from src.vector_db.vector_store import INDEX_FORMAT_VERSION
from src.verification import grounding
from src.verification.citations import strip_ungrounded_citations

router = APIRouter()

RAW_DIR = get("paths.raw_data_dir", "data/raw")
DOCUMENT_TYPES = (".pdf", ".md", ".txt")

# Heavy models are loaded once per process, not per request.
_retriever = None
_llm = None
_rewriter = None
_cache = None
_traces = None


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
    """A question, with the conversation so far (used only to rewrite follow ups into standalone questions)."""
    query: str
    history: List[Dict[str, str]] = []
    channel: str = "typed"           # typed | voice, recorded in the trace (spec 0006)


def build_citations(chunks: List[str]) -> List[Dict]:
    """Provenance derived from the chunks that were actually retrieved, in retrieval order."""
    seen, citations = set(), []
    for chunk in chunks:
        match = SOURCE_HEADER.match(chunk)
        if not match:
            continue
        key = parse_tag(match)
        if key in seen:
            continue
        seen.add(key)
        source, kind, number = key
        citations.append({
            "source": source,
            "page": number if kind == "Page" else None,
            "chunk": number if kind == "Chunk" else None,
        })
    return citations


def citation_key(citation: Dict) -> tuple:
    """The (source, kind, number) a citation's tag parses to."""
    if citation.get("page") is not None:
        return citation["source"], "Page", int(citation["page"])
    return citation["source"], "Chunk", int(citation["chunk"])


def citation_tag(citation: Dict) -> str:
    """The `[Source: file | Page: N]` (or `Chunk: N`) tag for a citation."""
    source, kind, number = citation_key(citation)
    return f"[Source: {source} | {kind}: {number}]"


def append_citations(answer: str, citations: List[Dict]) -> str:
    """Appends source tags the model failed to emit itself."""
    if not citations or not answer.strip():
        return answer

    already = {parse_tag(m) for m in CITATION_TAG.finditer(answer)}
    missing = [c for c in citations if citation_key(c) not in already]
    if not missing:
        return answer

    tags = " ".join(citation_tag(c) for c in missing)
    return f"{answer.rstrip()} {tags}"


def log_citations(citations: List[Dict]) -> None:
    """One terminal line per answered query, naming each cited file and page."""
    if not citations:
        return
    cited = ", ".join(
        f"{c['source']} p.{c['page']}" if c.get("page") is not None else f"{c['source']} chunk {c['chunk']}"
        for c in citations
    )
    print(f"[CITATIONS] {cited}")


def resolve_search_query(request: QueryRequest, tracer=None) -> str:
    """Resolves follow-up pronouns into a standalone query."""
    search_query = request.query

    # Refused turns carry no subject and would drag the next question off topic.
    history = usable_history(request.history)

    if history:
        if tracer:
            tracer.start_span("rewrite")
        search_query = get_rewriter().resolve_query(search_query, history)
        if tracer:
            tracer.end_span("rewrite", f"Standalone query — {search_query!r}")

    return search_query


def rebuild_index(db_manager=None) -> bool:
    """Rebuilds both indexes from every document in data/raw. True only on full success."""
    # Reuses the process-wide manager: embedded Qdrant is single-handle.
    db_manager = db_manager or get_retriever().store.db_manager
    db_manager.clear_index_format()
    print("[Ingestion] Starting vectorization...")
    try:
        documents = DocumentLoader().load_documents()
        chunks = DocumentChunker().chunk_documents(documents)
        if chunks:
            db_manager.ingest_chunks(chunks)
        else:
            # Clear both stores, or BM25 keeps serving deleted documents.
            db_manager.reset_indexes()
        get_cache().invalidate("corpus reindexed" if chunks else "corpus emptied")
        db_manager.mark_index_format(index_fingerprint())
        print(f"[Ingestion] Successfully indexed {len(chunks)} chunks." if chunks
              else "[Ingestion] Corpus is empty — indexes cleared.")
        return True
    except Exception as e:
        print(f"[Ingestion] Pipeline failed: {e}")
        return False


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


@router.get("/health")
async def health_check():
    """Whether Ollama answers, which models it has, and which table reader, figure reader and grounding
    checker are actually in effect (a configured one that failed to load reports `none`)."""
    try:
        models_response = ollama.list()
        available_models = [m.model for m in models_response.models] if hasattr(models_response, "models") else []
        ollama_status = "online"
        ollama_error = None
    except Exception as e:
        available_models = []
        ollama_status = "offline"
        # Keep the detail out of the status field so callers can render a short label
        ollama_error = str(e)

    return {
        "status": "healthy",
        "ollama_status": ollama_status,
        "ollama_error": ollama_error,
        "available_local_models": available_models,
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
async def list_corpus():
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
async def delete_document(filename: str):
    """Removes a source file and rebuilds the indexes from what remains."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(RAW_DIR, safe_name)

    # Refuse anything that resolves outside the corpus directory
    if os.path.realpath(os.path.dirname(file_path)) != os.path.realpath(RAW_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"{safe_name} is not in the corpus.")

    os.remove(file_path)
    get_cache().invalidate(f"{safe_name} removed")
    rebuild_index()
    return {"deleted": safe_name}


def retrieve(search_query: str, tracer) -> tuple:
    """(chunks, top score, per chunk details); details are empty for a retriever without them."""
    retriever = get_retriever()
    detailed = getattr(retriever, "search_and_rerank_detailed", None)
    if detailed:
        return detailed(search_query, tracer=tracer)
    chunks, score = retriever.search_and_rerank(search_query, tracer=tracer)
    return chunks, score, []


def generate_tokens(search_query: str, chunks: List[str]):
    """The answer model's tokens; a client without streaming yields its whole answer at once."""
    llm = get_llm()
    if hasattr(llm, "stream_generate"):
        yield from llm.stream_generate(search_query, chunks)
    else:
        yield llm.generate(search_query, chunks)


def trace_config() -> Dict:
    """What produced an answer: models, the reader fingerprint (when known) and the grounding checker."""
    llm = get_llm()
    config = {"generation_model": getattr(llm, "generation_model", None),
              "rewriter_model": getattr(llm, "rewriter_model", None),
              "answer_style": getattr(llm, "answer_style", None),
              "grounding_checker": grounding.effective_checker_name()}
    if tables._effective_parser is not None and figures._effective_parser is not None:
        config["index"] = index_fingerprint()
    return config


def with_kinds(citations: List[Dict], details: List[Dict]) -> List[Dict]:
    """Each citation gains the kinds (text, table, figure) of the retrieved chunks behind it (spec 0006)."""
    order = {"text": 0, "table": 1, "figure": 2}
    for c in citations:
        kinds = {d.get("kind") or "text" for d in details
                 if d.get("source") == c["source"] and d.get("page") == c.get("page")}
        c["kinds"] = sorted(kinds or {"text"}, key=lambda k: order.get(k, 3))
    return citations


def kept_citations(chunks: List[str], result) -> List[Dict]:
    """Today's citations (every retrieved source), minus sources only a dropped sentence cited."""
    citations = build_citations(chunks)
    tags = lambda texts: {parse_tag(m) for t in texts for m in CITATION_TAG.finditer(t)}
    only_dropped = tags(result.dropped_raw) - tags(result.kept)
    return [c for c in citations if citation_key(c) not in only_dropped]


def answer_events(request: QueryRequest):
    """The one answer pipeline (spec 0006): meta, approved sentences, done. Both query endpoints use it."""
    started = time.perf_counter()
    tracer = PipelineTracer()
    traces = get_traces()
    trace_id = traces.start(request.channel, request.query)
    record = {"path": "error", "search_query": request.query}

    def elapsed():
        """Milliseconds since the request arrived."""
        return (time.perf_counter() - started) * 1000

    def sentence(index: int, text: str) -> Dict:
        """A sentence event (citation tags removed for display), noting when the first one went out."""
        if record.get("first_sentence_ms") is None:
            record["first_sentence_ms"] = elapsed()
        return {"type": "sentence", "index": index, "text": grounding.plain(text) or text}

    try:
        search_query = resolve_search_query(request, tracer)
        record["search_query"] = search_query

        if grounding.is_empty_query(search_query):
            # Nothing to search for: no cache, retrieval or answer model (spec 0006, AC-5)
            answer = grounding.EMPTY_QUERY_ANSWER
            tracer.add_static_span("empty_query", "No content words — refused before retrieval")
            record.update(path="empty_query", answer=answer)
            yield {"type": "meta", "trace_id": trace_id, "confidence_score": None, "search_query": search_query,
                   "citations": [], "retrieved_context": [], "cache": None}
            yield sentence(0, answer)
            yield {"type": "done", "answer": answer, "citations": [], "grounded": None, "path": "empty_query",
                   "dropped_sentences": [], "dropped_citations": [], "cache": None, "trace": tracer.as_list(),
                   "total_ms": elapsed()}
            return

        cached, cache_level = get_cache().get(search_query)
        if cached is not None:
            tracer.add_static_span("cache", f"{cache_level.upper()} hit — retrieval and generation skipped")
            log_citations(cached.get("citations", []))
            record.update(path="cache", cache_level=cache_level, confidence=cached.get("confidence_score"),
                          answer=cached["answer"], dropped=cached.get("dropped_sentences", []),
                          retrieved=cached.get("retrieved"), grounding_checker=cached.get("grounding_checker"))
            yield {"type": "meta", "trace_id": trace_id, "confidence_score": cached.get("confidence_score"),
                   "search_query": search_query, "citations": cached.get("citations", []),
                   "retrieved_context": cached.get("retrieved_context", []), "retrieved": cached.get("retrieved") or [],
                   "cache": cache_level}
            for i, text in enumerate(cached.get("sentences") or grounding.split_sentences(cached["answer"])):
                yield sentence(i, text)
            yield {"type": "done", **{k: cached.get(k) for k in ("answer", "citations", "grounded", "path")},
                   "dropped_sentences": cached.get("dropped_sentences", []), "dropped_citations":
                   cached.get("dropped_citations", []), "cache": cache_level, "trace": tracer.as_list(),
                   "total_ms": elapsed()}
            return

        chunks, confidence, details = retrieve(search_query, tracer)
        confidence = float(confidence)
        record.update(confidence=confidence, retrieved=details)
        threshold = get_retriever().confidence_threshold
        gated = confidence < threshold
        tracer.add_static_span("confidence", f"Top chunk: {confidence:.3f}, threshold: {threshold} — "
                                             f"{'FAIL' if gated else 'PASS'}")
        yield {"type": "meta", "trace_id": trace_id, "confidence_score": confidence, "search_query": search_query,
               "citations": [] if gated else with_kinds(build_citations(chunks), details),
               "retrieved_context": [] if gated else chunks, "retrieved": [] if gated else details, "cache": None}

        if gated:
            answer, citations, path, result = "Insufficient context.", [], "gated", None
            sentences = [answer]
            yield sentence(0, answer)
        else:
            judge = grounding.SentenceJudge(chunks)
            buffer, index, gen_started = grounding.SentenceBuffer(), 0, time.perf_counter()
            for token in generate_tokens(search_query, chunks):
                for piece in buffer.feed(token):
                    for approved in judge.judge(piece):
                        yield sentence(index, approved)
                        index += 1
            for piece in buffer.finish():
                for approved in judge.judge(piece):
                    yield sentence(index, approved)
                    index += 1
            for approved in judge.finish():
                yield sentence(index, approved)
                index += 1
            result = judge.result
            gen_ms = (time.perf_counter() - gen_started - result.seconds) * 1000
            tracer.add_static_span("generation", f"LLM call — {getattr(get_llm(), 'last_eval_count', 0)} tokens "
                                                 f"generated", gen_ms)
            tracer.add_static_span("grounding", f"{result.checker or 'unchecked'}: {result.checked} checked, "
                                                f"{len(result.dropped)} dropped"
                                                + (f", {result.invalid} invalid replies kept" if result.invalid else "")
                                                + (f" — unchecked: {result.unchecked_reason}"
                                                   if result.unchecked_reason else ""), result.seconds * 1000)
            if judge.refused:
                answer, citations, path, sentences = "Insufficient context.", [], "refused_grounding", \
                    ["Insufficient context."]
                yield sentence(index, answer)
            else:
                sentences = result.kept
                answer, dropped_citations = strip_ungrounded_citations(
                    grounding.join_sentences(result.kept), chunks)
                record["dropped_citations"] = dropped_citations
                citations = with_kinds(kept_citations(chunks, result), details)
                answer = append_citations(answer, citations)
                path = "refused_model" if result.model_refused and all(is_refusal(k) for k in result.kept) \
                    else "answered"

        log_citations(citations)
        grounded = result.grounded if result else None
        dropped = result.dropped if result else []
        checker = None
        if result and result.checker:
            checker = result.checker + (f" (unchecked: {result.unchecked_reason})" if result.unchecked_reason else "")
        payload = {"answer": answer, "retrieved_context": [] if path == "gated" else chunks,
                   "confidence_score": confidence, "search_query": search_query, "citations": citations,
                   "grounded": grounded, "dropped_citations": record.get("dropped_citations", []),
                   "dropped_sentences": dropped, "sentences": sentences, "path": path,
                   "retrieved": details, "grounding_checker": checker}
        get_cache().put(search_query, payload)
        record.update(path=path, answer=answer, dropped=dropped, grounding_checker=checker,
                      kept=len(sentences) if result else None)
        yield {"type": "done", "answer": answer, "citations": citations, "grounded": grounded, "path": path,
               "dropped_sentences": dropped, "dropped_citations": payload["dropped_citations"], "cache": None,
               "trace": tracer.as_list(), "total_ms": elapsed()}
    except GeneratorExit:
        record["path"] = "aborted"
        raise
    except Exception as e:
        record.update(path="error", error=str(e))
        yield {"type": "error", "detail": str(e)}
    finally:
        tracer.print_trace()
        dropped = record.get("dropped") or []
        traces.finish(trace_id, spans=tracer.as_list(), path=record["path"],
                      search_query=record["search_query"], cache_level=record.get("cache_level"),
                      confidence=record.get("confidence"), answer=record.get("answer"),
                      sentences_kept=record.get("kept"), sentences_dropped=len(dropped) if record.get("answer") else None,
                      dropped_sentences=dropped or None, grounding_checker=record.get("grounding_checker"),
                      retrieved=record.get("retrieved"), total_ms=elapsed(),
                      first_sentence_ms=record.get("first_sentence_ms"), config=trace_config(),
                      error=record.get("error"))


@router.post("/query")
def manual_query(request: QueryRequest):
    """The whole answer at once; same pipeline as the stream."""
    meta, done = {}, None
    for event in answer_events(request):
        if event["type"] == "meta":
            meta = event
        elif event["type"] == "done":
            done = event
        elif event["type"] == "error":
            raise HTTPException(status_code=500, detail=event["detail"])
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


@router.post("/query/stream")
def stream_query(request: QueryRequest):
    """ndjson events: meta, one sentence per approved sentence, then done (or error)."""
    return StreamingResponse((json.dumps(event) + "\n" for event in answer_events(request)),
                             media_type="application/x-ndjson")


class VoiceTimings(BaseModel):
    """Speech timings the dashboard measured for a voice turn, in milliseconds."""
    stt_ms: Optional[float] = None
    first_audio_ms: Optional[float] = None
    spoken_total_ms: Optional[float] = None


@router.post("/traces/{trace_id}/voice")
def add_voice_timings(trace_id: int, timings: VoiceTimings):
    """The dashboard reports a voice turn's speech timings (spec 0006); only the API writes traces."""
    if not get_traces().add_voice(trace_id, **timings.model_dump()):
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}.")
    return {"updated": trace_id}


@router.get("/metrics")
def get_metrics(window: Literal["1h", "24h", "7d", "all"] = "24h"):
    """Counts, rates, stage and voice timings and the latest quality scores over a time window."""
    from src.observability import metrics
    return metrics.compute(get_traces(), window)


@router.get("/traces")
def list_traces(limit: int = Query(50, ge=1, le=200)):
    """The most recent finished traces, newest first (summary fields only)."""
    return get_traces().recent(limit)


@router.get("/traces/{trace_id}")
def get_trace(trace_id: int):
    """One full trace: stages, retrieved sources, dropped sentences, the answer and voice timings."""
    trace = get_traces().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}.")
    return trace


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Saves a PDF, Markdown or text file into data/raw and rebuilds the index in the background."""
    safe_name = os.path.basename(file.filename or "")
    if not safe_name or not safe_name.lower().endswith(DOCUMENT_TYPES):
        raise HTTPException(status_code=400, detail="Upload a .pdf, .md or .txt file.")
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, safe_name), "wb") as buffer:
        buffer.write(await file.read())
    background_tasks.add_task(rebuild_index)      # the response does not wait for indexing
    return {"filename": safe_name, "status": "Successfully saved. Queued for vector ingestion."}
