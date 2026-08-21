import json
import os
import time
from typing import Dict, List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama

from src.utils.metrics import PipelineTracer
from src.retrieval.retriever import HybridRetriever
from src.llm.llm_client import LocalLLMClient
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunking import DocumentChunker
from src.verification.guard_rails import HallucinationGuard
from src.retrieval.query_rewriter import QueryRewriter
from src.utils.text_sanitizer import SOURCE_HEADER, has_non_latin, usable_history
from src.utils.config import get
from src.utils.query_cache import QueryCache

router = APIRouter()

RAW_DIR = "data/raw"

# Heavy models (embedder, Qdrant handle, Tantivy index, cross-encoder, spacy NER)
# are loaded once per process. Rebuilding them per request cost ~2.5s a query.
_retriever = None
_llm = None
_guard = None
_rewriter = None
_cache = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def get_llm() -> LocalLLMClient:
    global _llm
    if _llm is None:
        _llm = LocalLLMClient(generation_model='llama3.2:3b')
    return _llm

def get_guard() -> HallucinationGuard:
    global _guard
    if _guard is None:
        _guard = HallucinationGuard()
    return _guard

def get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter

def get_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache(
            # Resolved on first semantic lookup so the cache never forces the
            # retrieval stack to be constructed (tests stub it out).
            embedder_provider=lambda: get_retriever().store.db_manager.embedder,
            raw_dir=RAW_DIR,
            max_entries=get("cache.max_entries", 256),
            similarity_threshold=get("cache.similarity_threshold", 0.95),
            enabled=get("cache.enabled", True),
        )
    return _cache

def warmup():
    """Loads every model at server boot so the first query is already warm."""
    get_retriever()
    get_llm()
    get_guard()
    get_rewriter()
    get_cache()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"
    # Prior turns as [{"role": "user"|"assistant", "content": str}]. Used only to
    # resolve pronouns into a standalone search query.
    history: List[Dict[str, str]] = []
    # Language the question was asked in. Retrieval is English-only, so anything
    # else is normalised to English first.
    language: str = "english"


def build_citations(chunks: List[str]) -> List[Dict[str, str]]:
    """
    Provenance derived from the chunks that were actually used. Deterministic:
    llama3.2:3b will not place inline citations reliably, and any tag it invents
    has to be thrown away anyway (see HallucinationGuard.strip_ungrounded_citations).
    """
    seen, citations = set(), []
    for chunk in chunks:
        match = SOURCE_HEADER.match(chunk)
        if not match:
            continue
        key = (match.group(1), match.group(2))
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": match.group(1), "chunk": match.group(2)})
    return citations


def append_citations(answer: str, citations: List[Dict[str, str]]) -> str:
    """
    Guarantees the answer carries its provenance.

    llama3.2:3b emits an inline [Source: ...] tag on fewer than half of its answers
    even when the prompt demands one, so the tags are appended deterministically from
    the chunks that were actually retrieved. The speech path strips them again.
    """
    if not citations or not answer.strip():
        return answer

    already = {(c["source"], c["chunk"]) for c in citations if f"| Chunk: {c['chunk']}]" in answer}
    missing = [c for c in citations if (c["source"], c["chunk"]) not in already]
    if not missing:
        return answer

    tags = " ".join(f"[Source: {c['source']} | Chunk: {c['chunk']}]" for c in missing)
    return f"{answer.rstrip()} {tags}"


def resolve_search_query(request: QueryRequest, tracer=None) -> str:
    """Resolves follow-up pronouns, then normalises to English if needed."""
    search_query = request.query

    # Refused turns carry no subject: rewriting against them drags the next question
    # onto a dead topic, and onto the wrong language if that turn was not English.
    history = usable_history(request.history)

    if history:
        if tracer:
            tracer.start_span("rewrite")
        search_query = get_rewriter().resolve_query(search_query, history)
        if tracer:
            tracer.end_span("rewrite", f"Standalone query — {search_query!r}")

    # Normalise when the mode says to, or whenever anything non-English survived the
    # rewrite — a Hinglish turn earlier in the conversation can leak into a later
    # English question.
    if get(f"voice.languages.{request.language}.normalize_query", False) or has_non_latin(search_query):
        if tracer:
            tracer.start_span("normalize")
        search_query = get_llm().normalize_query(search_query)
        if tracer:
            tracer.end_span("normalize", f"English search query — {search_query!r}")

    return search_query

def trigger_ingestion_pipeline():
    """
    Background task to process raw documents into normalized chunks 
    and index them into the dense and sparse vector databases.
    """
    print("[Ingestion] Starting vectorization...")
    try:
        # 1. Load and normalize texts
        loader = DocumentLoader()
        documents = loader.load_documents()
        
        # 2. Chunk with token-aware windows
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)
        
        # 3. Upsert to Qdrant (Dense) and Tantivy (Sparse).
        # Reuses the process-wide manager: embedded Qdrant holds an exclusive
        # file lock, so a second VectorDBManager here would fail.
        db_manager = get_retriever().store.db_manager
        if not chunks:
            # Nothing left in data/raw. Clear both stores, or BM25 keeps serving
            # text from documents the user has deleted.
            db_manager.reset_indexes()
            get_cache().invalidate("corpus emptied")
            print("[Ingestion] Corpus is empty — indexes cleared.")
            return
        db_manager.ingest_chunks(chunks)
        get_cache().invalidate("corpus reindexed")
        print(f"[Ingestion] Successfully indexed {len(chunks)} chunks.")
    except Exception as e:
        print(f"[Ingestion] Pipeline failed: {e}")

@router.get("/health")
async def health_check():
    """Basic health check to ensure the API and Ollama are responding."""
    try:
        models_response = ollama.list()

        if hasattr(models_response, 'models'):
            available_models = [m.model for m in models_response.models]
        else:
            available_models = []
            
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
        "pipeline": "VocalRAG-Zero"
    }

@router.get("/cache")
async def cache_stats():
    """Hit rates and invalidation counts for the two-level query cache."""
    return get_cache().stats()

@router.delete("/cache")
async def clear_cache():
    get_cache().invalidate("manual")
    return get_cache().stats()

@router.get("/corpus")
async def list_corpus():
    """Lists the raw source files alongside their indexed chunk counts."""
    manager = get_retriever().store.db_manager

    if not os.path.exists(RAW_DIR):
        return []

    filenames = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".pdf", ".md", ".txt"))
    )

    rows = []
    for name in filenames:
        chunk_count = 0
        if manager.qdrant_client:
            from qdrant_client import models
            chunk_count = manager.qdrant_client.count(
                collection_name=manager.collection_name,
                count_filter=models.Filter(must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=name)
                    )
                ]),
                exact=True
            ).count

        rows.append({
            "file": name,
            "chunks": chunk_count,
            "status": "Indexed" if chunk_count else "Pending"
        })
    return rows

@router.delete("/corpus/{filename:path}")
async def delete_document(filename: str):
    """
    Removes a source file and rebuilds both indexes from whatever remains.
    Rebuilding is synchronous so the corpus listing is truthful on return.
    """
    safe_name = os.path.basename(filename)
    file_path = os.path.join(RAW_DIR, safe_name)

    # Refuse anything that resolves outside the corpus directory
    if os.path.realpath(os.path.dirname(file_path)) != os.path.realpath(RAW_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"{safe_name} is not in the corpus.")

    os.remove(file_path)
    get_cache().invalidate(f"{safe_name} removed")
    trigger_ingestion_pipeline()

    return {"deleted": safe_name}

@router.post("/query")
async def manual_query(request: QueryRequest):
    tracer = PipelineTracer()
    started = time.perf_counter()
    try:
        retriever = get_retriever()
        search_query = resolve_search_query(request, tracer)

        cached, cache_level = get_cache().get(search_query)
        if cached is not None:
            tracer.add_static_span("cache", f"{cache_level.upper()} hit — retrieval and generation skipped")
            tracer.print_trace()
            return {
                **cached,
                "cache": cache_level,
                "trace": tracer.as_list(),
                "total_latency_ms": (time.perf_counter() - started) * 1000,
            }

        retrieved_chunks, confidence_score = retriever.search_and_rerank(search_query, tracer=tracer)
        
        confidence_score = float(confidence_score)
        
        if confidence_score < retriever.confidence_threshold:
            tracer.add_static_span("confidence", f"Top chunk: {confidence_score:.3f}, threshold: 0.65 — FAIL")
            tracer.print_trace()
            payload = {
                "answer": "Insufficient context.",
                "retrieved_context": [],
                "confidence_score": confidence_score,
                "search_query": search_query,
                "citations": [],
                "grounded": None,
                "dropped_citations": [],
            }
            get_cache().put(search_query, payload)
            return {
                **payload,
                "cache": None,
                "trace": tracer.as_list(),
                "total_latency_ms": (time.perf_counter() - started) * 1000
            }
            
        tracer.add_static_span("confidence", f"Top chunk: {confidence_score:.3f}, threshold: 0.65 — PASS")

        # Trace the LLM Generation
        answer = get_llm().generate(search_query, retrieved_chunks, tracer=tracer)

        # Post-generation NER / assertion grounding check
        tracer.start_span("verification")
        guard = get_guard()
        answer, dropped_citations = guard.strip_ungrounded_citations(answer, retrieved_chunks)
        citations = build_citations(retrieved_chunks)
        answer = append_citations(answer, citations)
        grounded = guard.verify_assertions(answer, retrieved_chunks)
        tracer.end_span(
            "verification",
            f"Assertion check — {'GROUNDED' if grounded else 'UNGROUNDED'}"
            f"; {len(dropped_citations)} ungrounded citation(s) removed"
        )
        
        # Output the final trace to your terminal
        tracer.print_trace()
        
        payload = {
            "answer": answer,
            "retrieved_context": retrieved_chunks,
            "confidence_score": confidence_score,
            "search_query": search_query,
            "citations": citations,
            "grounded": grounded,
            "dropped_citations": dropped_citations,
        }
        get_cache().put(search_query, payload)

        return {
            **payload,
            "cache": None,
            "trace": tracer.as_list(),
            "total_latency_ms": (time.perf_counter() - started) * 1000
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Same retrieval and gate as /query, but emits ndjson so a caller can start
    speaking on the first finished sentence instead of waiting for the full answer.
    """
    retriever = get_retriever()
    tracer = PipelineTracer()
    search_query = resolve_search_query(request, tracer)

    cached, cache_level = get_cache().get(search_query)
    if cached is not None:
        def emit_cached():
            yield json.dumps({
                "type": "meta",
                "confidence_score": cached["confidence_score"],
                "search_query": search_query,
                "citations": cached.get("citations", []),
                "retrieved_context": cached.get("retrieved_context", []),
                "cache": cache_level,
            }) + "\n"
            yield json.dumps({"type": "token", "text": cached["answer"]}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return StreamingResponse(emit_cached(), media_type="application/x-ndjson")

    retrieved_chunks, confidence_score = retriever.search_and_rerank(search_query, tracer=tracer)
    confidence_score = float(confidence_score)
    gated = confidence_score < retriever.confidence_threshold

    def emit():
        yield json.dumps({
            "type": "meta",
            "confidence_score": confidence_score,
            "search_query": search_query,
            "citations": [] if gated else build_citations(retrieved_chunks),
            "retrieved_context": [] if gated else retrieved_chunks,
        }) + "\n"

        if gated:
            yield json.dumps({"type": "token", "text": "Insufficient context."}) + "\n"
        else:
            streamed = []
            for token in get_llm().stream_generate(search_query, retrieved_chunks):
                streamed.append(token)
                yield json.dumps({"type": "token", "text": token}) + "\n"

            answer = append_citations("".join(streamed).strip(), build_citations(retrieved_chunks))
            get_cache().put(search_query, {
                "answer": answer,
                "retrieved_context": retrieved_chunks,
                "confidence_score": confidence_score,
                "search_query": search_query,
                "citations": build_citations(retrieved_chunks),
                "grounded": None,
                "dropped_citations": [],
            })

        tracer.print_trace()
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(emit(), media_type="application/x-ndjson")

@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload PDF/Markdown files and trigger hybrid index updates asynchronously."""
    os.makedirs(RAW_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # Queue the ingestion process to avoid blocking the HTTP response
    background_tasks.add_task(trigger_ingestion_pipeline)
    
    return {
        "filename": file.filename, 
        "status": "Successfully saved. Queued for vector ingestion."
    }
