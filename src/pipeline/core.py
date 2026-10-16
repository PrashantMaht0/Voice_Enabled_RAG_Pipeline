"""The one answer pipeline as steps (spec 0009): rewrite, the empty question rule, the cache, retrieval and the
confidence gate, streamed generation with a sentence by sentence grounding check, and citations.

`answer` is a plain generator. It yields two kinds of item:
- an effect, `{"effect": name, ...}`: work it cannot do itself; the driver does it and sends the result back;
- an event, `{"type": "meta" | "sentence" | "done", ...}`: something to show; the driver sends None back.
Events leave out `trace_id` and `trace`, which each driver fills in. Unexpected errors are not caught here: the
driver turns them into its error reply. `record` is kept up to date so the driver can write the trace even after
an error or a disconnect.

The server answers `Retrieve`, `Rewrite` and `Generate` with its own objects (which call the same rules); the
browser answers `Retrieve` with None, and the core then runs the retrieval steps itself."""
import time
from typing import Dict, Iterable, List, Optional

from src.pipeline.limits import check_history, check_query
from src.pipeline.prompts import answer_messages, clean_rewrite, refers_back, rewrite_messages
from src.pipeline.retrieval import fuse, gate, present, rerank_text
from src.pipeline.settings import Settings
from src.pipeline.sources import append_citations, build_citations, kept_citations, log_citations, with_kinds
from src.utils.text_sanitizer import is_refusal, usable_history
from src.verification import grounding
from src.verification.citations import strip_ungrounded_citations

INSUFFICIENT = "Insufficient context."


def span(layer: str, details: str, ms: float = 0.0) -> Dict:
    """A trace step whose time the core measured itself."""
    return {"effect": "Span", "layer": layer, "details": details, "ms": ms}


def timed(effect: Dict):
    """Yields one effect and returns (its result, milliseconds until the result came back)."""
    started = time.perf_counter()
    result = yield effect
    return result, (time.perf_counter() - started) * 1000


def retrieve_steps(query: str, settings: Settings, docs: Optional[Iterable[str]]):
    """Retrieval done by the core (the browser): embed, keyword and dense search, fusion, the deleted source
    filter, reranking and the gate. Returns (chunks, top score, details)."""
    docs = None if docs is None else list(docs)
    vectors, ms = yield from timed({"effect": "Embed", "texts": [query]})
    yield span("embed", "Question embedded", ms)
    sparse, ms = yield from timed({"effect": "SearchSparse", "query": query, "top_k": settings.top_k_retrieval,
                                   "docs": docs})
    yield span("bm25", f"Keyword search — {len(sparse)} candidates", ms)
    dense, ms = yield from timed({"effect": "SearchDense", "vector": vectors[0], "top_k": settings.top_k_retrieval,
                                  "docs": docs})
    yield span("vector", f"ANN search — {len(dense)} candidates", ms)

    started = time.perf_counter()
    candidates = present(fuse(dense, sparse, settings.hybrid_alpha, settings.top_k_rerank_pool), docs)
    yield span("fusion", f"Hybrid merge — {len(candidates)} unique candidates", (time.perf_counter() - started) * 1000)
    if not candidates:
        return [], 0.0, []

    scores, ms = yield from timed({"effect": "Rerank", "query": query,
                                   "texts": [rerank_text(c.get("content", "")) for c in candidates]})
    yield span("rerank", f"Cross-encoder scored {len(candidates)} chunks", ms)
    ranked = sorted(({"chunk": c.get("content", ""), "score": float(s), "metadata": c.get("metadata") or {}}
                     for c, s in zip(candidates, scores)), key=lambda r: r["score"], reverse=True)
    return gate(ranked, settings.confidence_threshold, settings.top_k_rerank)


def answer(request: Dict, settings: Settings, record: Dict, docs: Optional[Iterable[str]] = None):
    """One question, `{query, history, channel}`, as effects and events (see the module docstring). `docs` limits
    the documents searched when the driver searches through the core (the browser's scope selector)."""
    started = time.perf_counter()

    def elapsed():
        """Milliseconds since the question arrived."""
        return (time.perf_counter() - started) * 1000

    def sentence(index: int, text: str) -> Dict:
        """A sentence event (citation tags removed for display), noting when the first one went out."""
        if record.get("first_sentence_ms") is None:
            record["first_sentence_ms"] = elapsed()
        return {"type": "sentence", "index": index, "text": grounding.plain(text) or text}

    query = check_query(request["query"], settings.max_query_chars)
    history = usable_history(check_history(request.get("history") or [], settings.max_history_turns,
                                           settings.max_turn_chars))
    # Refused turns carry no subject and would drag the next question off topic (usable_history drops them).
    # A question that already stands alone skips the rewrite model, which would turn it into the earlier one.
    search_query = query
    if history and refers_back(query):
        reply, ms = yield from timed({"effect": "Rewrite", "query": query, "history": history,
                                      "messages": rewrite_messages(query, history)})
        search_query = clean_rewrite(reply, query)
        yield span("rewrite", f"Standalone query — {search_query!r}", ms)
    record["search_query"] = search_query

    if grounding.is_empty_query(search_query):
        # Nothing to search for: no cache, retrieval or answer model (spec 0006, AC-5)
        text = grounding.EMPTY_QUERY_ANSWER
        yield span("empty_query", "No content words — refused before retrieval")
        record.update(path="empty_query", answer=text)
        yield {"type": "meta", "confidence_score": None, "search_query": search_query, "citations": [],
               "retrieved_context": [], "cache": None}
        yield sentence(0, text)
        yield {"type": "done", "answer": text, "citations": [], "grounded": None, "path": "empty_query",
               "dropped_sentences": [], "dropped_citations": [], "cache": None, "trace": None,
               "total_ms": elapsed()}
        return

    hit = yield {"effect": "CacheLookup", "search_query": search_query}
    if hit:
        cached, level = hit["payload"], hit["level"]
        yield span("cache", f"{level.upper()} hit — retrieval and generation skipped")
        log_citations(cached.get("citations", []))
        record.update(path="cache", cache_level=level, confidence=cached.get("confidence_score"),
                      answer=cached["answer"], dropped=cached.get("dropped_sentences", []),
                      retrieved=cached.get("retrieved"), grounding_checker=cached.get("grounding_checker"))
        yield {"type": "meta", "confidence_score": cached.get("confidence_score"), "search_query": search_query,
               "citations": cached.get("citations", []), "retrieved_context": cached.get("retrieved_context", []),
               "retrieved": cached.get("retrieved") or [], "cache": level}
        for i, text in enumerate(cached.get("sentences") or grounding.split_sentences(cached["answer"])):
            yield sentence(i, text)
        yield {"type": "done", **{k: cached.get(k) for k in ("answer", "citations", "grounded", "path")},
               "dropped_sentences": cached.get("dropped_sentences", []),
               "dropped_citations": cached.get("dropped_citations", []), "cache": level, "trace": None,
               "total_ms": elapsed()}
        return

    found = yield {"effect": "Retrieve", "search_query": search_query, "docs": None if docs is None else list(docs)}
    if found is None:
        chunks, confidence, details = yield from retrieve_steps(search_query, settings, docs)
        generation, threshold = None, settings.confidence_threshold
    else:
        chunks, confidence, details = found["chunks"], found["score"], found["details"]
        generation, threshold = found.get("generation"), found.get("threshold", settings.confidence_threshold)
    confidence = float(confidence)
    record.update(confidence=confidence, retrieved=details)
    gated = confidence < threshold
    yield span("confidence", f"Top chunk: {confidence:.3f}, threshold: {threshold} — {'FAIL' if gated else 'PASS'}")
    yield {"type": "meta", "confidence_score": confidence, "search_query": search_query,
           "citations": [] if gated else with_kinds(build_citations(chunks), details),
           "retrieved_context": [] if gated else chunks, "retrieved": [] if gated else details, "cache": None}

    if gated:
        text, citations, path, result = INSUFFICIENT, [], "gated", None
        sentences = [text]
        yield sentence(0, text)
    else:
        gen_started = time.perf_counter()
        opened = yield {"effect": "Generate", "search_query": search_query, "chunks": chunks,
                        "messages": answer_messages(search_query, chunks, settings.answer_style)}
        judge = grounding.JudgeSteps(chunks, (opened or {}).get("checker", settings.checker),
                                     settings.grounding_threshold)
        buffer, index, usage = grounding.SentenceBuffer(), 0, {}
        while True:
            item = yield {"effect": "NextToken"}
            if item.get("done"):
                usage = item
                break
            for piece in buffer.feed(item["token"]):
                for approved in (yield from judge.judge_steps(piece)):
                    yield sentence(index, approved)
                    index += 1
        for piece in buffer.finish():
            for approved in (yield from judge.judge_steps(piece)):
                yield sentence(index, approved)
                index += 1
        for approved in judge.finish():
            yield sentence(index, approved)
            index += 1
        result = judge.result
        gen_ms = (time.perf_counter() - gen_started - result.seconds) * 1000
        yield span("generation", f"LLM call — {usage.get('output_tokens') or 0} tokens generated", gen_ms)
        yield span("grounding", f"{result.checker or 'unchecked'}: {result.checked} checked, "
                                f"{len(result.dropped)} dropped"
                                + (f", {result.invalid} invalid replies kept" if result.invalid else "")
                                + (f" — unchecked: {result.unchecked_reason}" if result.unchecked_reason else ""),
                   result.seconds * 1000)
        if judge.refused:
            text, citations, path, sentences = INSUFFICIENT, [], "refused_grounding", [INSUFFICIENT]
            yield sentence(index, text)
        else:
            sentences = result.kept
            text, dropped_citations = strip_ungrounded_citations(grounding.join_sentences(result.kept), chunks)
            record["dropped_citations"] = dropped_citations
            citations = with_kinds(kept_citations(chunks, result), details)
            text = append_citations(text, citations)
            path = "refused_model" if result.model_refused and all(is_refusal(k) for k in result.kept) \
                else "answered"

    log_citations(citations)
    grounded = result.grounded if result else None
    dropped = result.dropped if result else []
    checker = None
    if result and result.checker:
        checker = result.checker + (f" (unchecked: {result.unchecked_reason})" if result.unchecked_reason else "")
    payload = {"answer": text, "retrieved_context": [] if path == "gated" else chunks,
               "confidence_score": confidence, "search_query": search_query, "citations": citations,
               "grounded": grounded, "dropped_citations": record.get("dropped_citations", []),
               "dropped_sentences": dropped, "sentences": sentences, "path": path,
               "retrieved": details, "grounding_checker": checker, "generation": generation}
    yield {"effect": "CacheStore", "search_query": search_query, "payload": payload}
    record.update(path=path, answer=text, dropped=dropped, grounding_checker=checker,
                  kept=len(sentences) if result else None)
    yield {"type": "done", "answer": text, "citations": citations, "grounded": grounded, "path": path,
           "dropped_sentences": dropped, "dropped_citations": payload["dropped_citations"], "cache": None,
           "trace": None, "total_ms": elapsed()}
