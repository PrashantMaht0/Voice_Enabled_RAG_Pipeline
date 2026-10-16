"""The browser demo's index and the Pyodide bridge (spec 0009). Runs in CPython and in Pyodide (demo/pyodide-runner);
imports only numpy, the demo code and the standard library."""
import json
import os
import sys

import numpy as np

from src.demo_index import DemoIndex, tokens

sys.path.insert(0, os.path.join(os.getcwd(), "demo", "py"))
import bridge                                            # noqa: E402

DIMS = 4


def chunk(doc, i, text, page=1, kind="text"):
    return {"id": f"{doc}-{i}", "doc": doc, "doc_hash": doc * 3, "chunk_index": i, "page": page, "kind": kind,
            "content": f"[Source: {doc} | Page: {page}]\n{text}"}


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def sample_index():
    index = DemoIndex(dims=DIMS)
    chunks = [chunk("energy.pdf", 0, "Brent crude oil price forecast is 84 dollars per barrel.", 3, "table"),
              chunk("energy.pdf", 1, "Natural gas storage rose in the summer.", 4),
              chunk("health.pdf", 0, "The uninsured rate was 8.0 percent in 2023.", 9, "figure")]
    vectors = np.vstack([unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)])
    index.load(chunks, vectors.tobytes())
    return index


def test_tokens_follow_tantivys_default_analyzer():
    """covers: AC-4"""
    assert tokens("Brent-crude OIL, $84/bbl!") == ["brent", "crude", "oil", "84", "bbl"]
    assert tokens("x" * 41 + " ok") == ["ok"]


def test_dense_search_ranks_by_cosine_and_keeps_the_server_candidate_shape():
    """covers: AC-4"""
    hits = sample_index().search_dense(unit(0.1, 0, 1, 0), top_k=2)

    assert [h["id"] for h in hits] == ["health.pdf-0", "energy.pdf-0"]
    assert hits[0]["metadata"] == {"source": "health.pdf", "doc_hash": "health.pdfhealth.pdfhealth.pdf",
                                   "chunk_index": 0, "page": 9, "kind": "figure"}


def test_keyword_search_scores_matching_chunks_by_bm25():
    """covers: AC-4"""
    hits = sample_index().search_sparse("crude oil price", top_k=5)

    assert [h["id"] for h in hits] == ["energy.pdf-0"]
    assert hits[0]["score"] > 0


def test_a_query_with_no_searchable_term_finds_nothing():
    assert sample_index().search_sparse("?!", top_k=5) == []


def test_the_scope_limits_both_searches_to_the_chosen_documents():
    """covers: AC-10"""
    index = sample_index()

    assert {h["metadata"]["source"] for h in index.search_dense(unit(1, 1, 1, 0), 5, ["health.pdf"])} == {"health.pdf"}
    assert index.search_sparse("crude oil", 5, ["health.pdf"]) == []


def test_removing_a_document_removes_its_chunks():
    """covers: AC-9"""
    index = sample_index()

    index.remove_doc("energy.pdf")

    assert index.names() == ["health.pdf"]


def test_the_bridge_answers_searches_itself_and_hands_other_effects_to_the_page():
    """covers: AC-4"""
    bridge.index = sample_index()
    bridge.configure(json.dumps({"top_k_rerank": 3, "confidence_threshold": 0.84, "checker": "nli:x"}))

    first = json.loads(bridge.start(json.dumps({"query": "What is the crude oil price?", "history": []}), "null", False))
    assert first["effect"] == "CacheLookup"
    retrieve = json.loads(bridge.step("null"))
    assert retrieve["effect"] == "Retrieve"
    embed = json.loads(bridge.step("null"))
    assert embed["effect"] == "Embed"
    after_searches = json.loads(bridge.step(json.dumps([unit(1, 0, 0, 0).tolist()])))
    seen = [after_searches.get("effect") or after_searches.get("type")]
    while after_searches.get("effect") == "Span":
        after_searches = json.loads(bridge.step("null"))
        seen.append(after_searches.get("effect") or after_searches.get("type"))

    assert "SearchDense" not in seen and "SearchSparse" not in seen
    assert after_searches["effect"] == "Rerank"
    bridge.stop()


def test_an_oversized_question_returns_its_reason():
    """covers: AC-4"""
    bridge.configure(json.dumps({"max_query_chars": 10}))

    reply = json.loads(bridge.start(json.dumps({"query": "x" * 11, "history": []}), "null", False))

    assert reply == {"type": "error", "detail": "the question is longer than 10 characters", "end": True}
