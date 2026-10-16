"""The browser demo's index and the Pyodide bridge (spec 0009). Runs in CPython and in Pyodide (demo/pyodide-runner);
imports only numpy, the demo code and the standard library."""
import json
import os
import sys

import numpy as np
import pytest

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


def test_the_bridge_computes_the_dashboards_metrics_over_browser_traces():
    """The same summarize the dashboard uses (src/observability/metrics.py). covers: AC-12"""
    base = {"answer": None, "confidence": None, "grounding_checker": None, "sentences_dropped": None,
            "total_ms": None, "first_sentence_ms": None, "stt_ms": None, "first_audio_ms": None,
            "spoken_total_ms": None, "prompt_tokens": None, "output_tokens": None}
    rows = [{**base, "id": 1, "path": "answered", "answer": "It is $84.", "confidence": 0.99, "total_ms": 7000,
             "first_sentence_ms": 5000, "prompt_tokens": 2000, "output_tokens": 20},
            {**base, "id": 2, "path": "gated", "confidence": 0.1, "total_ms": 2000},
            {**base, "id": 3, "path": "error"}]
    spans = [{"trace_id": 1, "layer": "rerank", "elapsed_ms": 800}, {"trace_id": 2, "layer": "rerank", "elapsed_ms": 900}]

    m = json.loads(bridge.metrics(json.dumps(rows), json.dumps(spans)))

    assert m["counts"]["asked"] == 3 and m["counts"]["answered"] == 1 and m["counts"]["errors"] == 1
    assert m["rates"]["retrieval_pass_rate"] == 0.5 and m["rates"]["refusal_rate"] == 0.5
    assert m["stages"]["rerank"] == {"p50": 800, "p95": 800, "p99": 800, "n": 1}
    assert m["tokens"]["total"] == 2020 and m["latency"]["p50"] == 7000


def test_an_upload_joins_the_index_and_clearing_uploads_keeps_the_sample():
    """covers: AC-7, AC-9, AC-10"""
    bridge.index = DemoIndex(dims=DIMS)
    sample = [chunk("energy.pdf", 0, "Brent crude oil price.")]
    bridge.load_sample(json.dumps(sample), unit(1, 0, 0, 0).tobytes())
    mine = [chunk("notes.md", 0, "The pump runs at 40 litres a minute.")]

    bridge.add_doc("notes.md", json.dumps(mine), unit(0, 1, 0, 0).tobytes())
    found = bridge.index.search_sparse("pump litres", 5, ["notes.md"])
    bridge.clear_uploads()

    assert [c["metadata"]["source"] for c in found] == ["notes.md"]
    assert bridge.index.names() == ["energy.pdf"]


def drive(grounding, check_reply):
    """Runs one question through the bridge as the page would, with fake models; returns the effects the page saw
    and the final record."""
    bridge.index = sample_index()
    bridge.configure(json.dumps({"top_k_rerank": 3, "confidence_threshold": 0.84, "checker": "nli:x"}))
    tokens = iter(["The Brent price is 84 dollars per barrel. ", "It will double next week because of aliens. "])
    item = json.loads(bridge.start(json.dumps({"query": "What is the crude oil price?", "history": []}), "null", grounding))
    seen, events = [], []
    while not item.get("end"):
        effect = item.get("effect")
        seen.append(effect or item["type"])
        if effect == "Embed":
            reply = [unit(1, 0, 0, 0).tolist() for _ in item["texts"]]
        elif effect == "Rerank":
            reply = [0.99] * len(item["texts"])
        elif effect == "NextToken":
            reply = {"token": next(tokens)} if seen.count("NextToken") <= 2 else {"done": True}
        elif effect == "CheckSentence":
            reply = check_reply(item["sentence"])
        else:
            reply = None
            if not effect:
                events.append(item)
        item = json.loads(bridge.step(json.dumps(reply)))
    return seen, events, item["record"]


def test_nli_scores_from_the_page_drop_an_unsupported_sentence():
    """covers: AC-5"""
    seen, events, record = drive(True, lambda s: {"scores": [0.1] if "aliens" in s else [0.1, 0.93]})

    done = next(e for e in events if e["type"] == "done")
    assert seen.count("CheckSentence") == 2
    assert done["grounded"] is False and done["dropped_sentences"] == ["It will double next week because of aliens."]
    assert record["grounding_checker"] == "nli:x"


def test_with_the_switch_off_the_bridge_answers_the_check_and_the_answer_is_unchecked():
    """covers: AC-5"""
    seen, events, record = drive(False, lambda s: pytest.fail("the page was asked to check a sentence"))

    done = next(e for e in events if e["type"] == "done")
    assert "CheckSentence" not in seen
    assert done["grounded"] is None and not done["dropped_sentences"]
    assert record["grounding_checker"] == "nli:x (unchecked: the grounding switch is off)"


def test_a_failed_nli_model_fails_open_with_its_reason():
    """covers: AC-5"""
    _, events, record = drive(True, lambda s: {"error": "the grounding model did not load"})

    done = next(e for e in events if e["type"] == "done")
    assert done["grounded"] is None and "aliens" in done["answer"]
    assert record["grounding_checker"] == "nli:x (unchecked: the grounding model did not load)"


def test_the_bridge_speaks_with_the_servers_cleanup_and_hears_filler_as_silence():
    """covers: AC-6"""
    pieces = json.loads(bridge.speech("The **price** is $84 [Source: energy.pdf | Page: 3]."))

    assert pieces == ["The price is $84."]
    assert json.loads(bridge.speech("[Source: energy.pdf | Page: 3]")) == []
    assert bridge.heard_nothing(" You. ") and not bridge.heard_nothing("What is the price?")
