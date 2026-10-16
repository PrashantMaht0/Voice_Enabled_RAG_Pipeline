"""Builds the live demo's static sample index (spec 0009, build task 3).

Reads the documents in demo/sample/docs (listed in demo/sample/sources.yaml) with the real ingestion pipeline (the
configured table and figure readers, so layoutlmv3 and qwen2.5vl:7b), embeds every chunk with the same ONNX file
the browser loads, and writes demo/public/:
  index/manifest.json   documents, licenses, embedder id, chunk count
  index/chunks.json     every chunk (content keeps its [Source: …] header, which citations read)
  index/embeddings.f32  float32, one 384 wide row per chunk, in chunk order
  config.json           the core's settings for the browser, plus the pinned browser models
  index/suggested.json  (when demo/sample/questions.yaml exists) each suggested question answered by the local
                        server pipeline, so visitors without WebGPU still see real answers

Everything it builds on the way (Qdrant, Tantivy, caches, traces) lives in demo/sample/.build, never in vectordb/.

Usage, from the repo root (Ollama running):
  ./dev-env/bin/python demo/scripts/build_demo_index.py             # read, embed, export, then answers if questions exist
  ./dev-env/bin/python demo/scripts/build_demo_index.py --answers   # only the precomputed answers, from chunks.json
  ./dev-env/bin/python demo/scripts/build_demo_index.py --config    # only config.json (after editing config.yaml)
"""
import json
import os
import sys
from datetime import datetime, timezone

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE = os.path.join(ROOT, "demo", "sample")
DOCS = os.path.join(SAMPLE, "docs")
BUILD = os.path.join(SAMPLE, ".build")
OUT = os.path.join(ROOT, "demo", "public")
INDEX_VERSION = 1


def isolated_config() -> str:
    """config.yaml with every path pointed into demo/sample/.build (set before any src module is imported)."""
    with open(os.path.join(ROOT, "config", "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    os.makedirs(BUILD, exist_ok=True)
    config["paths"].update(raw_data_dir=DOCS, qdrant_storage=os.path.join(BUILD, "qdrant_storage"),
                           tantivy_index=os.path.join(BUILD, "tantivy_index"), collection_name="demo_sample")
    config.setdefault("observability", {})["db_path"] = os.path.join(BUILD, "traces.sqlite3")
    config.setdefault("cache", {})["enabled"] = False
    path = os.path.join(BUILD, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


os.environ["VRAG_CONFIG"] = isolated_config()
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np                                     # noqa: E402
from src.pipeline.settings import Settings              # noqa: E402
from src.utils.config import get                        # noqa: E402
from src.verification import grounding                  # noqa: E402


def sources() -> list:
    with open(os.path.join(SAMPLE, "sources.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["documents"]


def embedder_id() -> str:
    e = get("demo.embedder")
    return f"{e['id']}@{e['revision']}:{e['dtype']}"


def onnx_embed(texts: list) -> np.ndarray:
    """The browser's embedder (pinned ONNX file, CLS pooling, unit length), run with onnxruntime."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    e = get("demo.embedder")
    file = "onnx/model.onnx" if e["dtype"] == "fp32" else f"onnx/model_{e['dtype']}.onnx"
    session = ort.InferenceSession(hf_hub_download(e["id"], file, revision=e["revision"]),
                                   providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(hf_hub_download(e["id"], "tokenizer.json", revision=e["revision"]))
    tok.enable_truncation(max_length=512)
    tok.enable_padding()
    names = {i.name for i in session.get_inputs()}
    rows = []
    for start in range(0, len(texts), 32):
        enc = tok.encode_batch(texts[start:start + 32])
        feeds = {"input_ids": np.array([x.ids for x in enc], dtype=np.int64),
                 "attention_mask": np.array([x.attention_mask for x in enc], dtype=np.int64)}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.array([x.type_ids for x in enc], dtype=np.int64)
        cls = session.run(None, feeds)[0][:, 0, :]
        rows.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
    return np.vstack(rows).astype(np.float32)


def browser_config() -> dict:
    """The core's settings as the browser uses them (its own gate threshold and the NLI checker)."""
    settings = Settings.from_config(get)
    settings.confidence_threshold = float(get("demo.confidence_threshold"))
    settings.checker = get("demo.grounding_checker")
    models = {key: get(f"demo.{key}") for key in ("embedder", "reranker", "nli", "whisper", "kokoro", "answer_models")}
    models["nli"] = {**models["nli"], "max_tokens": grounding.NLI_MAX_TOKENS,          # the token window rule
                     "premise_tokens": grounding.NLI_PREMISE_TOKENS, "overlap_tokens": grounding.NLI_OVERLAP_TOKENS, "max_sentence_tokens": grounding.NLI_MAX_SENTENCE_TOKENS}
    return {**settings.as_dict(), "embedder_id": embedder_id(), "models": models,
            "space": get("demo.space"), "uploads": get("demo.uploads"), "judge_model": get("demo.judge_model")}


def build_index():
    from src.ingestion import figures, tables
    from src.ingestion.chunking import DocumentChunker
    from src.ingestion.loader import DocumentLoader

    listed = {d["name"]: d for d in sources()}
    on_disk = {f for f in os.listdir(DOCS) if f.lower().endswith(".pdf")}
    if set(listed) != on_disk:
        sys.exit(f"sources.yaml and demo/sample/docs disagree: {sorted(set(listed) ^ on_disk)}")

    print(f"[Demo] Reading {len(on_disk)} documents with the real pipeline (tables and figures)...")
    chunks = DocumentChunker().chunk_documents(DocumentLoader(raw_data_dir=DOCS).load_documents())
    # A reader that failed to load means the pipeline quietly indexed without it; the demo must never ship that.
    for reader in (tables, figures):
        if reader.effective_parser() != reader.configured_parser():
            sys.exit(f"{reader.__name__.split('.')[-1]} reader {reader.configured_parser()!r} did not load (see the "
                     "ERROR line above); nothing was written. Fix it and run again (results so far are cached).")
    unknown = {c["metadata"].get("source") for c in chunks} - set(listed)
    if unknown:
        sys.exit(f"chunks name documents not in sources.yaml: {sorted(unknown)}")

    print(f"[Demo] Embedding {len(chunks)} chunks with {embedder_id()}...")
    vectors = onnx_embed([c["content"] for c in chunks])

    os.makedirs(os.path.join(OUT, "index"), exist_ok=True)
    pages = {}
    for c in chunks:
        if c["metadata"].get("page"):
            pages[c["metadata"]["source"]] = max(pages.get(c["metadata"]["source"], 0), c["metadata"]["page"])
    manifest = {"index_version": INDEX_VERSION, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "embedder_id": embedder_id(), "chunk_count": len(chunks),
                "documents": [{"name": d["name"], "title": d["title"], "publisher": d["publisher"],
                               "pages": pages.get(d["name"]), "license": d["license"],
                               "source_url": d["source_url"]} for d in sources()]}
    records = [{"id": c["id"], "doc": c["metadata"]["source"], "doc_hash": c["metadata"].get("doc_hash"),
                "chunk_index": c["metadata"].get("chunk_index"), "page": c["metadata"].get("page"),
                "kind": c["metadata"].get("kind", "text"), "content": c["content"]} for c in chunks]
    write_json(os.path.join(OUT, "index", "manifest.json"), manifest)
    write_json(os.path.join(OUT, "index", "chunks.json"), records)
    vectors.tofile(os.path.join(OUT, "index", "embeddings.f32"))
    write_json(os.path.join(OUT, "config.json"), browser_config())
    kinds = {k: sum(r["kind"] == k for r in records) for k in ("text", "table", "figure")}
    print(f"[Demo] Wrote {len(records)} chunks {kinds} to demo/public/index/")


def build_answers():
    """Each suggested question through the local server pipeline over this sample index (Ollama answers)."""
    questions_path = os.path.join(SAMPLE, "questions.yaml")
    if not os.path.exists(questions_path):
        print("[Demo] No demo/sample/questions.yaml yet; skipping the precomputed answers.")
        return
    from src.api import routes
    from src.observability.store import TraceStore
    from src.retrieval.retriever import HybridRetriever
    from src.utils.query_cache import QueryCache

    with open(os.path.join(OUT, "index", "chunks.json"), encoding="utf-8") as f:
        records = json.load(f)
    chunks = [{"id": r["id"], "content": r["content"],
               "metadata": {"source": r["doc"], "doc_hash": r["doc_hash"], "chunk_index": r["chunk_index"],
                            "page": r["page"], "kind": r["kind"]}} for r in records]
    retriever = HybridRetriever(raw_dir=DOCS)
    retriever.store.db_manager.ingest_chunks(chunks)
    routes._retriever = retriever
    routes._cache = QueryCache(enabled=False)
    routes._traces = TraceStore(os.path.join(BUILD, "traces.sqlite3"))

    with open(questions_path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    suggested = []
    for q in questions:
        events = list(routes.answer_events(routes.QueryRequest(query=q["question"])))
        done = next(e for e in events if e["type"] in ("done", "error"))
        cited = {(c["source"], c.get("page")) for c in done.get("citations", [])}
        hit = any((q["expected_doc"], p) in cited for p in q["expected_pages"])
        print(f"[Demo] {'OK  ' if hit else 'MISS'} {q['question']} -> {done.get('path')} {sorted(cited)}")
        suggested.append({**q, "answer": done.get("answer") or done.get("detail"),
                          "citations": done.get("citations", []), "path": done.get("path"),
                          "cites_expected_page": hit})
    write_json(os.path.join(OUT, "index", "suggested.json"), suggested)


def write_json(path: str, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1 if path.endswith(("manifest.json", "config.json")) else None)


if __name__ == "__main__":
    if "--config" in sys.argv:
        write_json(os.path.join(OUT, "config.json"), browser_config())
        print("[Demo] Wrote demo/public/config.json")
    else:
        if "--answers" not in sys.argv:
            build_index()
        build_answers()
