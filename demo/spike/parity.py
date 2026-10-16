"""Spike (spec 0009, build task 1): does the browser's ONNX embedder and reranker agree with the server's?

Reads a COPY of the local Qdrant index (never the live folder), embeds every chunk and every golden question
with the ONNX files the browser will load, and compares:
  - dense top 5 per question: sentence-transformers (the stored vectors) versus ONNX fp32 and int8
  - the confidence gate: flashrank's score versus the ONNX cross encoder's sigmoid, on the same candidates

Usage: ./dev-env/bin/python demo/spike/parity.py <copy of qdrant_storage> <collection>
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from src.embeddings.embedder import DenseEmbedder          # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker, rerank_text   # noqa: E402

EMBEDDER, EMBEDDER_REV = "Xenova/bge-small-en-v1.5", "ea104dacec62"
RERANKER, RERANKER_REV = "Xenova/ms-marco-MiniLM-L-12-v2", "42a4a787e304"
GATE, TOP_K, POOL = 0.65, 5, 12


def onnx_session(repo, rev, file):
    return ort.InferenceSession(hf_hub_download(repo, file, revision=rev), providers=["CPUExecutionProvider"])


def tokenizer(repo, rev, max_len):
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding()
    return tok


def feed(session, enc):
    names = {i.name for i in session.get_inputs()}
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
    if "token_type_ids" in names:
        feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    return feeds


def onnx_embed(session, tok, texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[i:i + batch])
        hidden = session.run(None, feed(session, enc))[0]
        cls = hidden[:, 0, :]
        out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
    return np.vstack(out)


def onnx_rerank(session, tok, query, texts):
    enc = tok.encode_batch([(query, t) for t in texts])
    logits = session.run(None, feed(session, enc))[0].reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits))


def main(path, collection):
    client = QdrantClient(path=path)
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection, limit=256, offset=offset, with_vectors=True, with_payload=True)
        points.extend(batch)
        if offset is None:
            break
    contents = [p.payload["content"] for p in points]
    stored = np.array([p.vector for p in points], dtype=np.float32)
    golden = json.load(open("data/eval/golden_dataset.json"))
    questions = [q["question"] for q in golden]
    in_scope = [bool(q.get("in_scope")) for q in golden]
    print(f"{len(points)} chunks, {len(questions)} golden questions ({sum(in_scope)} in scope)")

    server = DenseEmbedder()
    server_q = np.array(server.embed_batch(questions), dtype=np.float32)
    server_top = np.argsort(-(server_q @ stored.T), axis=1)

    etok = tokenizer(EMBEDDER, EMBEDDER_REV, 512)
    results = {}
    for dtype, file in (("fp32", "onnx/model.onnx"), ("int8", "onnx/model_quantized.onnx")):
        sess = onnx_session(EMBEDDER, EMBEDDER_REV, file)
        chunks_v = onnx_embed(sess, etok, contents)
        q_v = onnx_embed(sess, etok, questions)
        top = np.argsort(-(q_v @ chunks_v.T), axis=1)
        overlaps = [len(set(server_top[i, :TOP_K]) & set(top[i, :TOP_K])) for i in range(len(questions))]
        cos = float(np.mean(np.sum(chunks_v * stored, axis=1)))
        results[dtype] = {"mean_top5_overlap": round(float(np.mean(overlaps)), 2),
                          "questions_with_5_of_5": sum(o == 5 for o in overlaps),
                          "questions_with_4_plus": sum(o >= 4 for o in overlaps),
                          "mean_cosine_to_server_vectors": round(cos, 4)}
        print("embedder", dtype, results[dtype])

    flash = CrossEncoderReranker()
    folder = f"{flash.cache_dir}/ms-marco-MiniLM-L-12-v2"
    variants = {
        "flashrank_file": (ort.InferenceSession(f"{folder}/flashrank-MiniLM-L-12-v2_Q.onnx",
                                                providers=["CPUExecutionProvider"]),
                           Tokenizer.from_file(f"{folder}/tokenizer.json")),
        "xenova_fp32": (onnx_session(RERANKER, RERANKER_REV, "onnx/model.onnx"), tokenizer(RERANKER, RERANKER_REV, 512)),
    }
    variants["flashrank_file"][1].enable_truncation(max_length=512)
    variants["flashrank_file"][1].enable_padding()
    reference = []
    for key, (sess, rtok) in variants.items():
        agree, diffs, flash_pass, onnx_pass, refused_agree = 0, [], 0, 0, 0
        for i, q in enumerate(questions):
            cands = [{"content": contents[j], "metadata": {}} for j in server_top[i, :POOL]]
            f_scores = {r["chunk"]: r["score"] for r in flash.rerank(q, cands)}
            texts = [rerank_text(c["content"]) for c in cands]
            o_scores = onnx_rerank(sess, rtok, q, texts)
            f = np.array([f_scores[c["content"]] for c in cands])
            diffs.append(float(np.max(np.abs(f - o_scores))))
            fp, op = bool(f.max() >= GATE), bool(o_scores.max() >= GATE)
            flash_pass += fp
            onnx_pass += op
            agree += fp == op
            refused_agree += (fp == op) and not in_scope[i]
            if key == "flashrank_file":
                reference.append({"query": q, "passages": texts, "flashrank_scores": [float(x) for x in f]})
        results["reranker_" + key] = {"gate_agreement": f"{agree}/{len(questions)}",
                                      "trick_question_agreement": f"{refused_agree}/{in_scope.count(False)}",
                                      "flashrank_passes": flash_pass, "onnx_passes": onnx_pass,
                                      "max_abs_score_diff": round(max(diffs), 6)}
        print("reranker", key, results["reranker_" + key])
    os.makedirs("demo/spike/.out", exist_ok=True)
    json.dump(reference, open("demo/spike/.out/rerank_reference.json", "w"))
    json.dump(results, open("demo/spike/parity_results.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
