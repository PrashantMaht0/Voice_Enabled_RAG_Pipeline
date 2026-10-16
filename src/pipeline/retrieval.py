"""Retrieval rules: reciprocal rank fusion, the deleted source filter, the reranker's text, and the confidence gate."""
import re
from typing import Dict, Iterable, List, Optional, Tuple

# A Markdown table separator row, e.g. "|---|:---:|".
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def fuse(dense_results: List[Dict], sparse_results: List[Dict], alpha: float, top_k: int = 30) -> List[Dict]:
    """Both candidate lists merged by reciprocal rank fusion, best top_k first.

    Each list adds weight / (rank + 60) per chunk; `alpha` is the dense weight and the rest goes to keywords. A
    chunk found by both keeps the dense copy (it has the full metadata)."""
    if not dense_results and not sparse_results:
        return []                     # an empty index has nothing to answer from
    rrf_scores, pool = {}, {}

    def apply_rrf(results, weight):
        for rank, res in enumerate(results):
            pool.setdefault(res["id"], res)
            rrf_scores[res["id"]] = rrf_scores.get(res["id"], 0.0) + weight * (1.0 / (rank + 60))

    apply_rrf(dense_results, alpha)
    apply_rrf(sparse_results, 1.0 - alpha)
    ranked = sorted(rrf_scores, key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
    return [pool[chunk_id] for chunk_id in ranked[:top_k]]


def present(candidates: List[Dict], docs: Optional[Iterable[str]]) -> List[Dict]:
    """Drops chunks whose document is not in `docs`, so a deleted document stops being cited before the rebuild
    finishes (spec 0008, AC-8). `docs` None means there is nothing to compare with."""
    if docs is None:
        return candidates
    docs = set(docs)
    return [c for c in candidates if (c.get("metadata") or {}).get("source") in docs]


def rerank_text(text: str) -> str:
    """What the cross-encoder reads: Markdown table pipes become spaces and separator rows go.

    ms-marco models score a pipe grid as noise, so correct table chunks fell under the
    confidence gate (spec 0004). The chunk itself, which the LLM reads, stays Markdown.
    """
    lines = []
    for line in text.splitlines():
        if TABLE_RULE.match(line):
            continue
        if line.lstrip().startswith("|"):
            line = " ".join(cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|")))
        lines.append(line)
    return "\n".join(lines)


def gate(ranked: List[Dict], threshold: float, top_k: int) -> Tuple[List[str], float, List[Dict]]:
    """The gate over reranked `{chunk, score, metadata}` (best first): the top top_k chunks that score at least the
    threshold, the top score, and each kept chunk's id, source, page, kind and score (spec 0006)."""
    valid_chunks, details = [], []
    highest_score = 0.0
    for i, res in enumerate(ranked[:top_k]):
        score = float(res["score"])           # a numpy float does not serialize to JSON
        if i == 0:
            highest_score = score
        if score >= threshold:
            valid_chunks.append(res["chunk"])
            meta = res.get("metadata") or {}
            doc_hash, index = meta.get("doc_hash", ""), meta.get("chunk_index")
            details.append({"chunk_id": f"{doc_hash[:16]}_{index}" if index is not None else None,
                            "source": meta.get("source"), "page": meta.get("page"),
                            "kind": meta.get("kind", "text"), "score": round(score, 4)})
    return valid_chunks, highest_score, details
