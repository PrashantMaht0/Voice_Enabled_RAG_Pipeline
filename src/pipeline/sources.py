"""Citations: which sources an answer cites, read from the retrieved chunks and the model's tags."""
from typing import Dict, List

from src.utils.text_sanitizer import CITATION_TAG, SOURCE_HEADER, parse_tag


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


def with_kinds(citations: List[Dict], details: List[Dict]) -> List[Dict]:
    """Each citation gains the kinds (text, table, figure) of the retrieved chunks behind it."""
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
