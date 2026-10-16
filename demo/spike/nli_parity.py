"""Build task 5 (spec 0009): does the browser's NLI grounding decide like the server's?

For each suggested answer in demo/public/index/suggested.json, every sentence is paired with the windows of the
chunks it cites (should be supported), the same sentence with its numbers changed (should not), and the sentence
against windows of an unrelated document (should not). The server's CrossEncoder (cross-encoder/nli-deberta-v3-small
at the pinned revision) scores every case; the browser scores the same cases with Xenova/nli-deberta-v3-small in fp16
and fp32 and compares decisions at grounding.threshold. Output goes to demo/spike/.out/ (gitignored).

Usage: ./dev-env/bin/python demo/spike/nli_parity.py
"""
import json
import os
import re
import sys

sys.path.insert(0, ".")
from src.utils.config import get                                       # noqa: E402
from src.verification.grounding import NLIChecker, chunk_body, plain, split_sentences, windows   # noqa: E402

INDEX = "demo/public/index"
OUT = "demo/spike/.out/nli_reference.json"


def changed_numbers(sentence: str) -> str:
    """The sentence with every number moved, so the documents no longer support it."""
    return re.sub(r"\d+(\.\d+)?", lambda m: str(round(float(m.group()) * 3 + 7, 1)).rstrip("0").rstrip("."), sentence)


def main():
    chunks = json.load(open(os.path.join(INDEX, "chunks.json")))
    suggested = json.load(open(os.path.join(INDEX, "suggested.json")))
    checker = NLIChecker()
    cases = []
    for s in suggested:
        cited = [c for c in chunks if any(c["doc"] == k["source"] and c.get("page") == k["page"] for k in s["citations"])]
        other_doc = next(c["doc"] for c in chunks if c["doc"] not in {k["source"] for k in s["citations"]})
        unrelated = [c for c in chunks if c["doc"] == other_doc][:3]
        for sentence in split_sentences(re.sub(r"\s*\[Source:[^\]]*\]", "", s["answer"])):
            sentence = plain(sentence)
            if len(sentence.split()) < 3:
                continue
            variants = [("cited", sentence, cited)]
            if re.search(r"\d", sentence):
                variants.append(("numbers changed", changed_numbers(sentence), cited))
            variants.append(("unrelated document", sentence, unrelated))
            for kind, text, evidence in variants:
                wins = [w for c in evidence for w in windows(chunk_body(c["content"]))]
                cases.append({"question": s["question"], "kind": kind, "sentence": text, "windows": wins,
                              "server": checker.entailment_scores(text, [c["content"] for c in evidence])})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"threshold": float(get("grounding.threshold", 0.5)), "model": checker.model_name,
               "revision": checker.revision, "cases": cases}, open(OUT, "w"))
    threshold = float(get("grounding.threshold", 0.5))
    for kind in ("cited", "numbers changed", "unrelated document"):
        rows = [c for c in cases if c["kind"] == kind]
        kept = sum(1 for c in rows if c["server"] and max(c["server"]) >= threshold)
        print(f"[NLI] {kind}: {len(rows)} cases, server keeps {kept}")
    print(f"[NLI] wrote {OUT}")


if __name__ == "__main__":
    main()
