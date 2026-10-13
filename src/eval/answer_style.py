"""Answer style runner (spec 0006, AC-21): should answers open with a short first sentence?

    python -m src.eval.answer_style run [--apply] [--skip-ragas]

Runs today's setup with llm.answer_style full (S0) and then short_first (S1), back to back on the same code,
through the sweep plumbing, unchecked (grounding.checker: none): the table and figure sets, the latency benchmark
(golden gates, first audio, gaps), and Ragas for both. short_first is adopted only when it passes every gate, starts
speaking at least 500 ms sooner, keeps the largest gap p95 at or under 300 ms, keeps answer relevancy within 0.05
and never copies the prompt's example. The style does not change the index, so both reuse one kept index.
"""
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

from src.eval import context_size as cs
from src.eval import system_sweep as ss
from src.eval import table_benchmark as tb
from src.eval.table_benchmark import BenchmarkError, REPO_ROOT
from src.prompts.prompt_template import EXAMPLE_LEAK_PHRASES
from src.verification import grounding

STYLES = {"S0": "full", "S1": "short_first"}
MIN_FIRST_AUDIO_GAIN_MS = 500
MAX_GAP_P95_MS = 300
MAX_RELEVANCY_DROP = 0.05


def percentile(values: list, q: float):
    """The value at quantile q (lower of the two neighbours); None for no values."""
    values = sorted(values)
    return values[int(q * (len(values) - 1))] if values else None


def answers(record: dict) -> list:
    """The golden answers in the record's Ragas CSV (empty when Ragas did not run)."""
    rag = record.get("ragas")
    if not isinstance(rag, dict) or not rag.get("file"):
        return []
    csv.field_size_limit(10 ** 9)
    with open(REPO_ROOT / rag["file"], encoding="utf-8") as f:
        return [row.get("response") or "" for row in csv.DictReader(f)]


def leaks(texts: list) -> int:
    """Answers that copied the prompt's made up example."""
    return sum(any(p in t.lower() for p in EXAMPLE_LEAK_PHRASES) for t in texts)


def shape(texts: list) -> dict:
    """Report only: first sentence words and sentences per answer (citation tags removed)."""
    firsts, counts = [], []
    for text in texts:
        sentences = grounding.split_sentences(text)
        if sentences:
            firsts.append(grounding.word_count(grounding.plain(sentences[0])))
            counts.append(len(sentences))
    n = len(firsts)
    return {"answers": n, "first_words_p50": percentile(firsts, 0.5), "first_words_p90": percentile(firsts, 0.9),
            "share_at_most_10_words": round(sum(w <= 10 for w in firsts) / n, 3) if n else None,
            "share_under_3_words": round(sum(w < 3 for w in firsts) / n, 3) if n else None,
            "sentences_p50": percentile(counts, 0.5)}


def gates(record: dict, base: dict) -> dict:
    """The context size gates plus this runner's own: first audio at least 500 ms sooner, gap p95 at most 300 ms,
    relevancy within 0.05, no copied example (spec 0006, AC-21)."""
    g = cs.gates(record, base)
    a, b = base["latency"].get("first_audio_p50_ms"), record["latency"].get("first_audio_p50_ms")
    gap = record["latency"].get("max_gap_p95_ms")
    rel_a = (base.get("ragas") or {}).get("answer_relevancy") if isinstance(base.get("ragas"), dict) else None
    rel_b = (record.get("ragas") or {}).get("answer_relevancy") if isinstance(record.get("ragas"), dict) else None
    g["first_audio_500ms_sooner"] = a is not None and b is not None and b <= a - MIN_FIRST_AUDIO_GAIN_MS
    g["gap_p95_300ms"] = gap is not None and gap <= MAX_GAP_P95_MS
    g["relevancy_within_005"] = rel_a is not None and rel_b is not None and rel_b >= rel_a - MAX_RELEVANCY_DROP
    g["no_example_leaks"] = record.get("example_leaks") == 0
    return g


def decide(records: dict) -> tuple:
    """(chosen style, reason): short_first only if it passes every gate against full."""
    full, short = records.get("S0"), records.get("S1")
    if not full or full.get("status") != "ran":
        return "full", "the full style run did not finish, so nothing changes"
    if not short or short.get("status") != "ran":
        return "full", "the short_first run did not finish"
    failed = [k for k, ok in short.get("gates", {}).items() if not ok]
    if failed:
        return "full", f"short_first fails: {', '.join(failed)}"
    a, b = full["latency"]["first_audio_p50_ms"], short["latency"]["first_audio_p50_ms"]
    return "short_first", f"short_first passes every gate and lowers first audio p50 from {a:.0f} ms to {b:.0f} ms"


def apply(style: str):
    """Writes the decided `llm.answer_style` into config.yaml, keeping its comments."""
    text = ss.CONFIG_PATH.read_text(encoding="utf-8")
    text, n = re.subn(r'^(\s+answer_style:\s*)"?[a-z_]+"?', rf'\g<1>"{style}"', text, count=1, flags=re.M)
    if n != 1:
        raise BenchmarkError(f"Could not find llm.answer_style in {ss.CONFIG_PATH}")
    ss.CONFIG_PATH.write_text(text, encoding="utf-8")


def run(skip_ragas: bool = False) -> tuple:
    """Runs full (S0) then short_first (S1) unchecked, with Ragas unless skipped, and decides; returns (result, run
    file path)."""
    sets = ss.load_sets()
    busy = ss.ollama_ps()
    if busy:
        raise BenchmarkError(f"Ollama already has models loaded ({', '.join(m['model'] for m in busy)}); "
                             "unload them, then run again.")
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    path = ss.OUT_DIR / f"style-{stamp}.json"
    work = ss.OUT_DIR / f"style-{stamp}-work"
    corpus = tb.corpus_dir(list(sets), work / "corpus")
    base = {**cs.current_settings(), "top_k_rerank": current_top_k()}
    source = cs.reusable_index(base)
    result = {**ss.new_run(stamp), "settings": {**base, "grounding_checker": "none"},
              "index_copied_from": str(source.relative_to(REPO_ROOT)) if source else None}
    for cid, style in STYLES.items():
        print(f"[style] {cid}: answer_style {style}…")
        record = ss.run_config(cid, {**base, "answer_style": style}, sets, corpus, work / cid, copy_from=source)
        if record["status"] == "ran":
            if not skip_ragas:
                print(f"[style] {cid}: Ragas…")
                record["ragas"] = ss.ragas(cid, record, f"style-{stamp}")
            texts = answers(record)
            record["example_leaks"] = leaks(texts) if texts else None
            record["shape"] = shape(texts)
        result["configs"][cid] = record
        ss.save_run(result, path)
    if result["configs"].get("S0", {}).get("status") == "ran" and result["configs"].get("S1", {}).get("status") == "ran":
        result["configs"]["S1"]["gates"] = gates(result["configs"]["S1"], result["configs"]["S0"])
    result["chosen_style"], result["reason"] = decide(result["configs"])
    chosen = result["configs"].get("S1" if result["chosen_style"] == "short_first" else "S0", {})
    result["chosen_ragas_file"] = chosen["ragas"].get("file") if isinstance(chosen.get("ragas"), dict) else None
    result["finished"] = datetime.now().isoformat(timespec="seconds")
    ss.save_run(result, path)
    return result, path


def current_top_k() -> int:
    """The configured `retrieval.top_k_rerank`, kept the same for both styles."""
    import yaml
    with open(ss.CONFIG_PATH, encoding="utf-8") as f:
        return int(yaml.safe_load(f)["retrieval"]["top_k_rerank"])


def print_report(result: dict, path: Path):
    """Prints both records, their gates and the decision."""
    print("\nAnswer style · full vs short_first (unchecked)\n")
    for cid, r in result["configs"].items():
        if r.get("status") != "ran":
            print(f"{cid}: {r.get('status')} ({r.get('reason')})")
            continue
        g, lat, rag = r["golden"], r["latency"], r.get("ragas")
        faith, rel = (rag.get("faithfulness"), rag.get("answer_relevancy")) if isinstance(rag, dict) else (rag, None)
        print(f"{cid} {STYLES[cid]}: tables {r['tables']['right_value_and_page']} · figures "
              f"{r['figures']['right_value_and_page']} · answered {g['answered']}/{g['in_scope']} · refused "
              f"{g['refused']}/{g['out_of_scope']} · faithfulness {faith} · relevancy {rel} · typed p50 "
              f"{lat.get('typed_p50_ms') or 0:.0f} ms · first audio p50 {lat.get('first_audio_p50_ms') or 0:.0f} ms · "
              f"gap p95 {lat.get('max_gap_p95_ms') or 0:.0f} ms · leaks {r.get('example_leaks')}")
        print(f"   shape {r.get('shape')}")
        if r.get("gates"):
            print(f"   gates {r['gates']}")
    print(f"\nChosen: {result['chosen_style']} ({result['reason']})")
    print(f"Saved {path.relative_to(REPO_ROOT)}")


def main(argv=None) -> int:
    """Command line entry: `python -m src.eval.answer_style run [--apply] [--skip-ragas]`."""
    parser = argparse.ArgumentParser(description="Answer style runner (spec 0006)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--apply", action="store_true", help="write the decided llm.answer_style to config.yaml")
    run_parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args(argv)
    try:
        result, path = run(args.skip_ragas)
        print_report(result, path)
        if args.apply:
            apply(result["chosen_style"])
            print(f"[style] Applied llm.answer_style {result['chosen_style']} to config/config.yaml.")
    except BenchmarkError as e:
        print(f"[answer_style] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
