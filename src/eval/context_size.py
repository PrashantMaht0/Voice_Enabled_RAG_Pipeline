"""Context size runner (spec 0006, AC-10): should the answer model read 3 passages instead of 5?

    python -m src.eval.context_size run [--apply] [--skip-ragas]

Runs today's setup (the four model values in config/config.yaml) with retrieval.top_k_rerank 5 and 3 through
the sweep plumbing, unchecked (grounding.checker: none): the table and figure sets, the latency benchmark
(golden gates, first audio), and Ragas for both. 3 is adopted only when it passes every gate and lowers first
audio p50. top_k_rerank does not change the index, so both settings reuse one kept index when its settings match.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.eval import system_sweep as ss
from src.eval import table_benchmark as tb
from src.eval.table_benchmark import BenchmarkError, REPO_ROOT

SIZES = (5, 3)
SLACK = 2                         # tables and figures may each fall at most this much below 5 chunks
MIN_ANSWERED = 0.90
MIN_FAITHFULNESS = 0.85


def current_settings() -> dict:
    """The four model values the run keeps fixed: table reader, figure reader, answer model, rewriter."""
    with open(ss.CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {"table_parser": config["ingestion"]["table_parser"], "figure_parser": config["ingestion"]["figure_parser"],
            "generation_model": config["llm"]["generation_model"], "rewriter_model": config["llm"]["rewriter_model"]}


def reusable_index(settings: dict):
    """The newest sweep work folder whose index was built with these readers (the index ignores top_k)."""
    for path in sorted(ss.OUT_DIR.glob("sweep-*.json"), reverse=True):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for record in (run.get("configs") or {}).values():
            s = record.get("settings") or {}
            folder = REPO_ROOT / record.get("workdir", "")
            if (record.get("status") == "ran" and not record.get("reused_from")
                    and (s.get("table_parser"), s.get("figure_parser")) ==
                    (settings["table_parser"], settings["figure_parser"]) and (folder / "index").is_dir()):
                return folder
    return None


def gates(record: dict, base: dict) -> dict:
    """The gates a setting must pass against the 5 chunk run: trick questions refused, 90% answered, faithfulness,
    tables and figures within 2 (spec 0006, AC-10)."""
    g = record["golden"]
    ragas = record.get("ragas")
    return {
        "all_trick_questions_refused": g["refused"] == g["out_of_scope"],
        "answered_90_percent": g["in_scope"] > 0 and g["answered"] / g["in_scope"] >= MIN_ANSWERED,
        "faithfulness_085": isinstance(ragas, dict) and (ragas.get("faithfulness") or 0) >= MIN_FAITHFULNESS,
        "tables_within_2": record["tables"]["right_value_and_page"] >= base["tables"]["right_value_and_page"] - SLACK,
        "figures_within_2": record["figures"]["right_value_and_page"] >= base["figures"]["right_value_and_page"] - SLACK,
    }


def decide(records: dict) -> tuple:
    """(chosen top_k, reason): 3 only if it passes every gate and lowers first audio p50."""
    five, three = records.get("K5"), records.get("K3")
    if not five or five.get("status") != "ran":
        return 5, "the 5 chunk run did not finish, so nothing changes"
    if not three or three.get("status") != "ran":
        return 5, "the 3 chunk run did not finish"
    failed = [k for k, ok in three["gates"].items() if not ok]
    if failed:
        return 5, f"3 chunks fail: {', '.join(failed)}"
    a, b = five["latency"].get("first_audio_p50_ms"), three["latency"].get("first_audio_p50_ms")
    if a is None or b is None:
        return 5, "first audio was not measured (voice skipped), so the speed gain is unknown"
    if b >= a:
        return 5, f"3 chunks pass every gate but first audio p50 is not lower ({b:.0f} ms vs {a:.0f} ms)"
    return 3, f"3 chunks pass every gate and lower first audio p50 from {a:.0f} ms to {b:.0f} ms"


def apply(top_k: int):
    """Writes the chosen `retrieval.top_k_rerank` into config.yaml, keeping its comments."""
    text = ss.CONFIG_PATH.read_text(encoding="utf-8")
    text, n = re.subn(r"^(\s+top_k_rerank:\s*)\d+", rf"\g<1>{top_k}", text, count=1, flags=re.M)
    if n != 1:
        raise BenchmarkError(f"Could not find retrieval.top_k_rerank in {ss.CONFIG_PATH}")
    ss.CONFIG_PATH.write_text(text, encoding="utf-8")


def run(skip_ragas: bool = False) -> tuple:
    """Runs 5 and then 3 chunks unchecked, with Ragas unless skipped, and decides; returns (result, run file path)."""
    sets = ss.load_sets()
    busy = ss.ollama_ps()
    if busy:
        raise BenchmarkError(f"Ollama already has models loaded ({', '.join(m['model'] for m in busy)}); "
                             "unload them, then run again.")
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    path = ss.OUT_DIR / f"context-{stamp}.json"
    work = ss.OUT_DIR / f"context-{stamp}-work"
    corpus = tb.corpus_dir(list(sets), work / "corpus")
    base = current_settings()
    source = reusable_index(base)
    result = {"started": datetime.now().isoformat(timespec="seconds"), "settings": {**base, "grounding_checker": "none"},
              "index_copied_from": str(source.relative_to(REPO_ROOT)) if source else None, "configs": {}}
    for size in SIZES:
        cid = f"K{size}"
        print(f"[context] {cid}: top_k_rerank {size}…")
        record = ss.run_config(cid, {**base, "top_k_rerank": size}, sets, corpus, work / cid, copy_from=source)
        if record["status"] == "ran" and not skip_ragas:
            print(f"[context] {cid}: Ragas…")
            record["ragas"] = ss.ragas(cid, record, f"context-{stamp}")
        result["configs"][cid] = record
        ss.save_run(result, path)
    if result["configs"].get("K5", {}).get("status") == "ran":
        for record in result["configs"].values():
            if record.get("status") == "ran":
                record["gates"] = gates(record, result["configs"]["K5"])
    result["chosen_top_k"], result["reason"] = decide(result["configs"])
    result["chosen_ragas_file"] = (result["configs"].get(f"K{result['chosen_top_k']}", {}).get("ragas") or {}).get("file") \
        if isinstance(result["configs"].get(f"K{result['chosen_top_k']}", {}).get("ragas"), dict) else None
    result["finished"] = datetime.now().isoformat(timespec="seconds")
    ss.save_run(result, path)
    return result, path


def print_report(result: dict, path: Path):
    """Prints both records, their gates and the decision."""
    print("\nContext size · top_k_rerank 5 vs 3 (unchecked)\n")
    for cid, r in result["configs"].items():
        if r.get("status") != "ran":
            print(f"{cid}: {r.get('status')} ({r.get('reason')})")
            continue
        g, lat, rag = r["golden"], r["latency"], r.get("ragas")
        faith = rag.get("faithfulness") if isinstance(rag, dict) else rag
        print(f"{cid}: tables {r['tables']['right_value_and_page']} · figures {r['figures']['right_value_and_page']} · "
              f"answered {g['answered']}/{g['in_scope']} · refused {g['refused']}/{g['out_of_scope']} · faithfulness "
              f"{faith} · typed p50 {lat.get('typed_p50_ms') or 0:.0f} ms · first audio p50 "
              f"{lat.get('first_audio_p50_ms') or 0:.0f} ms · gates {r.get('gates')}")
    print(f"\nChosen: top_k_rerank {result['chosen_top_k']} ({result['reason']})")
    print(f"Saved {path.relative_to(REPO_ROOT)}")


def main(argv=None) -> int:
    """Command line entry: `python -m src.eval.context_size run [--apply] [--skip-ragas]`."""
    parser = argparse.ArgumentParser(description="Context size runner (spec 0006)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--apply", action="store_true", help="write the chosen top_k_rerank to config.yaml")
    run_parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args(argv)
    try:
        result, path = run(args.skip_ragas)
        print_report(result, path)
        if args.apply:
            apply(result["chosen_top_k"])
            print(f"[context] Applied top_k_rerank {result['chosen_top_k']} to config/config.yaml.")
    except BenchmarkError as e:
        print(f"[context_size] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
