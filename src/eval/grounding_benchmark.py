"""Grounding benchmark: which sentence checker agrees best with Ragas?

    python -m src.eval.grounding_benchmark run [--apply]

Reads the unchecked answers of the newest context size run (else sweep run) with their retrieved contexts and
Ragas faithfulness, runs the NLI checker at several thresholds and the llama3.2:1b judge, and applies the
winner rule: at most 2 false alarms, golden answered at or above 90% of in scope after refusals, then the
most low faithfulness answers touched, near ties to the fastest. `--apply` writes the winner and then runs
the table and figure regression gate with the checker on.
"""
import argparse
import ast
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.eval import system_sweep as ss
from src.eval import table_benchmark as tb
from src.eval.table_benchmark import BenchmarkError, REPO_ROOT, nearest_rank
from src.utils.text_sanitizer import is_refusal
from src.verification import grounding

NLI_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
LOW_FAITHFULNESS = 0.85
MAX_FALSE_ALARMS = 2
MIN_ANSWERED = 0.90
NEAR_TIE = 1
GATE_SLACK = 2
GOLDEN_PATH = REPO_ROOT / "data" / "eval" / "golden_dataset.json"


# --- input ---

RUN_KINDS = {"context": lambda r: f"K{r.get('chosen_top_k')}",
             "style": lambda r: "S1" if r.get("chosen_style") == "short_first" else "S0",
             "sweep": lambda r: r.get("winner")}


def unchecked_record(path: Path):
    """(Ragas CSV, kept index folder, record) of a run file's chosen setting, or None when it has no Ragas CSV."""
    kind = path.name.split("-", 1)[0]
    run = json.loads(path.read_text(encoding="utf-8"))
    record = (run.get("configs") or {}).get(RUN_KINDS[kind](run) or "")
    if not record or not isinstance(record.get("ragas"), dict):
        return None
    checker = (record.get("settings") or {}).get("grounding_checker") or \
        (run.get("settings") or {}).get("grounding_checker", "none")
    if checker not in (None, "none"):
        raise BenchmarkError(f"{path.name} was run with grounding checker {checker!r}; its answers are "
                             "already filtered. Run the context size or answer style runner (unchecked) first.")
    return REPO_ROOT / record["ragas"]["file"], REPO_ROOT / record["workdir"], record


def newest_unchecked_run(out_dir: Path = ss.OUT_DIR, run_file: Path = None) -> tuple:
    """(run file, Ragas CSV, kept index folder, record): the given run file, else the newest context size, answer
    style or sweep run (by the time stamp in its name) whose chosen setting has a Ragas CSV."""
    if run_file is not None:
        found = unchecked_record(Path(run_file))
        if found is None:
            raise BenchmarkError(f"{Path(run_file).name} has no Ragas CSV for its chosen setting.")
        return (Path(run_file), *found)
    stamped = [(m.group(1), path) for kind in RUN_KINDS for path in out_dir.glob(f"{kind}-*.json")
               if (m := re.search(r"(\d{8}-\d{6})\.json$", path.name))]
    for _, path in sorted(stamped, reverse=True):
        found = unchecked_record(path)
        if found is not None:
            return (path, *found)
    raise BenchmarkError("No context size, answer style or sweep run with a Ragas CSV found in data/benchmarks/.")


def load_rows(csv_path: Path) -> list:
    """The Ragas CSV rows: question, answer, retrieved contexts and faithfulness (None when Ragas gave none)."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            faith = r.get("faithfulness")
            rows.append({"question": r["user_input"], "answer": r["response"],
                         "contexts": ast.literal_eval(r["retrieved_contexts"]),
                         "faithfulness": float(faith) if faith not in (None, "", "nan") else None})
    return rows


def in_scope_count(path: Path = GOLDEN_PATH) -> int:
    """How many golden questions are in scope (the answered count's denominator)."""
    return sum(1 for g in json.loads(path.read_text(encoding="utf-8")) if g.get("in_scope"))


# --- checkers over the rows ---

class CachedNLI:
    """The NLI checker's verdict at one threshold, from entailment scores computed once per sentence."""
    def __init__(self, scores: dict, threshold: float, model_name: str):
        self.scores, self.threshold, self.name = scores, threshold, f"nli:{model_name}@{threshold}"

    def supported(self, sentence, chunks):
        """The precomputed NLI score of this sentence against these chunks, compared with the threshold."""
        return self.scores[(sentence, tuple(chunks))] >= self.threshold


def nli_scores(rows: list) -> tuple:
    """(scores by (sentence, chunks), ms per sentence after warm up, model name)."""
    nli = grounding.NLIChecker()
    nli.entailment_scores("Warm up.", ["Warm up."])
    scores, timings = {}, []
    for row in rows:
        for sentence in grounding.split_sentences(row["answer"]):
            text = grounding.plain(sentence)
            if grounding.word_count(sentence) < grounding.MIN_CHECK_WORDS or is_refusal(sentence):
                continue
            t = time.perf_counter()
            values = nli.entailment_scores(text, row["contexts"])
            timings.append((time.perf_counter() - t) * 1000)
            scores[(text, tuple(row["contexts"]))] = max(values) if values else 0.0
    return scores, timings, nli.model_name


class TimedJudge:
    """The live 1B judge, timing every call for the ms per sentence column."""
    def __init__(self):
        self.judge = grounding.JudgeChecker()       # warms up on creation
        self.name, self.timings = self.judge.name, []

    def supported(self, sentence, chunks):
        """The judge's verdict, with its time recorded."""
        t = time.perf_counter()
        try:
            return self.judge.supported(sentence, chunks)
        finally:
            self.timings.append((time.perf_counter() - t) * 1000)


def evaluate(rows: list, checker, in_scope: int, timings: list) -> dict:
    """Touched, refused, low faithfulness caught, false alarms, answered after refusals, ms per sentence."""
    touched = refused = low_touched = false_alarms = 0
    per_row = []
    for row in rows:
        result = grounding.check_answer(row["answer"], row["contexts"], checker)
        was_refused = bool(result.dropped) and not result.model_refused \
            and not any(not is_refusal(k) for k in result.kept)
        hit = bool(result.dropped) or was_refused
        touched += hit
        refused += was_refused
        f = row["faithfulness"]
        if hit and f is not None:
            low_touched += f < LOW_FAITHFULNESS
            false_alarms += f >= LOW_FAITHFULNESS
        per_row.append({"question": row["question"], "faithfulness": f, "dropped": result.dropped,
                        "refused": was_refused, "invalid": result.invalid})
    answered = len(rows) - refused
    return {"answers": len(rows), "touched": touched, "refused": refused, "low_faithfulness_touched": low_touched,
            "false_alarms": false_alarms, "answered_after": answered, "in_scope": in_scope,
            "ms_per_sentence": {"p50": nearest_rank(timings, 0.5), "p95": nearest_rank(timings, 0.95), "n": len(timings)},
            "rows": per_row}


def pick_winner(candidates: dict) -> dict:
    """Qualify on false alarms and coverage; most low faithfulness touched; near ties to the fastest."""
    qualified = {name: c for name, c in candidates.items()
                 if c["false_alarms"] <= MAX_FALSE_ALARMS and c["answered_after"] >= MIN_ANSWERED * c["in_scope"]}
    if not qualified:
        return {"outcome": "none", "reason": "no checker qualifies (at most 2 false alarms and 90% answered)"}
    top = max(c["low_faithfulness_touched"] for c in qualified.values())
    tied = [n for n, c in qualified.items() if c["low_faithfulness_touched"] >= top - NEAR_TIE]
    speed = lambda n: qualified[n]["ms_per_sentence"]["p50"] if qualified[n]["ms_per_sentence"]["p50"] is not None else 1e9
    winner = min(tied, key=lambda n: (speed(n), -qualified[n]["low_faithfulness_touched"]))
    c = qualified[winner]
    return {"outcome": winner, "near_ties": tied,
            "reason": f"{winner} touches {c['low_faithfulness_touched']} low faithfulness answers with "
                      f"{c['false_alarms']} false alarms, {c['answered_after']} of {c['in_scope']} answered"
                      + ("; near tie broken by ms per sentence" if len(tied) > 1 else "")}


# --- apply and the regression gate ---

def apply_checker(name: str):
    """Writes grounding.checker (and the NLI threshold) to config/config.yaml, keeping comments."""
    import re
    checker, _, threshold = name.partition("@")
    kind = "nli" if checker.startswith("nli:") else ("judge" if checker.startswith("judge:") else "none")
    text = ss.CONFIG_PATH.read_text(encoding="utf-8")
    # Only inside the grounding section, so another section's `threshold:` key is never touched
    section = re.search(r"^grounding:\n(?:[ \t]+.*\n|[ \t]*\n)*", text, flags=re.M)
    if not section:
        raise BenchmarkError("Could not find the grounding section in config/config.yaml")
    block, n = re.subn(r'^(\s+checker:\s*)"[^"]*"', rf'\g<1>"{kind}"', section.group(0), count=1, flags=re.M)
    if n != 1:
        raise BenchmarkError("Could not find grounding.checker in config/config.yaml")
    if threshold:
        block = re.sub(r"^(\s+threshold:\s*)[\d.]+", rf"\g<1>{threshold}", block, count=1, flags=re.M)
    text = text[:section.start()] + block + text[section.end():]
    ss.CONFIG_PATH.write_text(text, encoding="utf-8")
    return kind, threshold


def regression_gate(run_record: dict, index_folder: Path, kind: str, threshold: str, stamp: str) -> dict:
    """Table and figure questions through the API with the checker on, against the same run unchecked."""
    sets = ss.load_sets()
    work = ss.OUT_DIR / f"grounding-{stamp}-gate"
    work.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((index_folder / "config.yaml").read_text(encoding="utf-8"))
    config.setdefault("grounding", {}).update(checker=kind)
    if threshold:
        config["grounding"]["threshold"] = float(threshold)
    with open(work / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    ss.unload_all()
    api = ss.Api(work, "gate")
    try:
        health, _, _ = api.start()
        if health is None or api.rebuilt():
            return {"passed": False, "reason": "the API did not start on the kept index without a rebuild"}
        effective = health.get("grounding_checker_effective")
        if not effective or effective == "none":
            return {"passed": False, "reason": f"grounding_checker_effective is {effective!r}"}
        tables = tb.ask_all(api.base, [q for q in sets[0]["questions"] if q.get("reviewed")], "gate tables")
        figures = tb.ask_all(api.base, [q for q in sets[1]["questions"] if q.get("reviewed")], "gate figures")
    finally:
        api.stop()
    t, f = tb.summarize(tables)["right_value_and_page"], tb.summarize(figures)["right_value_and_page"]
    t0, f0 = run_record["tables"]["right_value_and_page"], run_record["figures"]["right_value_and_page"]
    passed = t >= t0 - GATE_SLACK and f >= f0 - GATE_SLACK
    return {"passed": passed, "tables": t, "tables_unchecked": t0, "figures": f, "figures_unchecked": f0,
            "reason": f"tables {t} (unchecked {t0}), figures {f} (unchecked {f0})"}


# --- the run ---

def run(run_file: Path = None) -> tuple:
    """Scores every candidate checker on the chosen run's Ragas answers and picks the winner; returns (report, file,
    record, index folder, stamp)."""
    run_path, csv_path, index_folder, record = newest_unchecked_run(run_file=run_file)
    rows = load_rows(csv_path)
    in_scope = in_scope_count()
    print(f"[grounding] {len(rows)} answers from {csv_path.name} (run {run_path.name})")
    candidates = {}
    scores, nli_ms, model_name = nli_scores(rows)
    for threshold in NLI_THRESHOLDS:
        checker = CachedNLI(scores, threshold, model_name)
        candidates[checker.name] = evaluate(rows, checker, in_scope, nli_ms)
        print(f"[grounding] {checker.name}: {summary_line(candidates[checker.name])}")
    judge = TimedJudge()
    candidates[judge.name] = evaluate(rows, judge, in_scope, judge.timings)
    print(f"[grounding] {judge.name}: {summary_line(candidates[judge.name])}")
    report = {"started": datetime.now().isoformat(timespec="seconds"), "run_file": run_path.name,
              "ragas_file": csv_path.name, "index": str(index_folder.relative_to(REPO_ROOT)),
              "candidates": candidates, "winner": pick_winner(candidates)}
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    path = ss.OUT_DIR / f"grounding-{stamp}.json"
    ss.save_run(report, path)
    return report, path, record, index_folder, stamp


def summary_line(c: dict) -> str:
    """One printed line per candidate: touched, refused, low faithfulness touched, false alarms, answered, ms per
    sentence."""
    ms = c["ms_per_sentence"]["p50"]
    return (f"touched {c['touched']}, refused {c['refused']}, low faithfulness touched {c['low_faithfulness_touched']}, "
            f"false alarms {c['false_alarms']}, answered {c['answered_after']}/{c['in_scope']}, "
            f"{'-' if ms is None else f'{ms:.0f}'} ms per sentence")


def main(argv=None) -> int:
    """Command line entry: `python -m src.eval.grounding_benchmark run [--apply] [--run FILE]`."""
    parser = argparse.ArgumentParser(description="Grounding benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--apply", action="store_true")
    run_parser.add_argument("--run", type=Path, default=None, help="a context, style or sweep run file (default: newest)")
    args = parser.parse_args(argv)
    try:
        report, path, record, index_folder, stamp = run(args.run)
        w = report["winner"]
        print(f"\nOutcome: {w['outcome']} ({w['reason']})")
        if args.apply and w["outcome"] != "none":
            kind, threshold = apply_checker(w["outcome"])
            print(f"[grounding] Applied grounding.checker {kind}{f' at {threshold}' if threshold else ''}; "
                  "running the table and figure regression gate…")
            gate = regression_gate(record, index_folder, kind, threshold, stamp)
            report["regression_gate"] = gate
            if not gate["passed"]:
                apply_checker("none")
                print(f"[grounding] Regression gate failed ({gate['reason']}); grounding.checker set back to none.")
            else:
                print(f"[grounding] Regression gate passed ({gate['reason']}).")
            ss.save_run(report, path)
        print(f"Saved {path.relative_to(REPO_ROOT)}")
    except BenchmarkError as e:
        print(f"[grounding_benchmark] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
