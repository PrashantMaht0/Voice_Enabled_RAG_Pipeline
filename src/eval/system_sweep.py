"""Two stage configuration sweep (spec 0005, child config-sweep).

    python -m src.eval.system_sweep run [--stage A|B|all] [--skip-ragas] [--resume <run file>] [--apply]

Stage A compares figure readers (none, qwen2.5vl:3b, qwen2.5vl:7b) with today's answer setup; stage B compares
the answer setup (llama3.2:3b with llama3.2:1b, or qwen2.5vl:3b for both) on the stage A winner's index. Every
configuration runs through the real API on the four PDF corpus (Azure book, Raspberry Pi datasheet, Census
P60-273, UN SDG Report 2024) with its own work folder and no caches. A0 (today's setup) runs first and calibrates
the gates; a gate is never relaxed. The run file is saved after every configuration, and `--resume` continues it.
`config/config.yaml` changes only with `--apply`.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

from src.eval import figure_benchmark as fb
from src.eval import table_benchmark as tb
from src.eval.table_benchmark import BenchmarkError, REPO_ROOT

OUT_DIR = REPO_ROOT / "data" / "benchmarks"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
TABLE_PARSER = "layoutlmv3"              # spec 0004's winner, the same in every configuration
INGEST_TIMEOUT_S = 3 * 60 * 60
PS_POLL_S = 5
NEAR_TIE = 2                             # configurations within this many answers of the top tie
MIN_FIGURE_GAIN = 3                      # a figure reader must beat A0's figure score by this much
TABLE_SLACK = 2                          # tables may fall at most this many answers below A0
MIN_ANSWERED = 0.90
MIN_FAITHFULNESS = 0.85
MAX_EXTRA_RAGAS = 2
FOLLOW_UP = "What about its pricing?"

CONFIGS = {
    "A0": {"stage": "A", "figure_parser": "none", "generation_model": "llama3.2:3b", "rewriter_model": "llama3.2:1b"},
    "A3": {"stage": "A", "figure_parser": "qwen2.5vl:3b", "generation_model": "llama3.2:3b",
           "rewriter_model": "llama3.2:1b"},
    "A7": {"stage": "A", "figure_parser": "qwen2.5vl:7b", "generation_model": "llama3.2:3b",
           "rewriter_model": "llama3.2:1b"},
    "B-llama": {"stage": "B", "generation_model": "llama3.2:3b", "rewriter_model": "llama3.2:1b"},
    "B-qwen": {"stage": "B", "generation_model": "qwen2.5vl:3b", "rewriter_model": "qwen2.5vl:3b"},
}
NOTES = [
    "Every configuration indexes the four PDF corpus from scratch in its own work folder, with the answer cache "
    "and the detection, table and figure read caches off; B-qwen runs on an exact copy of the stage A winner's index.",
    "Golden answered and refused counts come from the latency benchmark's first pass records (is_refusal), so the "
    "golden scores come from the four PDF corpus and are not comparable with the 2026-08-21 Ragas run.",
    "The figure questions were drafted by qwen2.5vl:7b from each crop and reviewed by hand.",
    "Figure descriptions are generated and can be wrong; tables the detector misses are classified table and "
    "indexed nowhere.",
]


# --- the run file ---

def git(*args) -> str:
    """Output of a git command run in the repository; empty when git fails."""
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def model_digests() -> dict:
    """Each pulled Ollama model's digest, so a run records exactly which weights it used."""
    import ollama
    try:
        return {m.model: m.digest for m in ollama.list().models}
    except Exception:
        return {}


def new_run(stamp: str) -> dict:
    """A fresh run record: time, commit, dirty flag, model digests and machine."""
    return {"started": datetime.now().isoformat(timespec="seconds"), "updated": None, "stamp": stamp,
            "git_commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "model_digests": model_digests(),
            "machine": {"os": platform.platform(), "cpu": platform.processor() or platform.machine()},
            "configs": {}, "stage_a_winner": None, "winner": None, "reason": None, "notes": NOTES}


def save_run(run: dict, path: Path):
    """Writes the run file (after every configuration, so `--resume` can continue)."""
    run["updated"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- Ollama memory ---

def ollama_ps() -> list:
    """The models Ollama has loaded right now, with their sizes."""
    import ollama
    try:
        return [{"model": m.model, "size": m.size, "size_vram": m.size_vram} for m in ollama.ps().models]
    except Exception:
        return []


def unload_all():
    """Unloads every loaded Ollama model, so each configuration starts with free memory."""
    import ollama
    for m in ollama_ps():
        try:
            ollama.generate(model=m["model"], prompt="", keep_alive=0)
        except Exception as e:
            print(f"[sweep] Could not unload {m['model']}: {e}")


class PsPoller(threading.Thread):
    """Polls `ollama ps` every 5 s and keeps the snapshot with the most memory in use."""
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = [], threading.Event()

    def run(self):
        """Keeps the largest memory snapshot seen until stopped."""
        while not self.stop.is_set():
            snapshot = ollama_ps()
            if sum(m["size"] or 0 for m in snapshot) > sum(m["size"] or 0 for m in self.peak):
                self.peak = snapshot
            self.stop.wait(PS_POLL_S)


# --- one configuration ---

def config_settings(cid: str, figure_parser: str) -> dict:
    """A configuration's settings: the fixed table reader, its figure reader and its two models."""
    c = CONFIGS[cid]
    return {"table_parser": TABLE_PARSER, "figure_parser": figure_parser,
            "generation_model": c["generation_model"], "rewriter_model": c["rewriter_model"]}


def write_config(settings: dict, corpus: Path, workdir: Path) -> Path:
    """A work config.yaml for one configuration: its corpus copy, its own index folder, cache off, unchecked, and its
    readers and models."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"]["raw_data_dir"] = str(corpus)
    config["paths"]["qdrant_storage"] = str(workdir / "index" / "qdrant_storage")
    config["paths"]["tantivy_index"] = str(workdir / "index" / "tantivy_index")
    config["cache"]["enabled"] = False
    config["ingestion"].update(table_parser=settings["table_parser"], figure_parser=settings["figure_parser"],
                               table_cache=False)
    config["llm"].update(generation_model=settings["generation_model"], rewriter_model=settings["rewriter_model"])
    config["llm"]["answer_style"] = settings.get("answer_style") or "full"      # a record without one used today's prompt
    # Measurements record raw generations; the grounding benchmark reads them (spec 0006)
    config.setdefault("grounding", {})["checker"] = "none"
    if settings.get("top_k_rerank") is not None:
        config["retrieval"]["top_k_rerank"] = int(settings["top_k_rerank"])
    path = workdir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


class Api:
    """The API on a free port with the configuration's config and reader log; polls memory while it starts."""
    def __init__(self, workdir: Path, label: str):
        self.workdir, self.label = workdir, label
        self.port = tb.free_port()
        self.base = f"http://127.0.0.1:{self.port}/api"
        self.log_path = workdir / f"server-{label}.log"
        self.reader_log = workdir / "reader.jsonl"
        self.proc = None

    def start(self) -> tuple:
        """(health or None, wall seconds to the first healthy /api/health, peak ollama ps snapshot)."""
        env = dict(os.environ, VRAG_CONFIG=str(self.workdir / "config.yaml"), VRAG_TABLE_LOG=str(self.reader_log))
        poller = PsPoller()
        poller.start()
        start = time.perf_counter()
        with open(self.log_path, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                          "--port", str(self.port)], cwd=REPO_ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
        health = None
        try:
            while time.perf_counter() - start < INGEST_TIMEOUT_S and self.proc.poll() is None:
                try:
                    health = requests.get(f"{self.base}/health", timeout=5).json()
                    break
                except requests.RequestException:
                    time.sleep(2)
        finally:
            poller.stop.set()
        return health, round(time.perf_counter() - start, 1), poller.peak

    def rebuilt(self) -> bool:
        """True when the API rebuilt its index while starting (so a reused index was not reused)."""
        return "Starting vectorization" in self.log_path.read_text(encoding="utf-8", errors="replace")

    def stop(self):
        """Stops the API process, killing it if it does not exit in time."""
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


def latency(api: Api, cid: str) -> dict:
    """The latency benchmark against this API: typed, refusal and spoken latency plus the golden counts."""
    from src.eval.latency_benchmark import Benchmark
    path, run = Benchmark(api_base=api.base, ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                          label=f"sweep_{cid}").run()
    first = run["summary"]["first"]
    p = lambda block, metric, q: (first.get(block, {}).get("metrics", {}).get(metric) or {}).get(q)
    records = [r for r in run["text"] if r["pass"] == "first"]
    in_scope = [r for r in records if r["in_scope"]]
    out_scope = [r for r in records if not r["in_scope"]]
    return {
        "latency": {"typed_p50_ms": p("answered", "client_ms", "p50"), "typed_p95_ms": p("answered", "client_ms", "p95"),
                    "refusal_p50_ms": p("refused", "client_ms", "p50"),
                    "spoken_p50_ms": p("voice", "total_ms", "p50") if run["meta"].get("voice_valid") else None,
                    "first_audio_p50_ms": p("voice", "first_audio_ms", "p50") if run["meta"].get("voice_valid") else None,
                    "max_gap_p95_ms": p("voice", "max_gap_ms", "p95") if run["meta"].get("voice_valid") else None,
                    "power": run["meta"].get("machine", {}).get("power"),
                    "grounding_checker": run["meta"].get("grounding_checker_effective"),
                    "file": str(Path(path).relative_to(REPO_ROOT))},
        "golden": {"answered": sum(r["error"] is None and r["refused"] is False for r in in_scope),
                   "in_scope": len(in_scope),
                   "refused": sum(r["error"] is None and r["refused"] is True for r in out_scope),
                   "out_of_scope": len(out_scope),
                   "leaked": [r["id"] for r in out_scope if r["refused"] is not True]},
        "aborted": bool(run["meta"].get("aborted")),
    }


def follow_up(api: Api, question: str) -> list:
    """One follow up with a one turn history, so the rewriter loads; returns `ollama ps` while serving."""
    try:
        first = requests.post(f"{api.base}/query", json={"query": question, "history": []},
                              timeout=tb.QUERY_TIMEOUT_S).json()
        history = [{"role": "user", "content": question}, {"role": "assistant", "content": first.get("answer", "")}]
        requests.post(f"{api.base}/query", json={"query": FOLLOW_UP, "history": history}, timeout=tb.QUERY_TIMEOUT_S)
    except requests.RequestException as e:
        print(f"[sweep] Follow up failed: {e}")
    return ollama_ps()


def ingestion_metrics(reader_log: Path, figure_set: dict, wall: float, peak: list) -> dict:
    """Ingestion cost from the reader log: detector seconds per PDF, figure calls, wall time and peak model memory."""
    calls, detections, figure_lines = tb.read_table_log(reader_log)
    per_pdf = {}
    for d in detections:
        per_pdf[d["file"]] = per_pdf.get(d["file"], 0.0) + d["detect_seconds"]
    for line in figure_lines:
        if "seconds" in line and str(line.get("reader", "")).startswith("figure:"):
            per_pdf[line["file"]] = per_pdf.get(line["file"], 0.0) + line["seconds"]
    timing = fb.figure_timings(figure_lines)
    recall = fb.figure_recall(figure_lines, figure_set) if figure_lines else None
    return {"wall_seconds": wall, "reader_seconds_per_pdf": {k: round(v, 1) for k, v in per_pdf.items()},
            "seconds_per_figure": {"classify_p50": timing["classify"]["p50_s"], "extract_p50": timing["extract"]["p50_s"],
                                   "max": timing["extract"]["max_s"]},
            "figure_counts": {k: timing[k] for k in ("retries", "failed", "timeouts", "errors", "missed_tables")},
            "figures_found": recall["found"] if recall else 0, "figure_recall": recall["recall"] if recall else None,
            "peak_during_ingestion": peak}


def run_config(cid: str, settings: dict, sets: tuple, corpus: Path, workdir: Path, copy_from: Path = None) -> dict:
    """Steps 2 to 5 of the sweep flow for one configuration."""
    table_set, figure_set = sets
    workdir.mkdir(parents=True, exist_ok=True)
    if copy_from is not None:
        shutil.copytree(copy_from / "index", workdir / "index", dirs_exist_ok=False)
    write_config(settings, corpus, workdir)
    record = {"status": "failed", "settings": settings, "workdir": str(workdir.relative_to(REPO_ROOT))}
    unload_all()
    api = Api(workdir, "run")
    try:
        health, wall, peak = api.start()
        if health is None:
            return {**record, "reason": "the API did not start"}
        for key, wanted in (("table_parser_effective", settings["table_parser"]),
                            ("figure_parser_effective", settings["figure_parser"])):
            if health.get(key) != wanted:
                return {**record, "status": "skipped", "reason": f"{key} is {health.get(key)!r}, not {wanted!r}"}
        if copy_from is not None and api.rebuilt():
            return {**record, "reason": "the copied index was rebuilt"}
        tables = tb.ask_all(api.base, [q for q in table_set["questions"] if q.get("reviewed") is True], f"{cid} tables")
        figures = tb.ask_all(api.base, [q for q in figure_set["questions"] if q.get("reviewed") is True],
                             f"{cid} figures")
        lat = latency(api, cid)
        golden_question = next(q["question"] for q in json.loads(
            (REPO_ROOT / "data" / "eval" / "golden_dataset.json").read_text(encoding="utf-8")) if q.get("in_scope"))
        serving = follow_up(api, golden_question)
    finally:
        api.stop()
    record.update({
        "status": "failed" if lat["aborted"] else "ran",
        "reason": "the latency benchmark aborted" if lat["aborted"] else None,
        "ingestion": ingestion_metrics(api.reader_log, figure_set, wall, peak),
        "tables": tb.summarize(tables), "figures": tb.summarize(figures),
        "golden": lat["golden"], "latency": lat["latency"],
        "memory": {"peak_during_ingestion": peak, "while_serving": serving},
        "ragas": None,
    })
    return record


# --- gates and the winner rule (AC-13) ---

def right(record: dict, key: str) -> int:
    """A record's right value and page count for tables or figures."""
    return record[key]["right_value_and_page"]


def base_gates(record: dict, a0: dict) -> dict:
    """The gates that need no Ragas."""
    g = record["golden"]
    return {"all_trick_questions_refused": g["refused"] == g["out_of_scope"],
            "answered_90_percent": g["in_scope"] > 0 and g["answered"] / g["in_scope"] >= MIN_ANSWERED,
            "tables_within_2_of_a0": right(record, "tables") >= right(a0, "tables") - TABLE_SLACK}


def passes(record: dict) -> bool:
    """True when the configuration ran and passed every gate."""
    return record.get("status") == "ran" and all(record.get("gates", {}).values())


def rank(records: dict, key) -> list:
    """Ids by score, highest first; configurations within NEAR_TIE of each other's group top go to the lowest p50."""
    ids = sorted(records, key=lambda i: -key(records[i]))
    ordered = []
    while ids:
        top = key(records[ids[0]])
        tied = [i for i in ids if key(records[i]) >= top - NEAR_TIE]
        p50 = lambda i: records[i]["latency"]["typed_p50_ms"]
        tied.sort(key=lambda i: (p50(i) is None, p50(i) or 0.0, -key(records[i])))
        ordered += tied
        ids = [i for i in ids if i not in tied]
    return ordered


def stage_a_winner(run: dict) -> tuple:
    """The stage A configuration that passes every gate with the most table plus figure answers, and why."""
    configs = run["configs"]
    a0 = configs.get("A0")
    passing = {i: configs[i] for i in ("A0", "A3", "A7") if i in configs and passes(configs[i])}
    if not passing:
        return None, "no stage A configuration passed the gates"
    best = rank(passing, lambda r: right(r, "figures"))[0]
    if best != "A0" and "A0" in passing:
        gain = right(passing[best], "figures") - right(a0, "figures")
        if gain < MIN_FIGURE_GAIN:
            return "A0", f"{best} beats A0 on figures by {gain}, fewer than {MIN_FIGURE_GAIN}"
        return best, f"{best} scores {right(passing[best], 'figures')} on figures, {gain} above A0"
    return best, f"{best} has the most figure answers among the configurations that pass"


def overall_order(run: dict) -> list:
    """Every passing configuration ranked by table plus figure answers (a reused record counted once)."""
    configs = run["configs"]
    # B-llama is the stage A winner's own record, so it is ranked once, under the stage A id
    passing = {i: r for i, r in configs.items() if passes(r) and not r.get("reused_from")}
    return rank(passing, lambda r: right(r, "tables") + right(r, "figures"))


# --- Ragas for finalists ---

def ragas(cid: str, record: dict, stamp: str) -> object:
    """Restarts the configuration on its kept index and runs the Ragas gate; the mean scores or "inconclusive"."""
    import csv
    workdir = REPO_ROOT / record["workdir"]
    scores_path = REPO_ROOT / "data" / "eval" / "last_run_scores.csv"
    unload_all()
    api = Api(workdir, "ragas")
    try:
        health, _, _ = api.start()
        if health is None or api.rebuilt():
            return "inconclusive"
        if scores_path.exists():
            scores_path.unlink()
        env = dict(os.environ, VRAG_API_URL=f"{api.base}/query")
        # --noconftest: tests/conftest.py imports the API, whose packages eval-env deliberately leaves out
        subprocess.run([str(REPO_ROOT / "eval-env" / "bin" / "python"), "-m", "pytest", "tests/eval_test.py",
                        "-m", "ragas_ci", "--noconftest", "-q", "-s"], cwd=REPO_ROOT, env=env)
    finally:
        api.stop()
    if not scores_path.exists():
        return "inconclusive"
    copy = OUT_DIR / f"sweep-{stamp}-{cid}-ragas.csv"
    shutil.copy(scores_path, copy)
    with open(scores_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mean = lambda col: (lambda v: round(sum(v) / len(v), 3) if v else None)(
        [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")])
    return {"faithfulness": mean("faithfulness"), "answer_relevancy": mean("answer_relevancy"),
            "file": str(copy.relative_to(REPO_ROOT))}


def run_ragas(run: dict, order: list, stamp: str, path: Path):
    """Ragas for A0 and the top configuration; if the top fails faithfulness, the next one, at most twice more."""
    def gate(cid):
        record = run["configs"][cid]
        if record.get("ragas") is None:
            print(f"[sweep] Ragas for {cid}…")
            record["ragas"] = ragas(cid, record, stamp)
            save_run(run, path)
        r = record["ragas"]
        record["gates"]["faithfulness_085"] = isinstance(r, dict) and (r["faithfulness"] or 0) >= MIN_FAITHFULNESS
        save_run(run, path)
        return record["gates"]["faithfulness_085"]

    if "A0" in run["configs"] and run["configs"]["A0"].get("status") == "ran":
        gate("A0")
    tried = 0
    for cid in order:
        if cid == "A0":
            if passes(run["configs"]["A0"]):
                return "A0"
            continue
        if gate(cid):
            return cid
        tried += 1
        if tried > MAX_EXTRA_RAGAS:
            break
    return None


# --- apply ---

def apply_winner(settings: dict):
    """Writes the four model values into config/config.yaml, keeping its comments."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for key in ("figure_parser", "generation_model", "rewriter_model"):
        text, n = re.subn(rf'^(\s+{key}:\s*)"[^"]*"', rf'\g<1>"{settings[key]}"', text, count=1, flags=re.M)
        if n != 1:
            raise BenchmarkError(f"Could not find {key} in {CONFIG_PATH}")
    CONFIG_PATH.write_text(text, encoding="utf-8")


# --- the sweep ---

def load_sets() -> tuple:
    """The reviewed table and figure question sets, after checking their PDFs are present."""
    table_set, figure_set = tb.load_set(tb.QUESTIONS_PATH), tb.load_set(fb.QUESTIONS_PATH)
    tb.check_pdfs(table_set)
    tb.check_pdfs(figure_set)
    return table_set, figure_set


def sweep(stage: str, skip_ragas: bool, resume: Path = None) -> tuple:
    """Runs the chosen stages (or resumes a run file), then Ragas on the finalists and the winner rule; returns (run,
    path)."""
    sets = load_sets()
    if resume:
        path = resume.resolve()
        run = json.loads(path.read_text(encoding="utf-8"))
        stamp = run["stamp"]
    else:
        busy = ollama_ps()
        if busy:
            raise BenchmarkError(f"Ollama already has models loaded ({', '.join(m['model'] for m in busy)}); "
                                 "stop whatever is using them, or unload them, then run the sweep again.")
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        path = OUT_DIR / f"sweep-{stamp}.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        run = new_run(stamp)
    work = OUT_DIR / f"sweep-{stamp}-work"
    corpus = tb.corpus_dir(list(sets), work / "corpus")     # raises when the Azure book's SHA256 differs
    save_run(run, path)

    if stage in ("A", "all"):
        for cid in ("A0", "A3", "A7"):
            if cid in run["configs"]:
                continue
            print(f"[sweep] {cid}: indexing the corpus and asking every question…")
            record = run_config(cid, config_settings(cid, CONFIGS[cid]["figure_parser"]), sets, corpus, work / cid)
            if record["status"] == "ran":
                a0 = record if cid == "A0" else run["configs"].get("A0")
                record["gates"] = base_gates(record, a0)
            run["configs"][cid] = record
            save_run(run, path)
            if cid == "A0" and not passes(record):
                run["winner"], run["reason"] = None, "no winner, config unchanged: A0 (today's setup) failed a gate"
                run["leaked"] = (record.get("golden") or {}).get("leaked", [])
                save_run(run, path)
                return run, path
        run["stage_a_winner"], run["stage_a_reason"] = stage_a_winner(run)
        save_run(run, path)

    if stage in ("B", "all") and run.get("stage_a_winner"):
        winner = run["stage_a_winner"]
        base = run["configs"][winner]
        if "B-llama" not in run["configs"]:
            run["configs"]["B-llama"] = {**base, "reused_from": winner,
                                         "settings": config_settings("B-llama", base["settings"]["figure_parser"])}
            save_run(run, path)
        if "B-qwen" not in run["configs"]:
            print(f"[sweep] B-qwen: copying {winner}'s index and asking every question…")
            record = run_config("B-qwen", config_settings("B-qwen", base["settings"]["figure_parser"]), sets, corpus,
                                work / "B-qwen", copy_from=REPO_ROOT / base["workdir"])
            if record["status"] == "ran":
                record["gates"] = base_gates(record, run["configs"]["A0"])
            run["configs"]["B-qwen"] = record
            save_run(run, path)

    order = overall_order(run)
    if not order:
        run["winner"], run["reason"] = None, "no winner: no configuration passed the gates"
    elif skip_ragas:
        run["winner"], run["reason"] = None, f"undecided: Ragas skipped; best by the other gates is {order[0]}"
        run["candidate"] = order[0]
    else:
        winner = run_ragas(run, order, stamp, path)
        run["winner"] = winner
        run["reason"] = (f"{winner} passes every gate and has the most table plus figure answers "
                         f"(near ties go to the lowest typed p50)") if winner else \
            "no winner: no finalist passed the faithfulness gate"
    run["finished"] = datetime.now().isoformat(timespec="seconds")
    save_run(run, path)
    return run, path


def print_report(run: dict, path: Path):
    """Prints one line per configuration with its scores, timings and failed gates."""
    print(f"\nSystem sweep · {run['started']} · commit {run['git_commit'][:8]}{' (dirty)' if run['dirty'] else ''}\n")
    fmt = lambda v: "-" if v is None else (f"{v:.0f}" if isinstance(v, (int, float)) else str(v))
    print(f"{'config':<8} {'tables':>6} {'figures':>7} {'answered':>9} {'refused':>8} {'faith':>6} {'ingest s':>9} "
          f"{'extract p50':>11} {'typed p50':>9} {'spoken p50':>10}  gates")
    for cid, r in run["configs"].items():
        if r.get("status") != "ran":
            print(f"{cid:<8} {r.get('status')}: {r.get('reason')}")
            continue
        g, rag = r["golden"], r.get("ragas")
        faith = rag["faithfulness"] if isinstance(rag, dict) else rag
        faith = f"{faith:.3f}" if isinstance(faith, float) else faith
        print(f"{cid:<8} {right(r, 'tables'):>6} {right(r, 'figures'):>7} {g['answered']:>4}/{g['in_scope']:<4} "
              f"{g['refused']:>3}/{g['out_of_scope']:<4} {fmt(faith):>6} {r['ingestion']['wall_seconds']:>9.0f} "
              f"{fmt(r['ingestion']['seconds_per_figure']['extract_p50']):>11} {fmt(r['latency']['typed_p50_ms']):>9} "
              f"{fmt(r['latency']['spoken_p50_ms']):>10}  "
              f"{', '.join(k for k, v in r.get('gates', {}).items() if not v) or 'all pass'}")
    print(f"\nStage A winner: {run.get('stage_a_winner')} ({run.get('stage_a_reason')})")
    print(f"Outcome: {run.get('winner') or 'no winner'} ({run.get('reason')})")
    if run.get("leaked"):
        print(f"Leaked trick questions (answered, should be refused): {run['leaked']}")
    print(f"\nSaved {path.relative_to(REPO_ROOT)}")


def main(argv=None) -> int:
    """Command line entry: `python -m src.eval.system_sweep run [--stage A|B|all] [--apply] [--skip-ragas] [--resume
    FILE]`."""
    parser = argparse.ArgumentParser(description="Two stage configuration sweep (spec 0005)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run the sweep through the real API")
    run_parser.add_argument("--stage", choices=("A", "B", "all"), default="all")
    run_parser.add_argument("--skip-ragas", action="store_true")
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--apply", action="store_true", help="write the winner into config/config.yaml")
    args = parser.parse_args(argv)
    try:
        run, path = sweep(args.stage, args.skip_ragas, args.resume)
        print_report(run, path)
        if args.apply:
            if run.get("winner"):
                apply_winner(run["configs"][run["winner"]]["settings"])
                print(f"[sweep] Applied {run['winner']} to {CONFIG_PATH.relative_to(REPO_ROOT)}.")
            else:
                print("[sweep] No winner, so config/config.yaml is unchanged.")
    except BenchmarkError as e:
        print(f"[system_sweep] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
