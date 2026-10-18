"""Table reader benchmark.

    python -m src.eval.table_benchmark draft    # fill table boxes, draft candidate questions for review
    python -m src.eval.table_benchmark run [--readers none pdfplumber ...]

`draft` reads data/eval/table_questions.json, fills a missing table box from the detector (you
confirm it), and asks llama3.1:8b, from each table page's plain text, for questions whose answers
sit only in the table. Candidates land with `reviewed: false`; only reviewed ones are ever scored.

`run` starts the real API once per reader on a throwaway index built from the test PDFs, asks every
reviewed question, scores right value and right page, and applies the winner rule.
"""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "table_questions.json"
DRAFT_MODEL = "llama3.1:8b"          # the Ragas judge model, already pulled for the quality gate
QUESTIONS_PER_TABLE = 3

OUT_DIR = REPO_ROOT / "data" / "benchmarks"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
ALL_READERS = ("none", "pdfplumber", "qwen2.5vl:3b", "qwen2.5vl:7b", "layoutlmv3")
HEALTH_TIMEOUT_S = 60 * 60       # first ingestion with a VLM reader can take a while
QUERY_TIMEOUT_S = 300
NEAR_TIE = 2                     # readers within this many answers of the top score tie
MIN_GAIN_OVER_NONE = 3           # the winner must beat `none` by at least this many answers
NOTES = [
    "The 0.99 detector threshold was tuned on the Azure book's table pages, so its recall there is "
    "circular; recall on the second PDF is the honest check.",
    "Each reader gets a fresh index and no caches (answer cache, read cache and detection cache are off).",
    "Ingestion seconds per table page come from the reader log (ok and empty calls only, warm up excluded); "
    "total ingestion seconds run from API start to the first healthy /api/health.",
]

DRAFT_PROMPT = """Below is the plain text of one PDF page. Lines between <table> and </table> come from one table on the page.

Write {n} questions that can only be answered by reading that table, each about a different cell.
Each answer must be a short value copied exactly from one table cell (a number with its unit, a name, a version).
Do not ask about the table's title or about text outside the table.

Reply only with JSON: {{"questions": [{{"question": "...", "expected": ["exact cell value"]}}]}}

<page>
{page_text}
</page>

<table>
{table_text}
</table>"""


class BenchmarkError(Exception):
    """A benchmark cannot run (missing PDFs, model or reviewed questions); the message says what to fix."""
    pass


# --- value matching (shared with scoring) ---

_TOKEN = re.compile(r"[^\W_]+(?:\.[^\W_]+)*|[^\w\s]")


def value_tokens(text: str) -> list:
    """Lowercased tokens, thousands commas removed: "1,000 GB" -> ["1000", "gb"], "GPCLK0" -> ["gpclk0"]."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", (text or "").lower())
    return _TOKEN.findall(text)


def contains_value(text: str, value: str) -> bool:
    """True when value appears in text as a whole token sequence, whitespace ignored.

    "195 GB" matches "195GB", but "5" never matches "15" and "GPCLK" never matches "GPCLK0".
    """
    target = "".join(value_tokens(value))
    hay = value_tokens(text)
    if not target:
        return False
    for i in range(len(hay)):
        joined = ""
        for token in hay[i:]:
            joined += token
            if joined == target:
                return True
            if not target.startswith(joined):
                break
    return False


# --- the test set file ---

def load_set(path: Path = QUESTIONS_PATH) -> dict:
    """A question set file as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_set(data: dict, path: Path = QUESTIONS_PATH):
    """Writes a question set file, keeping non ASCII text readable."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def pdf_path(data: dict, filename: str) -> Path:
    """A document's own folder when it names one (figure_questions.json), else the set's pdf_dir."""
    doc = next((d for d in data["documents"] if d["file"] == filename), {})
    return REPO_ROOT / doc.get("dir", data.get("pdf_dir", "")) / filename


def check_pdfs(data: dict):
    """Every listed PDF exists and matches its SHA256; otherwise name the file and where to get it."""
    for doc in data["documents"]:
        path = pdf_path(data, doc["file"])
        if not path.exists():
            raise BenchmarkError(f"Missing {path}. Download it from {doc['url']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != doc["sha256"]:
            raise BenchmarkError(f"{doc['file']} has SHA256 {digest[:12]}…, expected {doc['sha256'][:12]}…. "
                                 f"Download the original from {doc['url']}")


def corpus_dir(sets: list, folder: Path) -> Path:
    """Links every PDF listed in the given question sets into `folder` (one file per name; same name must mean the
    same SHA256) and returns it, so one API run can index the documents of several sets."""
    folder.mkdir(parents=True, exist_ok=True)
    seen = {}
    for data in sets:
        for doc in data["documents"]:
            if seen.setdefault(doc["file"], doc["sha256"]) != doc["sha256"]:
                raise BenchmarkError(f"{doc['file']} is listed with two different SHA256 values.")
            link = folder / doc["file"]
            if not link.exists():
                link.symlink_to(pdf_path(data, doc["file"]))
    return folder


# --- page words ---

def _centre_inside(word: dict, box) -> bool:
    """True when the word's centre lies inside the box."""
    x, y = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def page_text_split(plumber_page, boxes) -> tuple:
    """(text inside any of the boxes, text outside all of them), from the page's words."""
    words = plumber_page.extract_words()
    inside = [w["text"] for w in words if any(_centre_inside(w, b) for b in boxes)]
    outside = [w["text"] for w in words if not any(_centre_inside(w, b) for b in boxes)]
    return " ".join(inside), " ".join(outside)


def box_text(plumber_page, box) -> str:
    """The table's words, one line per text line, for the drafting prompt."""
    region = plumber_page.crop((max(box[0], plumber_page.bbox[0]), max(box[1], plumber_page.bbox[1]),
                                min(box[2], plumber_page.bbox[2]), min(box[3], plumber_page.bbox[3])))
    return region.extract_text() or ""


# --- draft ---

def fill_boxes(data: dict) -> int:
    """Entries without a bbox get one per detected table (bbox_source: detector) for you to confirm."""
    import pdfplumber
    import pypdfium2 as pdfium
    from src.ingestion.tables import get_detector

    filled, entries = 0, []
    for entry in data["table_pages"]:
        if entry.get("bbox"):
            entries.append(entry)
            continue
        path = pdf_path(data, entry["file"])
        pdf = pdfium.PdfDocument(str(path))
        try:
            with pdfplumber.open(path) as plumber:
                boxes = get_detector().detect(pdf[entry["page"] - 1], plumber.pages[entry["page"] - 1])
        finally:
            pdf.close()
        if not boxes:
            print(f"[draft] No table detected on {entry['file']} page {entry['page']}; add its bbox by hand.")
            entries.append(entry)
            continue
        for box in boxes:
            entries.append({"file": entry["file"], "page": entry["page"],
                            "bbox": [round(box[k], 1) for k in ("x0", "top", "x1", "bottom")],
                            "bbox_source": "detector"})
            filled += 1
    data["table_pages"] = entries
    return filled


def require_model(model: str):
    """Raises BenchmarkError unless the Ollama model is pulled."""
    import ollama
    try:
        names = {m.model for m in ollama.list().models}
    except Exception as e:
        raise BenchmarkError(f"Ollama is not reachable ({e}). Start it with `ollama serve`.")
    if model not in names and f"{model}:latest" not in names:
        raise BenchmarkError(f"Ollama model {model} is not pulled. Run `ollama pull {model}`.")


def ask_model(page_text: str, table_text: str) -> list:
    """Candidate questions drafted by the model from one table's text (for a human to review)."""
    import ollama
    response = ollama.chat(
        model=DRAFT_MODEL,
        messages=[{"role": "user", "content": DRAFT_PROMPT.format(
            n=QUESTIONS_PER_TABLE, page_text=page_text, table_text=table_text)}],
        format="json",
        options={"temperature": 0},
    )
    try:
        return json.loads(response.message.content).get("questions", [])
    except (ValueError, AttributeError):
        return []


def candidate_problem(expected: list, inside: str, outside: str) -> str:
    """Why a drafted question cannot be used, or "" when it can."""
    if not expected or not all(isinstance(v, str) and v.strip() for v in expected):
        return "no expected value"
    for value in expected:
        if not contains_value(inside, value):
            return f"{value!r} is not in the table"
        if contains_value(outside, value):
            return f"{value!r} also appears outside the table"
    return ""


def draft(data: dict) -> dict:
    """Fills missing table boxes from the detector and adds drafted, unreviewed candidate questions to the set."""
    import pdfplumber

    check_pdfs(data)
    require_model(DRAFT_MODEL)
    filled = fill_boxes(data)
    print(f"[draft] Filled {filled} table boxes from the detector (confirm them before scoring).")

    existing = {q["question"].strip().lower() for q in data["questions"]}
    next_id = 1 + max((int(q["id"][1:]) for q in data["questions"]), default=0)
    kept = dropped = 0
    by_page = {}
    for entry in data["table_pages"]:
        if entry.get("bbox"):
            by_page.setdefault((entry["file"], entry["page"]), []).append(entry["bbox"])

    for (filename, page), boxes in by_page.items():
        with pdfplumber.open(pdf_path(data, filename)) as plumber:
            plumber_page = plumber.pages[page - 1]
            inside, outside = page_text_split(plumber_page, boxes)
            page_text = plumber_page.extract_text() or ""
            for box in boxes:
                for candidate in ask_model(page_text, box_text(plumber_page, box)):
                    question = str(candidate.get("question", "")).strip()
                    expected = candidate.get("expected", [])
                    expected = [expected] if isinstance(expected, str) else expected
                    problem = candidate_problem(expected, inside, outside)
                    if not question or problem or question.lower() in existing:
                        dropped += 1
                        print(f"[draft] dropped ({problem or 'empty or duplicate'}): {filename} p.{page}: {question}")
                        continue
                    existing.add(question.lower())
                    data["questions"].append({"id": f"t{next_id:03d}", "question": question,
                                              "expected": [str(v).strip() for v in expected],
                                              "file": filename, "page": page, "reviewed": False})
                    next_id += 1
                    kept += 1
    print(f"[draft] {kept} candidates added (reviewed: false), {dropped} dropped.")
    return data


# --- run ---

def nearest_rank(values, q):
    """Value at rank ceil(q * n) of the sorted list; None for an empty list."""
    import math
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def score_answer(question: dict, body: dict) -> dict:
    """Right value, right page and refusal for one API answer."""
    from src.utils.text_sanitizer import is_refusal, strip_citation_headers
    answer = body.get("answer") or ""
    citations = body.get("citations") or []
    clean = strip_citation_headers(answer)          # a page number in a tag never counts as a value
    right_value = all(contains_value(clean, v) for v in question["expected"])
    right_page = any(c.get("source") == question["file"] and c.get("page") == question["page"]
                     for c in citations)
    refused = not citations and is_refusal(answer)
    return {"right_value": right_value, "right_page": right_page, "refused": refused}


def summarize(results: list) -> dict:
    """Counts of right value and page, value only, page only, refused and wrong answers."""
    return {
        "right_value_and_page": sum(r["right_value"] and r["right_page"] for r in results),
        "right_value_only": sum(r["right_value"] and not r["right_page"] for r in results),
        "right_page_only": sum(r["right_page"] and not r["right_value"] for r in results),
        "refusals": sum(r["refused"] for r in results),
        "errors": sum(bool(r.get("error")) for r in results),
        "questions": len(results),
    }


def read_table_log(path: Path) -> tuple:
    """(table reader call lines, detection lines, figure lines) from a VRAG_TABLE_LOG file.

    Figure lines are figure calls (`reader` starts with "figure:"), per PDF candidate lines and
    dense page skips."""
    calls, detections, figures = [], [], []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "detected_pages" in record:
                detections.append(record)
            elif ("figure_candidates" in record or "figure_skip" in record
                  or str(record.get("reader", "")).startswith("figure:")):
                figures.append(record)
            else:
                calls.append(record)
    return calls, detections, figures


def reader_timings(calls: list) -> dict:
    """Per call reader seconds (warm up calls left out) and how many calls ended in each outcome."""
    timed = [c["seconds"] for c in calls if c["outcome"] in ("ok", "empty") and not c.get("warmup")]
    outcomes = {}
    for c in calls:
        outcomes[c["outcome"]] = outcomes.get(c["outcome"], 0) + 1
    return {"p50_s": nearest_rank(timed, 0.5), "max_s": max(timed) if timed else None,
            "timed_calls": len(timed), "outcomes": outcomes}


def detector_report(detections: list, data: dict) -> dict:
    """How many listed table pages the detector found, and any extra pages it flagged."""
    listed = {(e["file"], e["page"]) for e in data["table_pages"]}
    found = {(d["file"], p) for d in detections for p in d["detected_pages"]}
    return {
        "table_pages_detected": len(found),
        "listed_table_pages": len(listed),
        "recall": round(len(listed & found) / len(listed), 3) if listed else None,
        "missed": sorted([f, p] for f, p in listed - found),
        "false_hits": sorted([f, p] for f, p in found - listed),
        "detect_failed_pages": sum(len(d["detect_failed_pages"]) for d in detections),
        "detect_seconds": round(sum(d["detect_seconds"] for d in detections), 1),
    }


def pick_winner(readers: dict) -> dict:
    """Most right value and page; near ties go to the lowest p50; `none` must be beaten by 3."""
    scored = {name: r for name, r in readers.items() if r.get("status") == "ran"}
    baseline = scored.get("none")
    contenders = {n: r for n, r in scored.items() if n != "none"}
    if not contenders:
        return {"outcome": "none", "reason": "no reader other than none ran"}
    top = max(r["scores"]["right_value_and_page"] for r in contenders.values())
    tied = [n for n, r in contenders.items() if r["scores"]["right_value_and_page"] >= top - NEAR_TIE]
    p50 = lambda n: contenders[n]["timing"]["p50_s"]
    winner = min(tied, key=lambda n: (p50(n) is None, p50(n) if p50(n) is not None else 0.0,
                                      -contenders[n]["scores"]["right_value_and_page"]))
    score = contenders[winner]["scores"]["right_value_and_page"]
    if baseline is None:
        return {"outcome": "undecided", "candidate": winner, "near_ties": tied,
                "reason": "the none baseline did not run, so the 3 answer rule cannot be applied"}
    gain = score - baseline["scores"]["right_value_and_page"]
    if gain < MIN_GAIN_OVER_NONE:
        return {"outcome": "none", "candidate": winner, "near_ties": tied, "gain_over_none": gain,
                "reason": f"{winner} beats none by {gain} answer{'' if gain == 1 else 's'}, fewer than {MIN_GAIN_OVER_NONE}"}
    return {"outcome": winner, "near_ties": tied, "gain_over_none": gain,
            "reason": f"{winner} scores {score}, {gain} above none"
                      + (f"; near tie with {', '.join(t for t in tied if t != winner)} broken by p50 seconds"
                         if len(tied) > 1 else "")}


def free_port() -> int:
    """A free local TCP port for a benchmark API."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def reader_config(data: dict, reader: str, workdir: Path) -> Path:
    """A throwaway config: the test PDFs as the corpus, a fresh index, no caches, this reader."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"]["raw_data_dir"] = str(REPO_ROOT / data["pdf_dir"])
    config["paths"]["qdrant_storage"] = str(workdir / "index" / "qdrant_storage")
    config["paths"]["tantivy_index"] = str(workdir / "index" / "tantivy_index")
    config["cache"]["enabled"] = False
    config["ingestion"]["table_parser"] = reader
    config["ingestion"]["table_cache"] = False
    config["ingestion"]["figure_parser"] = "none"     # tables only, whatever the figure default becomes
    config.setdefault("grounding", {})["checker"] = "none"     # measurements run unchecked
    path = workdir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


def ask_all(base: str, questions: list, label: str) -> list:
    """Asks every question with an empty history and scores each answer (right value, right page, refusal)."""
    results = []
    for i, q in enumerate(questions, 1):
        try:
            r = requests.post(f"{base}/query", json={"query": q["question"], "history": []}, timeout=QUERY_TIMEOUT_S)
            r.raise_for_status()
            body = r.json()
            results.append({"id": q["id"], **score_answer(q, body), "answer": body.get("answer", ""),
                            "cited": [[c.get("source"), c.get("page")] for c in body.get("citations") or []]})
        except (requests.RequestException, ValueError) as e:
            results.append({"id": q["id"], "right_value": False, "right_page": False, "refused": False,
                            "error": str(e)})
        print(f"  [{label}] {i}/{len(questions)} {q['id']}: "
              f"{'value' if results[-1]['right_value'] else '-'} {'page' if results[-1]['right_page'] else '-'}")
    return results


def serve_and_ask(config_path: Path, workdir: Path, expect: dict, questions: list, label: str,
                  health_timeout: float = HEALTH_TIMEOUT_S) -> dict:
    """Starts the API on `config_path`, checks /api/health against `expect`, asks every question, stops the API.

    Shared by the table and figure benchmarks. The reader log lands in workdir/tables.jsonl."""
    log_path, table_log = workdir / "server.log", workdir / "tables.jsonl"
    port = free_port()
    env = dict(os.environ, VRAG_CONFIG=str(config_path), VRAG_TABLE_LOG=str(table_log))
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                 "--port", str(port)], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}/api"
    try:
        health = None
        while time.perf_counter() - start < health_timeout:
            if proc.poll() is not None:
                break
            try:
                health = requests.get(f"{base}/health", timeout=5).json()
                break
            except requests.RequestException:
                time.sleep(2)
        ingest_s = round(time.perf_counter() - start, 1)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:] if log_path.exists() else ""
        if health is None:
            return {"status": "skipped", "reason": "the API did not start", "log_tail": tail}
        for key, wanted in expect.items():
            if health.get(key) != wanted:
                return {"status": "skipped", "reason": f"{key} is {health.get(key)!r}, not {wanted!r}",
                        "log_tail": tail}

        results = ask_all(base, questions, label)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"status": "ran", "ingest_seconds": ingest_s, "scores": summarize(results), "results": results,
            "table_log": table_log}


def run_reader(data: dict, reader: str, questions: list) -> dict:
    """Starts the API for one reader, asks every question, stops the API."""
    with tempfile.TemporaryDirectory(prefix=f"tables-{reader.replace(':', '-')}-") as tmp:
        workdir = Path(tmp)
        result = serve_and_ask(reader_config(data, reader, workdir), workdir,
                               {"table_parser_effective": reader}, questions, reader)
        if result["status"] != "ran":
            return result
        calls, detections, _ = read_table_log(result.pop("table_log"))
        return {"status": "ran", "ingest_seconds": result["ingest_seconds"], "scores": result["scores"],
                "timing": reader_timings(calls), "detections": detections, "results": result["results"]}


def run(data: dict, readers: list) -> dict:
    """Asks the reviewed questions through an API per reader and scores each; returns the report."""
    check_pdfs(data)
    questions = [q for q in data["questions"] if q.get("reviewed") is True]
    if not questions:
        raise BenchmarkError("No reviewed questions in the test set; review candidates and set reviewed: true.")
    report = {"started": datetime.now().isoformat(timespec="seconds"), "questions": len(questions),
              "readers": {}, "notes": NOTES}
    for reader in readers:
        print(f"[run] {reader}: starting the API on a fresh index…")
        report["readers"][reader] = result = run_reader(data, reader, questions)
        if result["status"] == "ran":
            s = result["scores"]
            print(f"[run] {reader}: {s['right_value_and_page']}/{s['questions']} right value and page, "
                  f"ingestion {result['ingest_seconds']} s")
        else:
            print(f"[run] {reader}: skipped ({result['reason']})")
    first = next((report["readers"][r] for r in readers
                  if r != "none" and report["readers"][r]["status"] == "ran"), None)
    report["detector"] = detector_report(first["detections"], data) if first else None
    report["winner"] = pick_winner(report["readers"])
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    return report


def print_report(report: dict, path: Path):
    """Prints one line per reader with its scores and timings."""
    print(f"\nTable reader benchmark · {report['questions']} reviewed questions\n")
    print(f"{'reader':<14} {'value+page':>10} {'value':>6} {'page':>5} {'refused':>8} {'p50 s':>7} {'max s':>7} {'ingest s':>9}")
    for name, r in report["readers"].items():
        if r["status"] != "ran":
            print(f"{name:<14} skipped: {r['reason']}")
            continue
        s, t = r["scores"], r["timing"]
        fmt = lambda v: "-" if v is None else f"{v:.1f}"
        print(f"{name:<14} {s['right_value_and_page']:>10} {s['right_value_only']:>6} {s['right_page_only']:>5} "
              f"{s['refusals']:>8} {fmt(t['p50_s']):>7} {fmt(t['max_s']):>7} {r['ingest_seconds']:>9.1f}")
    d = report["detector"]
    if d:
        print(f"\nDetector: {d['table_pages_detected']} table pages, recall {d['recall']} on "
              f"{d['listed_table_pages']} listed pages, {len(d['false_hits'])} false hits, {d['detect_seconds']} s")
        if d["missed"]:
            print(f"  missed: {d['missed']}")
    w = report["winner"]
    print(f"\nOutcome: {w['outcome']} ({w['reason']})")
    for note in report["notes"]:
        print(f"Note: {note}")
    print(f"\nSaved {path.relative_to(REPO_ROOT)}")


def main(argv=None) -> int:
    """Command line entry: `python -m src.eval.table_benchmark draft | run [--readers ...] [--apply]`."""
    parser = argparse.ArgumentParser(description="Table reader benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draft", help="fill table boxes and draft candidate questions for review")
    run_parser = sub.add_parser("run", help="benchmark every reader end to end through the real API")
    run_parser.add_argument("--readers", nargs="+", choices=ALL_READERS, default=list(ALL_READERS))
    args = parser.parse_args(argv)
    try:
        if args.command == "draft":
            save_set(draft(load_set()))
            print(f"[draft] Wrote {QUESTIONS_PATH.relative_to(REPO_ROOT)}. Review each candidate, then set reviewed: true.")
        elif args.command == "run":
            report = run(load_set(), args.readers)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / f"tables-{datetime.now():%Y%m%d-%H%M%S}.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print_report(report, path)
    except BenchmarkError as e:
        print(f"[table_benchmark] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
