"""Figure reader test set and benchmark.

    python -m src.eval.figure_benchmark draft    # list figure boxes, save crops, draft candidate questions
    python -m src.eval.figure_benchmark run      # every figure reader end to end through the real API

`draft` works in two passes so you confirm the figures before any question is drafted:
1. With no `figure_pages` listed, it lists every candidate the ingestion would find in the
   Census and UN reports (`bbox_source: "candidates"`) and saves each crop for you to look at.
   Delete the entries that are not real charts or diagrams, and add at most 4 Azure book pages
   by hand (an entry with only `file` and `page` gets its box from the candidates).
2. Run it again: the figure model drafts 2 to 3 questions per listed figure from its crop.
   A question is dropped when an expected value also appears in the page's words outside the
   figure, or is missing from the figure's own words when the figure has a text layer.
   New questions are `reviewed: false` until you check them.
"""
import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

from src.eval.table_benchmark import (BenchmarkError, CONFIG_PATH, OUT_DIR, REPO_ROOT, _centre_inside, check_pdfs,
                                      contains_value, corpus_dir, load_set, nearest_rank, pdf_path, read_table_log,
                                      require_model, save_set, serve_and_ask)

QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "figure_questions.json"
CROPS_DIR = REPO_ROOT / "data" / "eval" / "figure_crops"
DRAFT_MODEL = "qwen2.5vl:7b"
QUESTIONS_PER_FIGURE = 3
MIN_BOX_WORDS = 5              # a figure with fewer words than this has no text layer to check against
AUTO_LIST_DIR = "data/eval/figure_pdfs"   # pass 1 lists candidates only from the chart reports
ALL_READERS = ("none", "qwen2.5vl:3b", "qwen2.5vl:7b")
FIGURE_HEALTH_TIMEOUT_S = 3 * 60 * 60     # first ingestion with the 7B reads every candidate
MATCH_IOU = 0.3                           # a candidate this close to a listed figure box is that figure
NOTES = [
    "Each reader gets a fresh index of the figure set's documents and no caches (answer, detection, table and "
    "figure read caches are off); table_parser stays as configured.",
    "Seconds per figure come from the reader log, warm up excluded: classify is one call, extract sums every "
    "attempt for that figure (the retry included); figures that timed out or errored are counted, not timed.",
    "Recall: a listed figure is found when a candidate overlaps it with IoU at least 0.3; classified when that "
    "candidate was answered chart or diagram (any other kind, table included, is a miss).",
    "The figure questions were drafted by qwen2.5vl:7b from each crop and then reviewed by hand.",
]

DRAFT_PROMPT = """This image is one figure (a chart or a diagram) from a report.

Write {n} questions that can only be answered by reading this figure, each about a different value.
Each answer must be a short value copied exactly as printed in the figure (a number with its unit, a year, a label).
Mention what the figure shows in each question, so it makes sense without seeing the figure.

Reply only with JSON: {{"questions": [{{"question": "...", "expected": ["exact value"]}}]}}"""


def crop_name(filename: str, page: int, n: int) -> str:
    """The file name of a figure crop saved for review."""
    return f"{Path(filename).stem}-p{page}-{n}.png"


def page_candidates(data: dict, filename: str) -> dict:
    """{page: [candidate]} as the ingestion finds them (detector boxes exclude tables)."""
    from src.ingestion.figures import FigureExtractor
    found = {}
    for c in FigureExtractor(parser="qwen2.5vl:7b", use_cache=True).candidates(str(pdf_path(data, filename))):
        found.setdefault(c["page"], []).append(c)
    return found


def fill_boxes(data: dict) -> int:
    """Pass 1 lists every candidate in the chart reports; entries without a bbox get one per candidate."""
    files = ([d["file"] for d in data["documents"] if d.get("dir") == AUTO_LIST_DIR]
             if not data["figure_pages"] else sorted({e["file"] for e in data["figure_pages"] if not e.get("bbox")}))
    if not files:
        return 0
    found = {f: page_candidates(data, f) for f in files}
    if not data["figure_pages"]:
        entries = [{"file": f, "page": page, "bbox": c["box"], "bbox_source": "candidates", "caption": c["caption"]}
                   for f in files for page, cs in sorted(found[f].items()) for c in cs]
    else:
        entries = []
        for entry in data["figure_pages"]:
            if entry.get("bbox"):
                entries.append(entry)
                continue
            cs = found[entry["file"]].get(entry["page"], [])
            if not cs:
                print(f"[draft] No figure candidate on {entry['file']} page {entry['page']}; add its bbox by hand.")
                entries.append(entry)
            entries += [{"file": entry["file"], "page": entry["page"], "bbox": c["box"],
                         "bbox_source": "candidates", "caption": c["caption"]} for c in cs]
    filled = sum(1 for e in entries if e.get("bbox_source") == "candidates") - \
        sum(1 for e in data["figure_pages"] if e.get("bbox_source") == "candidates")
    data["figure_pages"] = entries
    return filled


def save_crops(data: dict) -> dict:
    """Saves each listed figure as a PNG for the reviewer; returns {(file, page, n): png bytes}."""
    import pdfplumber
    import pypdfium2 as pdfium
    from src.ingestion import figures
    from src.utils.config import get
    image_px = int(get("ingestion.figure_image_px", 800))
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    crops, counts = {}, {}
    for entry in data["figure_pages"]:
        if not entry.get("bbox"):
            continue
        key = (entry["file"], entry["page"])
        counts[key] = n = counts.get(key, 0) + 1
        path = pdf_path(data, entry["file"])
        pdf = pdfium.PdfDocument(str(path))
        try:
            with pdfplumber.open(path) as plumber:
                page = plumber.pages[entry["page"] - 1]
                x0, top, x1, bottom = entry["bbox"]
                box = figures.pad({"x0": x0, "top": top, "x1": x1, "bottom": bottom}, page.bbox)
                image = figures.crop(pdf[entry["page"] - 1], page.bbox, box, image_px)
        finally:
            pdf.close()
        image.save(CROPS_DIR / crop_name(entry["file"], entry["page"], n))
        crops[key + (n,)] = figures.png_bytes(image)
    return crops


def ask_model(png: bytes) -> list:
    """Candidate questions drafted by the vision model from one figure crop (for a human to review)."""
    import ollama
    response = ollama.chat(model=DRAFT_MODEL,
                           messages=[{"role": "user", "content": DRAFT_PROMPT.format(n=QUESTIONS_PER_FIGURE),
                                      "images": [png]}],
                           format="json", options={"temperature": 0, "num_predict": 400}, keep_alive="10m")
    try:
        return json.loads(response.message.content).get("questions", [])
    except (ValueError, AttributeError):
        return []


def words_split(plumber_page, box, boxes) -> tuple:
    """(words inside this figure, words outside every listed figure on the page)."""
    words = plumber_page.extract_words()
    inside = [w["text"] for w in words if _centre_inside(w, box)]
    outside = [w["text"] for w in words if not any(_centre_inside(w, b) for b in boxes)]
    return inside, " ".join(outside)


def candidate_problem(expected: list, inside: list, outside: str) -> str:
    """Why a drafted question cannot be used, or "" when it can (the absent outside check)."""
    if not expected or not all(isinstance(v, str) and v.strip() for v in expected):
        return "no expected value"
    for value in expected:
        if contains_value(outside, value):
            return f"{value!r} also appears outside the figure"
        if len(inside) >= MIN_BOX_WORDS and not contains_value(" ".join(inside), value):
            return f"{value!r} is not among the figure's words"
    return ""


def draft(data: dict) -> dict:
    """Lists figure boxes and saves crops on the first pass; drafts unreviewed candidate questions per listed figure
    on the second."""
    import pdfplumber
    check_pdfs(data)
    require_model(DRAFT_MODEL)
    first_pass = not data["figure_pages"]
    filled = fill_boxes(data)
    crops = save_crops(data)
    print(f"[draft] {filled} figure boxes filled from the candidates; crops in "
          f"{CROPS_DIR.relative_to(REPO_ROOT)} ({len(crops)} files).")
    if first_pass:
        print("[draft] Delete the figure_pages entries that are not real charts or diagrams, add at most 4 Azure "
              "book pages, then run draft again to draft questions.")
        return data

    existing = {q["question"].strip().lower() for q in data["questions"]}
    drafted = {(q["file"], q["page"]) for q in data["questions"]}
    next_id = 1 + max((int(q["id"][1:]) for q in data["questions"]), default=0)
    kept = dropped = 0
    by_page = {}
    for entry in data["figure_pages"]:
        if entry.get("bbox"):
            by_page.setdefault((entry["file"], entry["page"]), []).append(entry["bbox"])
    for (filename, page), boxes in by_page.items():
        if (filename, page) in drafted:
            continue                      # already drafted; delete its questions to draft again
        with pdfplumber.open(pdf_path(data, filename)) as plumber:
            plumber_page = plumber.pages[page - 1]
            for n, box in enumerate(boxes, start=1):
                inside, outside = words_split(plumber_page, box, boxes)
                for candidate in ask_model(crops[(filename, page, n)]):
                    question = str(candidate.get("question", "")).strip()
                    expected = candidate.get("expected", [])
                    expected = [expected] if isinstance(expected, str) else expected
                    expected = [str(v).strip() for v in expected] if isinstance(expected, list) else []
                    problem = candidate_problem(expected, inside, outside)
                    if not question or problem or question.lower() in existing:
                        dropped += 1
                        print(f"[draft] dropped ({problem or 'empty or duplicate'}): {filename} p.{page}: {question}")
                        continue
                    existing.add(question.lower())
                    data["questions"].append({"id": f"f{next_id:03d}", "question": question, "expected": expected,
                                              "file": filename, "page": page, "bbox": box,
                                              "crop": crop_name(filename, page, n), "reviewed": False})
                    next_id += 1
                    kept += 1
    print(f"[draft] {kept} candidates added (reviewed: false), {dropped} dropped.")
    return data


# --- run ---

def _iou(a, b) -> float:
    """Intersection over union of two boxes given as [x0, top, x1, bottom]."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def figure_calls_by_figure(lines: list) -> dict:
    """{(file, page, box): [call lines in order]} from the reader log's figure call lines."""
    groups = {}
    for line in lines:
        if str(line.get("reader", "")).startswith("figure:"):
            groups.setdefault((line["file"], line["page"], tuple(line["box"])), []).append(line)
    return groups


def figure_timings(lines: list) -> dict:
    """Seconds per figure and outcome counts."""
    classify, extract = [], []
    counts = {"retries": 0, "failed": 0, "timeouts": 0, "errors": 0, "missed_tables": 0, "classified_figures": 0}
    for calls in figure_calls_by_figure(lines).values():
        final = calls[-1]["outcome"]
        for c in calls:
            counts["retries"] += c["outcome"] == "retry"
            counts["timeouts"] += c["outcome"] == "timeout"
            counts["errors"] += c["outcome"] == "error"
            if c["stage"] == "classify":
                counts["missed_tables"] += c.get("kind") == "table"
                counts["classified_figures"] += c.get("kind") in ("chart", "diagram")
        counts["failed"] += calls[-1]["stage"] == "extract" and final == "failed"
        if final not in ("ok", "empty", "failed"):
            continue                      # timed out or errored: counted above, not timed
        first = [c for c in calls if c["stage"] == "classify"]
        if first and not first[0].get("warmup"):
            classify.append(first[0]["seconds"])
        tries = [c for c in calls if c["stage"] == "extract"]
        if tries and not any(c.get("warmup") for c in tries):
            extract.append(round(sum(c["seconds"] for c in tries), 3))
    stat = lambda v: {"p50_s": nearest_rank(v, 0.5), "max_s": max(v) if v else None, "timed": len(v)}
    return {"classify": stat(classify), "extract": stat(extract), **counts}


def figure_recall(lines: list, data: dict) -> dict:
    """Listed figures found among the candidates, and classified chart or diagram."""
    candidates = [(line["file"], c["page"], c["box"]) for line in lines if "figure_candidates" in line
                  for c in line["figure_candidates"]]
    kinds = {key: next((c.get("kind") for c in calls if c["stage"] == "classify"), None)
             for key, calls in figure_calls_by_figure(lines).items()}
    listed = [e for e in data["figure_pages"] if e.get("bbox")]
    found = classified = 0
    missed = []
    for e in listed:
        hit = any(f == e["file"] and p == e["page"] and _iou(box, e["bbox"]) >= MATCH_IOU for f, p, box in candidates)
        found += hit
        kind = next((k for (f, p, box), k in kinds.items()
                     if f == e["file"] and p == e["page"] and _iou(box, e["bbox"]) >= MATCH_IOU), None)
        classified += kind in ("chart", "diagram")
        if not hit or kind not in ("chart", "diagram"):
            missed.append({"file": e["file"], "page": e["page"], "found": hit, "kind": kind})
    return {"listed": len(listed), "found": found, "recall": round(found / len(listed), 3) if listed else None,
            "classified_chart_or_diagram": classified, "missed": missed}


def figure_config(data: dict, reader: str, workdir: Path) -> Path:
    """A throwaway config: the figure set's documents as the corpus, a fresh index, no caches, this reader."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"]["raw_data_dir"] = str(corpus_dir([data], workdir / "corpus"))
    config["paths"]["qdrant_storage"] = str(workdir / "index" / "qdrant_storage")
    config["paths"]["tantivy_index"] = str(workdir / "index" / "tantivy_index")
    config["cache"]["enabled"] = False
    config["ingestion"]["figure_parser"] = reader
    config["ingestion"]["table_cache"] = False
    config.setdefault("grounding", {})["checker"] = "none"     # measurements run unchecked
    path = workdir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


def run_reader(data: dict, reader: str, questions: list) -> dict:
    """One figure reader end to end: a fresh index, an API, every reviewed question asked and scored."""
    with tempfile.TemporaryDirectory(prefix=f"figures-{reader.replace(':', '-')}-") as tmp:
        workdir = Path(tmp)
        result = serve_and_ask(figure_config(data, reader, workdir), workdir, {"figure_parser_effective": reader},
                               questions, reader, FIGURE_HEALTH_TIMEOUT_S)
        if result["status"] != "ran":
            return result
        _, _, lines = read_table_log(result.pop("table_log"))
        out = {**result, "timing": figure_timings(lines)}
        if reader != "none":
            out["recall"] = figure_recall(lines, data)
        return out


def run(data: dict, readers: list) -> dict:
    """Runs every reader and picks the winner; returns the report."""
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
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    return report


def print_report(report: dict, path: Path):
    """Prints one line per reader with its scores, timings and figure recall."""
    print(f"\nFigure reader benchmark · {report['questions']} reviewed questions\n")
    fmt = lambda v: "-" if v is None else f"{v:.1f}"
    print(f"{'reader':<14} {'value+page':>10} {'value':>6} {'page':>5} {'refused':>8} {'cls p50':>8} "
          f"{'ext p50':>8} {'ext max':>8} {'recall':>7} {'invalid':>8} {'ingest s':>9}")
    for name, r in report["readers"].items():
        if r["status"] != "ran":
            print(f"{name:<14} skipped: {r['reason']}")
            continue
        s, t, rc = r["scores"], r["timing"], r.get("recall") or {}
        print(f"{name:<14} {s['right_value_and_page']:>10} {s['right_value_only']:>6} {s['right_page_only']:>5} "
              f"{s['refusals']:>8} {fmt(t['classify']['p50_s']):>8} {fmt(t['extract']['p50_s']):>8} "
              f"{fmt(t['extract']['max_s']):>8} {fmt(rc.get('recall')):>7} {t['failed']:>8} {r['ingest_seconds']:>9.1f}")
        if rc:
            print(f"  {rc['found']}/{rc['listed']} listed figures found, {rc['classified_chart_or_diagram']} classified "
                  f"chart or diagram; {t['retries']} retries, {t['timeouts']} timeouts, {t['errors']} errors, "
                  f"{t['missed_tables']} tables the detector missed")
    for note in report["notes"]:
        print(f"Note: {note}")
    print(f"\nSaved {path.relative_to(REPO_ROOT)}")


def main(argv=None) -> int:
    """Command line entry: `python -m src.eval.figure_benchmark draft | run [--readers ...] [--apply]`."""
    parser = argparse.ArgumentParser(description="Figure reader benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draft", help="list figure boxes, save crops and draft candidate questions for review")
    run_parser = sub.add_parser("run", help="benchmark every figure reader end to end through the real API")
    run_parser.add_argument("--readers", nargs="+", choices=ALL_READERS, default=list(ALL_READERS))
    args = parser.parse_args(argv)
    try:
        if args.command == "draft":
            save_set(draft(load_set(QUESTIONS_PATH)), QUESTIONS_PATH)
            print(f"[draft] Wrote {QUESTIONS_PATH.relative_to(REPO_ROOT)}. Review each candidate, then set reviewed: true.")
        elif args.command == "run":
            report = run(load_set(QUESTIONS_PATH), args.readers)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / f"figures-{datetime.now():%Y%m%d-%H%M%S}.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print_report(report, path)
    except BenchmarkError as e:
        print(f"[figure_benchmark] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
