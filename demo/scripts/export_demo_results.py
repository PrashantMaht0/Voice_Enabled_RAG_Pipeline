"""Exports the measured results behind the README's "Choosing the setup by measurement" table into
demo/public/results.json for the live demo's Results tab (spec 0009, AC-12).

The numbers come from your local sweep (data/benchmarks is not in git, so run this on your machine and commit the
output). They describe your local corpus and setup, which the tab says plainly.

Usage, from the repo root: ./dev-env/bin/python demo/scripts/export_demo_results.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, "data", "benchmarks", "sweep-20261009-190140.json")
OUT = os.path.join(ROOT, "demo", "public", "results.json")
# The README rows, in its order, with its labels.
ROWS = [("A0", "No figure reading"), ("A3", "Qwen2.5 VL 3B reads figures"),
        ("A7", "Qwen2.5 VL 7B reads figures (chosen)"), ("B-qwen", "3B figures, Qwen 3B also answers")]
CORPUS_NOTE = ("Measured on your local corpus (the Azure book, the Raspberry Pi 4 datasheet and two reports) with the "
               "local setup: Ollama models and the judge grounding checker. The demo's sample documents are different.")


def seconds(ms):
    return None if ms is None else round(ms / 1000, 1)


def row(key: str, label: str, config: dict, run_date: str) -> dict:
    tables, figures, golden = config["tables"], config["figures"], config["golden"]
    latency, ragas = config.get("latency") or {}, config.get("ragas") or {}
    return {"setup": label, "key": key, "run_date": run_date,
            "tables_right": tables["right_value_and_page"], "tables_total": tables["questions"],
            "figures_right": figures["right_value_and_page"], "figures_total": figures["questions"],
            "answered": golden["answered"], "answered_total": golden["in_scope"],
            "refused": golden["refused"], "refused_total": golden["out_of_scope"],
            "faithfulness": ragas.get("faithfulness"), "checker": "judge:llama3.2:1b",
            "typed_p50_s": seconds(latency.get("typed_p50_ms")), "typed_p95_s": seconds(latency.get("typed_p95_ms")),
            "spoken_p50_s": seconds(latency.get("spoken_p50_ms"))}


def main():
    with open(SWEEP, encoding="utf-8") as f:
        sweep = json.load(f)
    run_date = sweep["started"][:10]
    rows = [row(key, label, sweep["configs"][key], run_date) for key, label in ROWS
            if sweep["configs"].get(key, {}).get("status") == "ran"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"generated_at": sweep["updated"], "corpus_note": CORPUS_NOTE, "rows": rows}, f, indent=1)
    print(f"Wrote {len(rows)} rows to {os.path.relpath(OUT, ROOT)}")
    for r in rows:
        print(f"  {r['setup']}: tables {r['tables_right']}/{r['tables_total']}, figures {r['figures_right']}/"
              f"{r['figures_total']}, answered {r['answered']}/{r['answered_total']}, refused {r['refused']}/"
              f"{r['refused_total']}, faithfulness {r['faithfulness']}, p50 {r['typed_p50_s']} s")


if __name__ == "__main__":
    main()
