"""Per stage latency benchmark over the golden questions (spec 0003).

Text goes through the running API; voice is timed in this process with the
dashboard's own speech classes. Run from the repo root, with the API up:

    python -m src.eval.latency_benchmark [--label NAME] [--limit N] [--skip-voice]
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from src.utils.config import get
from src.utils.text_sanitizer import is_refusal, sanitize_for_speech

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "data" / "eval" / "golden_dataset.json"
OUT_DIR = REPO_ROOT / "data" / "benchmarks"

MODEL_ERROR = "Error processing response"   # what llm_client yields when Ollama fails mid answer
VOICE_QUESTIONS = 10                         # text to speech costs about 9 s per answer
MAX_CONSECUTIVE_ERRORS = 3
SHORT_TIMEOUT, QUERY_TIMEOUT = 5, 120
PASSES = ("first", "repeat")
TEXT_GROUPS = ("answered", "refused", "cache_hit")
VOICE_METRICS = ("stt_ms", "query_ms", "clean_ms", "tts_ms", "total_ms")

NOTES = [
    "Every question is asked with empty history, so the follow-up rewriter (1B model) is never timed.",
    "Voice is timed in this process, so the dashboard's Gradio and audio encoding overhead is not counted.",
    "A query that times out keeps running on the server, so the next question may be slower.",
]


class ApiUnavailable(Exception):
    pass


class Aborted(Exception):
    pass


# --- statistics ---

def nearest_rank(values, q):
    """Value at rank ceil(q * n) of the sorted list; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def summarize(values):
    return {"n": len(values), "p50": nearest_rank(values, 0.50), "p95": nearest_rank(values, 0.95)}


# --- small helpers ---

def normalize_host(host: str) -> str:
    host = (host or "").strip().rstrip("/")
    return host if "://" in host else f"http://{host}"


def model_key(name: str) -> str:
    """Ollama names without a tag mean ':latest'."""
    return name if ":" in name else f"{name}:latest"


def words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def safe_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", (label or "run").lower()) or "run"


def fact(fn):
    """Machine facts are best effort: any failure is reported as unknown (null)."""
    try:
        return fn()
    except Exception:
        return None


def cpu_name():
    system = platform.system()
    if system == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if system == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def ram_gb():
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def default_speech():
    """The dashboard's own speech classes, imported only when voice runs."""
    from src.voice.stt import WhisperSTT
    from src.voice.tts import KokoroTTS
    return WhisperSTT(), KokoroTTS()


# --- the benchmark ---

class Benchmark:
    def __init__(self, api_base, ollama_host, session=None, limit=None, skip_voice=False,
                 label="run", golden_path=GOLDEN_PATH, out_dir=OUT_DIR, speech_factory=default_speech):
        self.api = api_base.rstrip("/")
        self.ollama_host = normalize_host(ollama_host)
        self.session = session or requests.Session()
        self.limit = limit
        self.skip_voice = skip_voice
        self.label = safe_label(label)
        self.golden_path = Path(golden_path)
        self.out_dir = Path(out_dir)
        self.speech_factory = speech_factory
        self.consecutive_errors = 0
        self.warnings = []

    # questions

    def load_questions(self):
        golden = json.loads(self.golden_path.read_text(encoding="utf-8"))
        items = [{"id": i, "question": g["question"], "in_scope": bool(g.get("in_scope", True))}
                 for i, g in enumerate(golden)]
        in_scope = [q for q in items if q["in_scope"]]
        out_scope = [q for q in items if not q["in_scope"]]
        if self.limit:
            in_scope, out_scope = in_scope[:self.limit], out_scope[:self.limit]
        # Keep golden set order across both kinds
        chosen = sorted(in_scope + out_scope, key=lambda q: q["id"])
        voice_n = min(VOICE_QUESTIONS, self.limit or VOICE_QUESTIONS)
        return chosen, in_scope[:voice_n]

    # HTTP

    def preflight(self):
        try:
            health = self.session.get(f"{self.api}/health", timeout=SHORT_TIMEOUT).json()
        except Exception as e:
            raise ApiUnavailable(
                f"The API at {self.api} is not reachable ({e}). Start it first:\n"
                "  ./dev-env/bin/python -m uvicorn main:app --port 8000") from e
        if health.get("ollama_status") != "online":
            raise ApiUnavailable(
                f"The API is up but reports Ollama {health.get('ollama_status')!r}. Start Ollama "
                f"(ollama serve) and pull {health.get('generation_model')}.")
        try:
            cache = self.session.get(f"{self.api}/cache", timeout=SHORT_TIMEOUT).json()
            if not cache.get("enabled", True):
                self.warnings.append("The answer cache is disabled, so the repeat pass shows no cache hits.")
        except Exception:
            self.warnings.append("Could not read /api/cache.")
        return health

    def clear_cache(self):
        self.session.delete(f"{self.api}/cache", timeout=SHORT_TIMEOUT)

    def ask(self, question):
        """(response json or None, client ms, error or None)."""
        started = time.perf_counter()
        try:
            response = self.session.post(f"{self.api}/query", json={"query": question, "history": []},
                                         timeout=QUERY_TIMEOUT)
            client_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return None, (time.perf_counter() - started) * 1000, str(e) or type(e).__name__
        if str(data.get("answer", "")).strip().startswith(MODEL_ERROR):
            return data, client_ms, data["answer"].strip()
        return data, client_ms, None

    def track(self, error):
        """Stops the run after too many failures in a row, so a dead API can't hang it for hours."""
        self.consecutive_errors = self.consecutive_errors + 1 if error else 0
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            raise Aborted(f"{MAX_CONSECUTIVE_ERRORS} errors in a row, last: {error}")

    # text

    @staticmethod
    def text_record(pass_name, item, data, client_ms, error):
        record = {"pass": pass_name, "id": item["id"], "in_scope": item["in_scope"],
                  "path": None, "refused": None, "gated": None, "cache": None,
                  "stages": {}, "untraced_ms": None, "server_total_ms": None,
                  "client_ms": client_ms, "error": error}
        if data is None or error:
            return record   # errors carry no path and never enter a percentile
        trace = data.get("trace", [])
        # Spans with 0 ms (cache, confidence) are labels, not timings
        stages = {s["layer"]: s["elapsed_ms"] for s in trace if s.get("elapsed_ms", 0) > 0}
        total = data.get("total_latency_ms")
        cache = data.get("cache")
        refused = is_refusal(data.get("answer", ""))
        record.update({
            "refused": refused,
            "gated": cache is None and not any(s["layer"] == "generation" for s in trace),
            "cache": cache,
            "path": "cache_hit" if cache else ("refused" if refused else "answered"),
            "stages": stages,
            "server_total_ms": total,
            "untraced_ms": None if total is None else total - sum(stages.values()),
        })
        return record

    def text_pass(self, pass_name, questions):
        records = []
        for item in questions:
            data, client_ms, error = self.ask(item["question"])
            records.append(self.text_record(pass_name, item, data, client_ms, error))
            print(f"  [{pass_name}] {item['id']:>2} {records[-1]['path'] or 'error':<9} {client_ms:8.0f} ms")
            self.track(error)
        return records

    # voice

    def voice_turns(self, questions, records):
        from src.utils.audio_utils import float32_to_int16, int16_to_float32, resample_audio

        stt, tts = self.speech_factory()
        if getattr(stt, "model", None) is None or getattr(tts, "model", None) is None:
            self.warnings.append("A speech model is in mock mode (weights missing), so voice was skipped.")
            return False
        # Untimed: the spoken questions, and one warmup of each model
        audio = {q["id"]: tts.synthesize_array(q["question"]) for q in questions}
        rate0, pcm0 = audio[questions[0]["id"]]
        stt.transcribe(float32_to_int16(resample_audio(int16_to_float32(pcm0), orig_sr=rate0)).tobytes())
        tts.synthesize_array("Warm up.")

        self.clear_cache()
        for pass_name in PASSES:
            for item in questions:
                rate, pcm = audio[item["id"]]
                record = {"pass": pass_name, "id": item["id"], "transcript": None, "transcript_matches": None,
                          "stt_ms": None, "query_ms": None, "clean_ms": None, "tts_ms": None,
                          "total_ms": None, "cache": None, "refused": None, "spoken_chars": None,
                          "audio_s": None, "error": None}
                started = time.perf_counter()
                # Same steps as app.voice_query: resample to 16 kHz, then transcribe
                mono = resample_audio(int16_to_float32(pcm), orig_sr=rate, target_sr=16000)
                transcript = stt.transcribe(float32_to_int16(mono).tobytes())
                record["stt_ms"] = (time.perf_counter() - started) * 1000
                record["transcript"] = transcript
                record["transcript_matches"] = words(transcript) == words(item["question"])
                if not transcript.strip():
                    record["error"] = "empty transcript"
                else:
                    data, record["query_ms"], record["error"] = self.ask(transcript)
                    if record["error"] is None:
                        record["cache"] = data.get("cache")
                        record["refused"] = is_refusal(data.get("answer", ""))
                        t = time.perf_counter()
                        spoken = sanitize_for_speech(data.get("answer", ""))
                        record["clean_ms"] = (time.perf_counter() - t) * 1000
                        t = time.perf_counter()
                        out_rate, out = tts.synthesize_array(spoken) if spoken else (24000, [])
                        record["tts_ms"] = (time.perf_counter() - t) * 1000
                        record["spoken_chars"] = len(spoken)
                        record["audio_s"] = len(out) / out_rate
                record["total_ms"] = (time.perf_counter() - started) * 1000
                records.append(record)
                print(f"  [voice {pass_name}] {item['id']:>2} {record['total_ms']:8.0f} ms"
                      f"{'  ERROR ' + record['error'] if record['error'] else ''}")
                self.track(record["error"])
        return True

    # facts

    def placement(self, model):
        def read():
            models = self.session.get(f"{self.ollama_host}/api/ps", timeout=SHORT_TIMEOUT).json().get("models", [])
            for m in models:
                if model_key(m.get("name") or m.get("model", "")) == model_key(model):
                    size, vram = m.get("size", 0), m.get("size_vram", 0)
                    where = "gpu" if size and vram >= size else ("partial" if vram > 0 else "cpu")
                    return {"placement": where, "size": size, "size_vram": vram}
            return None
        return fact(read) or {"placement": None, "size": None, "size_vram": None}

    def facts(self, health):
        return {
            "label": self.label,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": fact(lambda: git("rev-parse", "--short", "HEAD")),
            "git_dirty": fact(lambda: bool(git("status", "--porcelain", "--untracked-files=no"))),
            "api_base": self.api,
            "machine": {"os": fact(platform.platform), "cpu": fact(cpu_name), "ram_gb": fact(ram_gb)},
            "ollama": {"host": self.ollama_host, "model": health.get("generation_model")},
            "corpus": fact(lambda: self.session.get(f"{self.api}/corpus", timeout=SHORT_TIMEOUT).json()),
            "golden_sha256": fact(lambda: hashlib.sha256(self.golden_path.read_bytes()).hexdigest()),
            "config": {"retrieval": get("retrieval"), "llm": get("llm")},
            "notes": NOTES,
        }

    # summary and output

    @staticmethod
    def summarize_run(text, voice):
        summary = {}
        for pass_name in PASSES:
            ok = [r for r in text if r["pass"] == pass_name and r["error"] is None]
            groups = {}
            for group in TEXT_GROUPS:
                rows = [r for r in ok if r["path"] == group]
                stage_names = sorted({name for r in rows for name in r["stages"]})
                metrics = {name: summarize([r["stages"][name] for r in rows if name in r["stages"]])
                           for name in stage_names}
                for key in ("untraced_ms", "server_total_ms", "client_ms"):
                    metrics[key] = summarize([r[key] for r in rows if r[key] is not None])
                groups[group] = {"n": len(rows), "metrics": metrics}
            vrows = [r for r in voice if r["pass"] == pass_name and r["error"] is None]
            groups["voice"] = {"n": len(vrows),
                               "metrics": {m: summarize([r[m] for r in vrows if r[m] is not None])
                                           for m in VOICE_METRICS}}
            summary[pass_name] = groups
        summary["errors"] = sum(r["error"] is not None for r in text + voice)
        # An in scope question refused, or an out of scope one answered
        summary["in_scope_mismatch"] = sum(1 for r in text
                                           if r["error"] is None and r["refused"] is not None
                                           and r["refused"] == r["in_scope"])
        return summary

    def write(self, run):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.label}"
        for n in range(1, 100):
            path = self.out_dir / (f"{stem}.json" if n == 1 else f"{stem}-{n}.json")
            try:
                with open(path, "x", encoding="utf-8") as f:   # never overwrite an earlier run
                    json.dump(run, f, indent=2)
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f"Too many runs named {stem}")

    @staticmethod
    def report(run):
        meta, summary = run["meta"], run["summary"]
        machine, ollama = meta["machine"], meta["ollama"]
        print("\n" + "=" * 72)
        print(f"Latency benchmark · {meta['label']} · {meta['started_at']} · commit {meta['git_commit']}"
              f"{' (dirty)' if meta['git_dirty'] else ''}")
        print(f"Machine: {machine['cpu']} · {machine['ram_gb']} GB · {machine['os']}")
        print(f"Model:   {ollama['model']} at {ollama['host']} · placement start "
              f"{ollama['placement_start']}, end {ollama['placement_end']}")

        def fmt(v):
            return "    —" if v is None else f"{v:7.0f}"

        for pass_name in PASSES:
            for group, block in summary[pass_name].items():
                if not block["n"]:
                    continue
                title = "first run (cache cleared)" if pass_name == "first" else "repeat"
                print(f"\n{title} · {group} (n={block['n']})        p50 ms   p95 ms")
                for metric, s in block["metrics"].items():
                    print(f"  {metric:<24} {fmt(s['p50'])}  {fmt(s['p95'])}   n={s['n']}")
        print(f"\nErrors: {summary['errors']} · refusals that disagree with in_scope: "
              f"{summary['in_scope_mismatch']}{' · ABORTED early' if meta.get('aborted') else ''}")
        if not meta.get("voice_valid"):
            print("Voice: not measured (skipped or mock mode).")
        for line in meta.get("warnings", []) + meta["notes"]:
            print(f"Note: {line}")

    def run(self):
        health = self.preflight()
        questions, voice_questions = self.load_questions()
        meta = self.facts(health)
        model = health.get("generation_model") or ""
        text, voice, aborted = [], [], None
        voice_valid = False
        unknown = {"placement": None, "size": None, "size_vram": None}
        start = unknown
        try:
            # Warm the answer model with a real question ("warmup" itself would be refused)
            warm = next((q for q in questions if q["in_scope"]), questions[0])
            self.ask(warm["question"])
            start = self.placement(model)
            self.clear_cache()
            for pass_name in PASSES:
                text += self.text_pass(pass_name, questions)
            if not self.skip_voice and voice_questions:
                voice_valid = self.voice_turns(voice_questions, voice)
        except Aborted as e:
            aborted = str(e)
        end = self.placement(model)

        meta["ollama"].update({"placement_start": start["placement"], "placement_end": end["placement"],
                               "size": end["size"] or start["size"], "size_vram": end["size_vram"] or start["size_vram"]})
        meta.update({"voice_valid": voice_valid, "aborted": bool(aborted), "abort_reason": aborted,
                     "warnings": self.warnings, "resampler": fact(_resampler),
                     "questions": {"in_scope": sum(q["in_scope"] for q in questions),
                                   "out_of_scope": sum(not q["in_scope"] for q in questions),
                                   "voice": len(voice_questions) if voice_valid else 0}})
        run = {"meta": meta, "text": text, "voice": voice, "summary": self.summarize_run(text, voice)}
        self.report(run)
        return self.write(run), run


def _resampler():
    try:
        import librosa  # noqa: F401
        return "librosa"
    except ImportError:
        return "numpy"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per stage latency benchmark over the golden questions.")
    parser.add_argument("--label", default="run")
    parser.add_argument("--limit", type=int, default=None, help="first N questions of each kind")
    parser.add_argument("--skip-voice", action="store_true")
    parser.add_argument("--api", default=os.environ.get("VRAG_API_BASE", "http://localhost:8000/api"))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = parser.parse_args(argv)

    bench = Benchmark(args.api, args.ollama_host, limit=args.limit, skip_voice=args.skip_voice, label=args.label)
    try:
        path, _ = bench.run()
    except ApiUnavailable as e:
        print(f"[benchmark] {e}")
        return 1
    print(f"\nSaved {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
