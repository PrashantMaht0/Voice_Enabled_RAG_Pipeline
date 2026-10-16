"""The dashboard (Gradio): Ask (typed and spoken questions), Documents, and Observability.

It talks to the API only over HTTP (`VRAG_API_BASE`), except for speech: speech to text and text to speech run
in this process, so audio never travels to the API. Start it with `python app.py`."""
import json
import os
import queue
import re
import threading
import time
from urllib.parse import quote

import gradio as gr
import numpy as np
import requests

from src.utils.audio_utils import float32_to_int16, int16_to_float32, resample_audio
from src.utils.config import get
from src.utils.text_sanitizer import CITATION_TAG, SOURCE_HEADER, is_filler_transcript, parse_tag
from src.voice.streaming import SentenceSpeaker
from src.voice.stt import WhisperSTT
from src.voice.tts import KokoroTTS

API_BASE = os.environ.get("VRAG_API_BASE", "http://localhost:8000/api")
MAX_QUESTION_CHARS = 1000      # the API refuses longer questions (spec 0008, AC-4)
MAX_TURN_CHARS = 2000          # and longer history turns
TRUST_FORWARDED = os.environ.get("VRAG_UI_TRUST_FORWARDED", "").lower() in ("1", "true", "yes")


class ApiError(Exception):
    """An API reply that is not a success, already turned into a message for the user."""


def client_headers(request) -> dict:
    """X-VRAG-Client: the browser's address, so the API's rate limits count each user (spec 0008, AC-15). Behind a
    trusted proxy (VRAG_UI_TRUST_FORWARDED) it is the first X-Forwarded-For hop."""
    if request is None:
        return {}
    address = None
    if TRUST_FORWARDED:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        address = forwarded or None
    if address is None and request.client is not None:
        address = request.client.host
    return {"X-VRAG-Client": address} if address else {}


def api_message(response, name=None) -> str:
    """A plain message for an API reply that is not a success."""
    code = response.status_code
    try:
        detail = response.json().get("detail")
    except Exception:
        detail = None
    if isinstance(detail, list):                       # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)).removeprefix("Value error, ") for d in detail)
    if code == 409 and name:
        return f"A document named {name} is already in the corpus. Delete it first to replace it."
    if code == 413:
        return f"{name} is too large (limit 25 MB or 300 pages)." if name else "The request is too large."
    if code == 429:
        return f"Too many requests, please wait {response.headers.get('Retry-After', 'a few')} seconds."
    if code == 503:
        return "The system is busy, please try again in a few seconds."
    return detail or f"API error {code}."

custom_css = """
/* The spoken answer player says so while it is empty (spec 0006, polish child) */
#voice-reply .empty::after { content: "No spoken answer yet."; margin-left: 8px; }
#trace_box textarea { background-color: #0d1117; color: #58a6ff; font-family: monospace; font-size: 12px; }
.metric-box { text-align: center; padding: 15px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; }
.metric-value { font-size: 1.5rem; font-weight: bold; color: #38bdf8; }
.metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
.table-scroll { overflow-x: auto; }
/* Observability tab (spec 0007): two rows of five cards, the trace panel beside the recent table */
.metric-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
.metric-note { font-size: 0.7rem; color: #94a3b8; margin-top: 4px; overflow-wrap: anywhere; }
.obs-warning { background: #451a03; color: #fde68a; border: 1px solid #b45309; border-radius: 8px; padding: 8px 12px; }
.trace-panel { min-height: 240px; max-height: 560px; overflow-y: auto; border: 1px solid #334155; border-radius: 8px; padding: 4px 12px; }
/* Phone width: columns stack, buttons stay reachable, tables scroll inside themselves (spec 0006). Gradio puts
   its own container selector in front of every rule here, so selectors start below the container. */
@media (max-width: 767px) {
  html, body, .gradio-container { overflow-x: hidden; max-width: 100vw; }
  .row { flex-direction: column !important; flex-wrap: nowrap !important; }
  .row > * { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
  .row button { width: 100%; min-height: 44px; }
  .table-wrap, table { overflow-x: auto; display: block; max-width: 100%; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

"""

NOTHING_HEARD = "_I didn't catch that, please try again._"


def heard_nothing(transcript: str, no_speech_probs) -> bool:
    """True for a recording that is silence (every segment above the threshold) or only filler."""
    threshold = float(get("voice.no_speech_threshold", 0.6))
    if is_filler_transcript(transcript):
        return True
    return bool(no_speech_probs) and all(p > threshold for p in no_speech_probs)


# Speech models live in the dashboard process and load on first use.
_stt = None
_tts = None

HISTORY_TURNS = 6


def get_stt():
    """The speech to text model, loaded on the first voice turn."""
    global _stt
    if _stt is None:
        _stt = WhisperSTT()
    return _stt


def get_tts():
    """The text to speech model, loaded on the first voice turn."""
    global _tts
    if _tts is None:
        _tts = KokoroTTS()
    return _tts


def report_voice_timings(trace_id, **timings):
    """Voice timings go to the API's trace (only the API writes the database); a failure is ignored."""
    if trace_id is None:
        return
    try:
        requests.post(f"{API_BASE}/traces/{trace_id}/voice", json=timings, timeout=5)
    except Exception:
        pass


def voice_query(audio, history, request: gr.Request = None):
    """Transcribes a recording, streams the answer into the chat, and speaks each approved sentence as soon
    as it arrives (spec 0006). Yields the Ask tab's outputs plus one audio chunk per spoken sentence."""
    history = history or []
    if audio is None:
        yield (history, history, "", "", "_No audio captured._", *blank_metrics(), gr.skip(), gr.skip())
        return

    started = time.perf_counter()
    sample_rate, array = audio
    samples = np.asarray(array)
    if samples.ndim > 1:                       # stereo -> mono
        samples = samples.mean(axis=1)
    if samples.dtype != np.int16:
        samples = float32_to_int16(samples.astype(np.float32))

    mono = resample_audio(int16_to_float32(samples), orig_sr=sample_rate, target_sr=16000)
    question, no_speech = get_stt().transcribe_detailed(float32_to_int16(mono).tobytes())
    stt_ms = (time.perf_counter() - started) * 1000

    if heard_nothing(question, no_speech):
        # Silence or filler never reaches the API (spec 0006, AC-5)
        print(f"[VOICE] nothing heard: {question!r} {no_speech}")
        yield (history, history, "", "", NOTHING_HEARD, *blank_metrics(), gr.skip(), gr.skip())
        return

    speaker = SentenceSpeaker(get_tts())
    turn = history + [{"role": "user", "content": question}]
    meta, shown, outputs, first_audio_ms, last_audio_ms = None, [], None, None, None

    def audio_card():
        return metric_card(f"{first_audio_ms:.0f} ms" if first_audio_ms else "…", "First audio")

    def audio_chunks(wait=False):
        nonlocal first_audio_ms, last_audio_ms
        for spoken in speaker.ready(wait=wait):
            ready_ms = (spoken.ready_at - started) * 1000      # synthesized, from before resampling and STT
            first_audio_ms = ready_ms if first_audio_ms is None else first_audio_ms
            last_audio_ms = ready_ms
            yield spoken.wav_bytes()

    events = queue.Queue()
    headers = client_headers(request)                # read here: the request is not valid in another thread

    def read_stream():
        try:
            for event in stream_events(question, history, channel="voice", headers=headers):
                events.put(event)
        except ApiError as e:
            events.put({"type": "error", "detail": str(e)})
        except Exception as e:
            events.put({"type": "error", "detail": f"The API could not be reached ({type(e).__name__})."})
        finally:
            events.put(None)

    threading.Thread(target=read_stream, daemon=True).start()
    stream_open = True
    while stream_open:
        try:
            event = events.get(timeout=0.05)        # short waits, so ready audio is never held back
        except queue.Empty:
            event = False
        if event is None:
            stream_open = False
            speaker.close()
        elif event:
            if event["type"] == "meta":
                meta = event
                outputs = answer_outputs(turn, meta, None, [], "…")
            elif event["type"] == "sentence":
                shown.append(event["text"])
                speaker.say(event["text"])
                outputs = answer_outputs(turn, meta, None, [], " ".join(shown))
            elif event["type"] == "done":
                outputs = answer_outputs(turn, meta, event, event.get("trace", []))
            elif event["type"] == "error":
                outputs = error_outputs(history, turn, event["detail"])
            if outputs is not None:
                yield (*outputs, gr.skip(), audio_card())
        for chunk in audio_chunks():
            yield (*(outputs or (history, history, "", "", "", *blank_metrics())), chunk, audio_card())

    for chunk in audio_chunks(wait=True):      # sentences still being synthesized after the stream ended
        yield (*outputs, chunk, audio_card())

    print(f"[VOICE] q={question!r}")
    print(f"[VOICE] STT {stt_ms:.0f}ms | first audio {first_audio_ms or 0:.0f}ms | last audio {last_audio_ms or 0:.0f}ms")
    report_voice_timings((meta or {}).get("trace_id"), stt_ms=stt_ms, first_audio_ms=first_audio_ms,
                         spoken_total_ms=last_audio_ms)


def metric_card(value, label, note=None):
    """One metric card (a value over a small label, and an optional smaller second line) as HTML."""
    note_html = f"<div class='metric-note'>{note}</div>" if note else ""
    return (f"<div class='metric-box'><div class='metric-value'>{value}</div>"
            f"<div class='metric-label'>{label}</div>{note_html}</div>")


def blank_metrics():
    """The Ask tab's four cards before any answer."""
    return (
        metric_card("—", "Confidence"),
        metric_card("—", "Grounded"),
        metric_card("—", "Retrieval"),
        metric_card("—", "Total"),
    )


def format_trace(trace):
    """Renders the PipelineTracer spans returned by the API."""
    lines = []
    for span in trace:
        lines.append(f"[{span['layer'].upper():<12}] {span['elapsed_ms']:>8.1f} ms  {span['details']}")
    return "\n".join(lines)


KIND_LABELS = {"text": "text", "table": "table", "figure": "figure, generated"}
KIND_TITLES = {"text": "Text", "table": "Table", "figure": "Figure (generated)"}


def label_citations(answer, citations):
    """Each citation tag in the answer is followed by its source kinds, e.g. '(figure, generated)'."""
    kinds = {(c["source"], c.get("page")): c.get("kinds") or ["text"] for c in citations or []}

    def label(match):
        source, kind, number = parse_tag(match)
        found = kinds.get((source, number if kind == "Page" else None))
        return f"{match.group(0)} ({', '.join(KIND_LABELS.get(k, k) for k in found)})" if found else match.group(0)
    return CITATION_TAG.sub(label, answer or "")


def format_sources(chunks, details=None):
    """One card per retrieved chunk, headed by its file, page and kind."""
    if not chunks:
        return "_No chunks passed the confidence gate._"

    details = details or []
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        match = SOURCE_HEADER.match(chunk)
        if match:
            source, kind, number = parse_tag(match)
            locator = f"p.{number}" if kind == "Page" else f"chunk {number}"
            body = chunk[match.end():]
        else:
            source, locator, body = "unknown", "chunk -", chunk
        chunk_kind = details[i - 1].get("kind", "text") if i - 1 < len(details) else "text"
        body = "  \n".join(body.strip().splitlines())       # keep each line (figure fields, table rows) on its own
        blocks.append(f"**[{i}] {source} · {locator} · {KIND_TITLES.get(chunk_kind, chunk_kind)}**\n\n{body}")
    return "\n\n---\n\n".join(blocks)


def fetch_status():
    """The header's status line: whether the API and Ollama are online, and the answer model."""
    try:
        data = requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception as e:
        return f"### API offline\n`{e}`"

    if data.get("ollama_status") == "online":
        return f"### Online · {data.get('generation_model', 'unknown model')}"
    return "### Ollama offline"


def fetch_corpus():
    """The Documents table rows: file, chunk count, status (or one row saying the API is offline)."""
    try:
        rows = requests.get(f"{API_BASE}/corpus", timeout=30).json()
    except Exception as e:
        return [["API offline", 0, str(e)[:40]]]

    if not rows:
        return [["No documents in data/raw", 0, "Empty"]]
    # The API says Pending until a document's chunks are in the index; here that reads as work in progress
    return [[r["file"], r["chunks"], INDEXING if r["status"] == "Pending" else r["status"]] for r in rows]


INDEXING = "Indexing…"
POLL_EVERY_S = 3              # while a document is indexing, the Documents table refreshes itself this often
POLL_LIMIT_S = 15 * 60        # and gives up after this long (a file that yields no text never leaves Pending)


def still_indexing(rows) -> bool:
    """True while any document in the table is still being indexed."""
    return any(row[2] == INDEXING for row in rows)


def corpus_filenames():
    """The documents the delete picker offers."""
    try:
        rows = requests.get(f"{API_BASE}/corpus", timeout=30).json()
    except Exception:
        return []
    return [r["file"] for r in rows if isinstance(r, dict) and r.get("chunks") is not None]


def refresh_corpus():
    """Table, a refreshed delete picker, and the auto refresh switched on while anything is indexing (its start
    time goes to state)."""
    rows = fetch_corpus()
    busy = index_busy(rows, fetch_index_status())
    return rows, gr.update(choices=corpus_filenames(), value=None), gr.Timer(active=busy), time.time()


def fetch_index_status():
    """The API's index worker status (spec 0008), or None when it cannot be read."""
    try:
        status = requests.get(f"{API_BASE}/index", timeout=10).json()
    except Exception:
        return None
    return status if isinstance(status, dict) and "state" in status else None


def index_busy(rows, status) -> bool:
    """Whether to keep polling: the worker is building or has a build queued; without a status, any document
    still Pending."""
    if status is None:
        return still_indexing(rows)
    return status["state"] == "building" or bool(status.get("pending"))


def failure_line(status):
    """The Documents tab's line after a failed build."""
    if not status or status.get("state") != "failed":
        return None
    when = (status.get("finished_at") or "")[:19].replace("T", " ")
    return (f"Last indexing failed at {when} UTC: {status.get('last_error') or 'unknown error'}. "
            "The previous index is still answering.")


def poll_corpus(started):
    """One auto refresh tick: the table and picker choices (the picked value is kept), stopping with a status line
    once nothing is indexing or after POLL_LIMIT_S."""
    rows = fetch_corpus()
    status = fetch_index_status()
    if not index_busy(rows, status):
        return rows, gr.update(choices=corpus_filenames()), gr.Timer(active=False), \
            failure_line(status) or "Indexing finished."
    if time.time() - (started or 0) > POLL_LIMIT_S:
        return rows, gr.skip(), gr.Timer(active=False), \
            "Still indexing after 15 minutes. Check the API console, then press Refresh."
    return rows, gr.skip(), gr.Timer(active=True), gr.skip()


def delete_document(filename, request: gr.Request = None):
    """Deletes the chosen document through the API, which re indexes in the background; returns the table, the
    picker, a status line and the polling timer with its start time."""
    if not filename:
        return fetch_corpus(), gr.update(), "Pick a document first.", gr.Timer(active=False), 0

    try:
        response = requests.delete(f"{API_BASE}/corpus/{quote(filename)}", timeout=30,
                                   headers=client_headers(request))
        if response.status_code != 200:
            return fetch_corpus(), gr.update(choices=corpus_filenames()), \
                f"Delete failed: {api_message(response, filename)}", gr.Timer(active=False), 0
    except Exception as e:
        return fetch_corpus(), gr.update(choices=corpus_filenames()), \
            f"Delete failed: the API could not be reached ({type(e).__name__}).", gr.Timer(active=False), 0

    return (
        fetch_corpus(),
        gr.update(choices=corpus_filenames(), value=None),
        f"Deleted **{filename}**. Re indexing…",
        gr.Timer(active=True),
        time.time(),
    )


def upload_document(file_path, request: gr.Request = None):
    """Uploads a file through the API (indexing runs in the background); returns the table, the picker and a status
    line."""
    if file_path is None:
        return fetch_corpus(), gr.update(), "No file selected.", gr.Timer(active=False), 0
    name = os.path.basename(file_path)
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{API_BASE}/upload", files={"file": (name, f)}, timeout=120,
                                     headers=client_headers(request))
        if response.status_code != 200:
            return fetch_corpus(), gr.update(), f"Upload failed: {api_message(response, name)}", \
                gr.Timer(active=False), 0
    except Exception as e:
        return fetch_corpus(), gr.update(), f"Upload failed: the API could not be reached ({type(e).__name__}).", \
            gr.Timer(active=False), 0

    return (
        fetch_corpus(),
        gr.update(choices=corpus_filenames()),
        f"Indexing {name}… the table updates by itself when it is done (large PDFs with tables and charts can "
        "take a few minutes).",
        gr.Timer(active=True),
        time.time(),
    )


def stream_events(question, history, channel="typed", headers=None):
    """Events from /api/query/stream: meta, one per approved sentence, then done (or error). A refused request
    raises ApiError with the message to show."""
    turns = [{"role": t["role"], "content": str(t["content"])[:MAX_TURN_CHARS]} for t in history[-HISTORY_TURNS:]]
    with requests.post(f"{API_BASE}/query/stream", stream=True, timeout=300, headers=headers or {},
                       json={"query": question, "history": turns, "channel": channel}) as r:
        if r.status_code != 200:
            raise ApiError(api_message(r))
        for line in r.iter_lines():
            if line:
                yield json.loads(line)


def answer_outputs(history, meta, done, trace, partial=None):
    """The Ask tab's outputs for a finished (done) or growing (partial) answer."""
    answer = label_citations(done["answer"], done.get("citations")) if done else partial
    transcript = history + [{"role": "assistant", "content": answer}]
    grounded = done.get("grounded") if done else None
    dropped = len(done.get("dropped_sentences") or []) if done else 0
    grounded_label = "N/A" if not done else {True: "PASS", False: f"FAIL ({dropped} removed)", None: "Unchecked"}[grounded]
    retrieval_ms = sum(s["elapsed_ms"] for s in trace if s["layer"] in ("bm25", "vector", "fusion", "rerank"))
    sources = format_sources(meta.get("retrieved_context", []), meta.get("retrieved")) if meta else ""
    if dropped:
        sources = (f"_{dropped} sentence{'s' if dropped > 1 else ''} removed: not supported by your documents._"
                   f"\n\n{sources}")
    return (
        transcript,                           # state
        transcript,                           # transcript
        "",                                   # clear the input box
        format_trace(trace),
        sources,
        metric_card(f"{(meta or {}).get('confidence_score') or 0.0:.3f}", "Confidence"),
        metric_card(grounded_label, "Grounded"),
        metric_card(f"{retrieval_ms:.0f} ms", "Retrieval"),
        metric_card(f"{done['total_ms']:.0f} ms" if done else "…", "Total"),
    )


def error_outputs(history, turn, message):
    """The Ask tab's outputs when a question could not be answered: the message shows in the chat, but the
    conversation sent with the next question stays as it was."""
    shown = turn + [{"role": "assistant", "content": message}]
    return history, shown, "", "", "", *blank_metrics()


def run_query(question, history, request: gr.Request = None, channel="typed"):
    """Text turn: stream approved sentences into the chat as they arrive (spec 0006)."""
    history = history or []

    if not question.strip():
        yield history, history, "", "", "", *blank_metrics()
        return

    turn = history + [{"role": "user", "content": question}]
    meta, sentences = None, []
    try:
        for event in stream_events(question, history, channel, headers=client_headers(request)):
            if event["type"] == "meta":
                meta = event
                yield answer_outputs(turn, meta, None, [], "…")
            elif event["type"] == "sentence":
                sentences.append(event["text"])
                yield answer_outputs(turn, meta, None, [], " ".join(sentences))
            elif event["type"] == "done":
                outputs = answer_outputs(turn, meta, event, event.get("trace", []))
                yield outputs
                return
            elif event["type"] == "error":
                yield error_outputs(history, turn, event["detail"])
                return
    except ApiError as e:
        yield error_outputs(history, turn, str(e))
    except Exception as e:
        yield error_outputs(history, turn, f"The API could not be reached ({type(e).__name__}).")


def clear_conversation():
    """Empties the chat, sources, trace and every card, including first audio."""
    return [], [], "", "", "_Ask a question to see the grounding chunks._", *blank_metrics(), \
        metric_card("—", "First audio")


WINDOWS = {"Last hour": "1h", "Last 24 hours": "24h", "Last 7 days": "7d", "All": "all"}
STAGE_NAMES = {"rewrite": "Rewrite", "bm25": "Keyword search", "vector": "Vector search", "fusion": "Fusion",
               "rerank": "Rerank", "generation": "Generation", "grounding": "Grounding check",
               "first_sentence_ms": "First sentence", "total_ms": "Total", "stt_ms": "Speech recognition (voice)",
               "first_audio_ms": "First audio (voice)", "spoken_total_ms": "Spoken total (voice)"}
NO_TRAFFIC = "<p><em>No questions in this window yet.</em></p>"   # shown in gr.HTML, so HTML, not Markdown
NO_TRACE = "_Select a question to see its trace._"
TOKEN_LAYERS = ("rewrite", "generation", "grounding")


def percent(value):
    """A rate as a whole percentage, or a dash when there is none."""
    return "—" if value is None else f"{value * 100:.0f}%"


def plural(n, word):
    """'1 error', '3 errors'."""
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def format_ms(value):
    """Rounded to whole ms first, then ms below 1,000 and seconds with one decimal from 1,000 up."""
    if value is None:
        return "—"
    ms = round(value)
    return f"{ms:,} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def error_rate(errors, asked):
    """Errors over asked from the counts, one decimal; never '0.0%' while there are errors."""
    if not asked:
        return "—"
    value = errors / asked * 100
    return "<0.1%" if errors and round(value, 1) == 0 else f"{value:.1f}%"


def failed_writes_line(failed):
    """The warning above the cards, only when the trace store failed to save something."""
    if not failed:
        return ""
    who = "1 question" if failed == 1 else f"{failed:,} questions"
    return f"<div class='obs-warning'>{who} could not be saved to the trace store since the API started.</div>"


def observability_cards(m):
    """The ten cards in the wireframe's order (spec 0007, AC-2)."""
    c, r = m["counts"], m["rates"]
    ret, grd, lat, tok = m["retrieval"], m["grounding"], m["latency"], m["tokens"]
    refused = c["refused_gate"] + c["refused_grounding"] + c["refused_model"] + c["refused_empty"]
    note = lambda rate, text: text if rate is not None else None
    cards = [
        metric_card(f"{c['asked']:,}", "Questions"),
        metric_card(f"{c['answered']:,}", "Answered"),
        metric_card(percent(r["retrieval_pass_rate"]), "Retrieval quality",
                    note(r["retrieval_pass_rate"], f"{ret['passed']:,} of {ret['reached']:,} passed the gate")),
        metric_card(percent(r["grounding_pass_rate"]), "Generation quality",
                    note(r["grounding_pass_rate"],
                         f"{grd['passed']:,} of {grd['checked']:,} checked answers had nothing removed")),
        metric_card(percent(r["refusal_rate"]), "Refusal rate", note(r["refusal_rate"], f"{refused:,} refused")),
        metric_card(format_ms(lat["p50"]), "p50 latency", note(lat["p50"], f"over {lat['n']:,} answered")),
        metric_card(format_ms(lat["p99"]), "p99 latency", note(lat["p99"], f"over {lat['n']:,} answered")),
        metric_card(error_rate(c["errors"], c["asked"]), "Error rate",
                    plural(c["errors"], "error") if c["asked"] else None),
        metric_card(f"{tok['total']:,}", "Tokens used",
                    (f"{tok['per_question']:,} per question · " if tok["per_question"] is not None else "")
                    + f"{tok['counted']:,} of {tok['asked']:,} counted" if tok["asked"] else None),
        metric_card(f"{c['cache_hits']:,}", "Cache hits",
                    note(r["cache_hit_rate"], f"{percent(r['cache_hit_rate'])} hit rate")),
    ]
    html = "<div class='metric-grid'>" + "".join(cards) + "</div>"
    return html + (NO_TRAFFIC if not c["asked"] else "")


def fetch_observability(window_label):
    """Warning line, cards, recent questions (with their ids), the trace panel's default and the stage table for
    the chosen window (spec 0007)."""
    window = WINDOWS[window_label]
    try:
        m = requests.get(f"{API_BASE}/metrics", params={"window": window}, timeout=30).json()
        recent = requests.get(f"{API_BASE}/traces", params={"limit": 50, "window": window}, timeout=30).json()
    except Exception as e:
        return gr.update(value="", visible=False), f"API error: {e}", [], [], NO_TRACE, []
    fmt = lambda v: "—" if v is None else f"{v:.0f}"
    stages = [[STAGE_NAMES.get(k, k), fmt(v["p50"]), fmt(v["p99"]), v["n"]]
              for k, v in {**m["stages"], **m["voice"]}.items() if v["n"]]
    rows = [[t["created_at"][:19].replace("T", " "), t["channel"], t["question"], "yes" if t["answered"] else "no"]
            for t in recent]
    picks = [[row, t["id"]] for row, t in zip(rows, recent)]
    warning = failed_writes_line(m.get("failed_writes"))
    return gr.update(value=warning, visible=bool(warning)), observability_cards(m), rows, picks, NO_TRACE, stages


def token_line(prompt, output):
    """'prompt 1,234 · output 56 · total 1,290'."""
    return f"prompt {prompt:,} · output {output:,} · total {prompt + output:,}"


def selected_trace_id(picks, event):
    """The id of the selected row, found by its values so a sorted table still opens its own trace."""
    value = getattr(event, "row_value", None)
    if value is not None:
        for row, trace_id in picks:
            if [str(v) for v in row] == [str(v) for v in value]:
                return trace_id
    index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    return picks[index][1] if index is not None and index < len(picks) else None


def open_trace(picks, event: gr.SelectData):
    """The full trace of the selected recent question."""
    trace_id = selected_trace_id(picks, event)
    if trace_id is None:
        return NO_TRACE
    try:
        t = requests.get(f"{API_BASE}/traces/{trace_id}", timeout=30).json()
    except Exception as e:
        return f"API error: {e}"
    lines = [f"### {t['question']}", f"{t['created_at'][:19].replace('T', ' ')} · {t['channel']} · path **{t['path']}** · "
             f"total {t['total_ms']:.0f} ms" + (f" · confidence {t['confidence']:.3f}" if t.get("confidence") is not None else "")]
    if t.get("first_audio_ms"):
        lines.append(f"Voice: speech recognition {t['stt_ms']:.0f} ms · first audio {t['first_audio_ms']:.0f} ms · "
                     f"spoken total {t['spoken_total_ms']:.0f} ms")
    if t.get("prompt_tokens") is None:
        lines += ["", "**Tokens:** not recorded"]
    else:
        lines += ["", f"**Tokens:** {token_line(t['prompt_tokens'], t['output_tokens'] or 0)}", ""]
        for s in t.get("spans", []):
            if s["layer"] in TOKEN_LAYERS:
                pair = "—" if s.get("prompt_tokens") is None else token_line(s["prompt_tokens"], s["output_tokens"] or 0)
                lines.append(f"- {STAGE_NAMES[s['layer']]}: {pair}")
    lines += ["", "**Stages**", "", "| Stage | ms | Details |", "|---|---:|---|"]
    lines += [f"| {s['layer']} | {s['elapsed_ms']:.0f} | {s['details'] or ''} |" for s in t.get("spans", [])]
    if t.get("retrieved"):
        lines += ["", "**Retrieved sources**", ""]
        lines += [f"- {d['source']} · p.{d['page']} · {KIND_TITLES.get(d['kind'], d['kind'])} · score {d['score']}"
                  for d in t["retrieved"]]
    if t.get("dropped_sentences"):
        lines += ["", "**Removed sentences** (not supported by your documents)", ""]
        lines += [f"- {d}" for d in t["dropped_sentences"]]
    lines += ["", "**Answer**", "", t.get("answer") or "_none_"]
    if t.get("error"):
        lines += ["", f"**Error:** {t['error']}"]
    return "\n".join(lines)


def picker_changed(value):
    """Delete document is only usable once a document is chosen."""
    return gr.update(interactive=bool(value))


with gr.Blocks(title="V-RAG", theme=gr.themes.Soft(), css=custom_css) as demo:

    with gr.Row(equal_height=True):
        gr.Markdown("## V-RAG\n*Ask your documents by text or voice. Answers are checked against them and cited.*")
        status_md = gr.Markdown("### Checking…")

    history_state = gr.State([])

    with gr.Tabs():

        # ---------------------------------------------------------------- Ask
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=420,
                        show_label=False,
                        placeholder="Ask about the indexed documents.",
                    )
                    query_input = gr.Textbox(
                        placeholder="What is an Azure App Service Web App?",
                        show_label=False,
                        lines=1,            # one line box: Enter asks (Gradio submits on Enter only then)
                        max_lines=4,        # and it still grows while you type a long question
                        max_length=MAX_QUESTION_CHARS,
                    )
                    with gr.Row():
                        ask_btn = gr.Button("Ask", variant="primary", scale=3)
                        clear_btn = gr.Button("Clear chat", scale=1)

                    with gr.Accordion("Or speak your question", open=False):
                        mic = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Hold the mic: press Record, speak, press Stop",
                        )
                        gr.Markdown(
                            "Recording stops on your command, so background noise cannot cut you "
                            "off mid-answer.\n\n"
                            "**Microphone access needs a secure origin.** Use "
                            "`http://localhost:7860`; a LAN address like `http://192.168.x.x:7860` "
                            "is blocked by the browser unless served over HTTPS."
                        )
                        voice_reply = gr.Audio(label="Spoken answer", autoplay=True, interactive=False, elem_id="voice-reply",
                                               streaming=True)

                with gr.Column(scale=4):
                    with gr.Row():
                        m_conf = gr.HTML(metric_card("—", "Confidence"))
                        m_grounded = gr.HTML(metric_card("—", "Grounded"))
                    with gr.Row():
                        m_retrieval = gr.HTML(metric_card("—", "Retrieval"))
                        m_total = gr.HTML(metric_card("—", "Total"))
                    m_audio = gr.HTML(metric_card("—", "First audio"))

                    with gr.Accordion("Retrieved sources", open=True):
                        sources_output = gr.Markdown("_Ask a question to see the grounding chunks._")

                    with gr.Accordion("Pipeline trace", open=False):
                        trace_output = gr.Textbox(
                            lines=9,
                            elem_id="trace_box",
                            show_label=False,
                            interactive=False,
                            placeholder="Timings appear here after a query.",
                        )

        # ------------------------------------------------------------- Corpus
        with gr.Tab("Documents") as docs_tab:
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="Add a PDF, TXT or MD file")
                    upload_status = gr.Markdown("")
                    refresh_btn = gr.Button("Refresh")

                    gr.Markdown("### Remove a document")
                    delete_picker = gr.Dropdown(
                        label="Document",
                        choices=[],
                        interactive=True,
                    )
                    delete_btn = gr.Button("Delete document", variant="stop", interactive=False)
                    gr.Markdown(
                        "_Deletes the file from `data/raw` and rebuilds both indexes "
                        "from what remains. This cannot be undone._"
                    )
                with gr.Column(scale=3):
                    corpus_timer = gr.Timer(POLL_EVERY_S, active=False)
                    poll_started = gr.State(0)
                    corpus_table = gr.Dataframe(
                        elem_classes=["table-scroll"],
                        headers=["File", "Chunks", "Status"],
                        datatype=["str", "number", "str"],
                        interactive=False,
                        label="Indexed documents",
                    )

        # ------------------------------------------------------ Observability
        with gr.Tab("Observability") as obs_tab:
            gr.Markdown("### Observability\nHow the system behaved in the chosen window, from every question asked.")
            with gr.Row():
                obs_window = gr.Dropdown(list(WINDOWS), value="Last 24 hours", label="Window", scale=3)
                obs_refresh = gr.Button("Refresh", scale=1)
            obs_warning = gr.HTML("", visible=False)
            obs_cards = gr.HTML(NO_TRAFFIC)
            obs_ids = gr.State([])
            with gr.Row():
                with gr.Column(scale=3, min_width=0):
                    obs_recent = gr.Dataframe(headers=["Time (UTC)", "Channel", "Question", "Answered"],
                                              interactive=False, wrap=True, max_height=560,
                                              column_widths=["22%", "13%", "50%", "15%"],
                                              label="Recent questions (select one to see its trace)",
                                              elem_classes=["table-scroll"])
                with gr.Column(scale=2, min_width=0):
                    obs_trace = gr.Markdown(NO_TRACE, elem_classes=["trace-panel"])
            obs_stages = gr.Dataframe(headers=["Stage", "p50 ms", "p99 ms", "n"], interactive=False,
                                      label="Time per stage", elem_classes=["table-scroll"])

    ask_outputs = [history_state, chatbot, query_input, trace_output, sources_output,
                   m_conf, m_grounded, m_retrieval, m_total]

    ask_btn.click(fn=run_query, inputs=[query_input, history_state], outputs=ask_outputs)
    query_input.submit(fn=run_query, inputs=[query_input, history_state], outputs=ask_outputs)
    clear_btn.click(fn=clear_conversation, outputs=ask_outputs + [m_audio])

    mic.stop_recording(fn=voice_query, inputs=[mic, history_state],
                       outputs=ask_outputs + [voice_reply, m_audio])

    file_input.upload(fn=upload_document, inputs=[file_input],
                      outputs=[corpus_table, delete_picker, upload_status, corpus_timer, poll_started])
    refresh_btn.click(fn=refresh_corpus, outputs=[corpus_table, delete_picker, corpus_timer, poll_started])
    docs_tab.select(fn=refresh_corpus, outputs=[corpus_table, delete_picker, corpus_timer, poll_started])
    corpus_timer.tick(fn=poll_corpus, inputs=[poll_started],
                      outputs=[corpus_table, delete_picker, corpus_timer, upload_status])
    delete_btn.click(fn=delete_document, inputs=[delete_picker],
                     outputs=[corpus_table, delete_picker, upload_status, corpus_timer, poll_started])

    delete_picker.change(fn=picker_changed, inputs=[delete_picker], outputs=[delete_btn])

    obs_outputs = [obs_warning, obs_cards, obs_recent, obs_ids, obs_trace, obs_stages]
    obs_tab.select(fn=fetch_observability, inputs=[obs_window], outputs=obs_outputs)
    obs_refresh.click(fn=fetch_observability, inputs=[obs_window], outputs=obs_outputs)
    obs_window.change(fn=fetch_observability, inputs=[obs_window], outputs=obs_outputs)
    obs_recent.select(fn=open_trace, inputs=[obs_ids], outputs=[obs_trace])

    demo.load(fn=fetch_status, outputs=[status_md])
    demo.load(fn=refresh_corpus, outputs=[corpus_table, delete_picker, corpus_timer, poll_started])

if __name__ == "__main__":
    # This machine only unless VRAG_UI_HOST says otherwise (Docker sets 0.0.0.0 inside its container)
    host = os.environ.get("VRAG_UI_HOST", "127.0.0.1")
    print("\nOpen http://localhost:7860 — microphone access is blocked on other hostnames.\n")
    # Gradio refuses files over the API's upload limit before sending them, and runs at most 4 events of a kind at
    # once, matching the API's answer slots (spec 0008)
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name=host, server_port=7860, max_file_size="25mb")
