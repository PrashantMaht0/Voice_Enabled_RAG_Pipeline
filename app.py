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
from src.utils.text_sanitizer import CITATION_TAG, SOURCE_HEADER, parse_tag
from src.voice.streaming import SentenceSpeaker
from src.voice.stt import WhisperSTT
from src.voice.tts import KokoroTTS

API_BASE = os.environ.get("VRAG_API_BASE", "http://localhost:8000/api")

custom_css = """
/* The spoken answer player says so while it is empty (spec 0006, polish child) */
#voice-reply .empty::after { content: "No spoken answer yet."; margin-left: 8px; }
#trace_box textarea { background-color: #0d1117; color: #58a6ff; font-family: monospace; font-size: 12px; }
.metric-box { text-align: center; padding: 15px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; }
.metric-value { font-size: 1.5rem; font-weight: bold; color: #38bdf8; }
.metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
.table-scroll { overflow-x: auto; }
/* Phone width: columns stack, buttons stay reachable, tables scroll inside themselves (spec 0006) */
@media (max-width: 767px) {
  html, body, .gradio-container { overflow-x: hidden; max-width: 100vw; }
  .gradio-container .row { flex-direction: column !important; flex-wrap: nowrap !important; }
  .gradio-container .row > * { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
  .gradio-container button { width: 100%; min-height: 44px; }
  .gradio-container .table-wrap, .gradio-container table { overflow-x: auto; display: block; max-width: 100%; }
}

"""

NOTHING_HEARD = "_I didn't catch that, please try again._"
# What Whisper writes for silence or a cough; never worth a search
FILLER_TRANSCRIPTS = {"", "you", "thank you", "thanks for watching", "bye"}


def heard_nothing(transcript: str, no_speech_probs) -> bool:
    """True for a recording that is silence (every segment above the threshold) or only filler."""
    threshold = float(get("voice.no_speech_threshold", 0.6))
    words = re.sub(r"[^\w\s]", "", (transcript or "").lower()).strip()
    words = re.sub(r"\s+", " ", words)
    if words in FILLER_TRANSCRIPTS:
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


def voice_query(audio, history):
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

    def read_stream():
        try:
            for event in stream_events(question, history, channel="voice"):
                events.put(event)
        except Exception as e:
            events.put({"type": "error", "detail": str(e)})
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
                outputs = (history, history, "", f"API error: {event['detail']}", "", *blank_metrics())
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


def metric_card(value, label):
    """One metric card (a value over a small label) as HTML."""
    return f"<div class='metric-box'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>"


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
    return [[r["file"], r["chunks"], r["status"]] for r in rows]


def corpus_filenames():
    """The documents the delete picker offers."""
    try:
        rows = requests.get(f"{API_BASE}/corpus", timeout=30).json()
    except Exception:
        return []
    return [r["file"] for r in rows if isinstance(r, dict) and r.get("chunks") is not None]


def refresh_corpus():
    """Table plus a refreshed delete picker."""
    return fetch_corpus(), gr.update(choices=corpus_filenames(), value=None)


def delete_document(filename):
    """Deletes the chosen document through the API (which rebuilds the index); returns the table, the picker and a
    status line."""
    if not filename:
        return fetch_corpus(), gr.update(), "Pick a document first."

    try:
        response = requests.delete(f"{API_BASE}/corpus/{quote(filename)}", timeout=600)
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            return fetch_corpus(), gr.update(choices=corpus_filenames()), f"Delete failed: {detail}"
    except Exception as e:
        return fetch_corpus(), gr.update(choices=corpus_filenames()), f"Delete failed: {e}"

    return (
        fetch_corpus(),
        gr.update(choices=corpus_filenames(), value=None),
        f"Deleted **{filename}** and rebuilt the indexes.",
    )


def upload_document(file_path):
    """Uploads a file through the API (indexing runs in the background); returns the table, the picker and a status
    line."""
    if file_path is None:
        return fetch_corpus(), gr.update(), "No file selected."
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            requests.post(f"{API_BASE}/upload", files=files, timeout=60)
    except Exception as e:
        return fetch_corpus(), gr.update(), f"Upload failed: {e}"

    return (
        fetch_corpus(),
        gr.update(choices=corpus_filenames()),
        f"Indexing {os.path.basename(file_path)}… hit Refresh in a moment to see it indexed.",
    )


def stream_events(question, history, channel="typed"):
    """Events from /api/query/stream: meta, one per approved sentence, then done (or error)."""
    with requests.post(f"{API_BASE}/query/stream", stream=True, timeout=300,
                       json={"query": question, "history": history[-HISTORY_TURNS:], "channel": channel}) as r:
        r.raise_for_status()
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


def run_query(question, history, channel="typed"):
    """Text turn: stream approved sentences into the chat as they arrive (spec 0006)."""
    history = history or []

    if not question.strip():
        yield history, history, "", "", "", *blank_metrics()
        return

    turn = history + [{"role": "user", "content": question}]
    meta, sentences = None, []
    try:
        for event in stream_events(question, history, channel):
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
                yield history, history, "", f"API error: {event['detail']}", "", *blank_metrics()
                return
    except Exception as e:
        yield history, history, "", f"API error: {e}", "", *blank_metrics()


def clear_conversation():
    """Empties the chat, sources, trace and every card, including first audio."""
    return [], [], "", "", "_Ask a question to see the grounding chunks._", *blank_metrics(), \
        metric_card("—", "First audio")


WINDOWS = {"Last hour": "1h", "Last 24 hours": "24h", "Last 7 days": "7d", "All": "all"}
STAGE_NAMES = {"rewrite": "Rewrite", "bm25": "Keyword search", "vector": "Vector search", "fusion": "Fusion",
               "rerank": "Rerank", "generation": "Generation", "grounding": "Grounding check",
               "first_sentence_ms": "First sentence", "total_ms": "Total", "stt_ms": "Speech recognition (voice)",
               "first_audio_ms": "First audio (voice)", "spoken_total_ms": "Spoken total (voice)"}
NO_TRAFFIC = "_No questions in this window yet._"


def percent(value):
    """A rate as a whole percentage, or a dash when there is none."""
    return "—" if value is None else f"{value * 100:.0f}%"


def fetch_observability(window_label):
    """Rate cards, stage table, quality card and recent questions for the chosen window."""
    try:
        m = requests.get(f"{API_BASE}/metrics", params={"window": WINDOWS[window_label]}, timeout=30).json()
        recent = requests.get(f"{API_BASE}/traces", params={"limit": 50}, timeout=30).json()
    except Exception as e:
        return f"API error: {e}", [], "", [], [], ""
    c, r = m["counts"], m["rates"]
    cards = [metric_card(c["asked"], "Asked"), metric_card(c["answered"], "Answered"),
             metric_card(percent(r["refusal_rate"]), "Refusal rate"), metric_card(percent(r["cache_hit_rate"]), "Cache hits"),
             metric_card(percent(r["grounding_pass_rate"]), "Grounding pass")]
    if m.get("failed_writes"):
        cards.append(metric_card(m["failed_writes"], "Failed writes"))
    cards_html = "<div style='display:flex;flex-wrap:wrap;gap:8px'>" + "".join(
        f"<div style='flex:1 1 120px'>{card}</div>" for card in cards) + "</div>"
    if not c["asked"]:
        cards_html += f"\n\n{NO_TRAFFIC}"
    fmt = lambda v: "—" if v is None else f"{v:.0f}"
    stages = [[STAGE_NAMES.get(k, k), fmt(v["p50"]), fmt(v["p95"]), v["n"]]
              for k, v in {**m["stages"], **m["voice"]}.items() if v["n"]]
    q = m.get("quality")
    quality = "_No benchmark run found._" if not q else (
        f"**Latest quality scores** · {q['file']} · finished {q['finished']} · setup {q['winner']}\n\n"
        f"Golden answered {q['golden_answered']} of {q['golden_in_scope']} · trick questions refused "
        f"{q['golden_refused']} of {q['golden_out_of_scope']} · faithfulness {q['faithfulness']} · "
        f"tables {q['tables']} of 29 · figures {q['figures']} of 31")
    rows = [[t["created_at"][:19].replace("T", " "), t["channel"], t["question"], t["path"], fmt(t["total_ms"])]
            for t in recent]
    return cards_html, stages, quality, rows, [t["id"] for t in recent], "_Select a question to see its trace._"


def open_trace(ids, event: gr.SelectData):
    """The full trace of the selected recent question."""
    row = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    if row is None or row >= len(ids):
        return "_Select a question to see its trace._"
    try:
        t = requests.get(f"{API_BASE}/traces/{ids[row]}", timeout=30).json()
    except Exception as e:
        return f"API error: {e}"
    lines = [f"### {t['question']}", f"{t['created_at'][:19].replace('T', ' ')} · {t['channel']} · path **{t['path']}** · "
             f"total {t['total_ms']:.0f} ms" + (f" · confidence {t['confidence']:.3f}" if t.get("confidence") is not None else "")]
    if t.get("first_audio_ms"):
        lines.append(f"Voice: speech recognition {t['stt_ms']:.0f} ms · first audio {t['first_audio_ms']:.0f} ms · "
                     f"spoken total {t['spoken_total_ms']:.0f} ms")
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
        with gr.Tab("Documents"):
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
                    corpus_table = gr.Dataframe(
                        elem_classes=["table-scroll"],
                        headers=["File", "Chunks", "Status"],
                        datatype=["str", "number", "str"],
                        interactive=False,
                        label="Indexed documents",
                    )

        # ------------------------------------------------------ Observability
        with gr.Tab("Observability") as obs_tab:
            with gr.Row():
                obs_window = gr.Dropdown(list(WINDOWS), value="Last 24 hours", label="Window", scale=3)
                obs_refresh = gr.Button("Refresh", scale=1)
            obs_cards = gr.HTML(NO_TRAFFIC)
            with gr.Row():
                with gr.Column(scale=1):
                    obs_stages = gr.Dataframe(headers=["Stage", "p50 ms", "p95 ms", "n"], interactive=False,
                                              label="Time per stage", elem_classes=["table-scroll"])
                with gr.Column(scale=1):
                    obs_quality = gr.Markdown("_No benchmark run found._")
            obs_ids = gr.State([])
            obs_recent = gr.Dataframe(headers=["Time (UTC)", "Channel", "Question", "Path", "Total ms"],
                                      interactive=False, label="Recent questions (select one to see its trace)",
                                      elem_classes=["table-scroll"])
            obs_trace = gr.Markdown("_Select a question to see its trace._")

    ask_outputs = [history_state, chatbot, query_input, trace_output, sources_output,
                   m_conf, m_grounded, m_retrieval, m_total]

    ask_btn.click(fn=run_query, inputs=[query_input, history_state], outputs=ask_outputs)
    query_input.submit(fn=run_query, inputs=[query_input, history_state], outputs=ask_outputs)
    clear_btn.click(fn=clear_conversation, outputs=ask_outputs + [m_audio])

    mic.stop_recording(fn=voice_query, inputs=[mic, history_state],
                       outputs=ask_outputs + [voice_reply, m_audio])

    file_input.upload(fn=upload_document, inputs=[file_input],
                      outputs=[corpus_table, delete_picker, upload_status])
    refresh_btn.click(fn=refresh_corpus, outputs=[corpus_table, delete_picker])
    delete_btn.click(fn=delete_document, inputs=[delete_picker],
                     outputs=[corpus_table, delete_picker, upload_status])

    delete_picker.change(fn=picker_changed, inputs=[delete_picker], outputs=[delete_btn])

    obs_outputs = [obs_cards, obs_stages, obs_quality, obs_recent, obs_ids, obs_trace]
    obs_tab.select(fn=fetch_observability, inputs=[obs_window], outputs=obs_outputs)
    obs_refresh.click(fn=fetch_observability, inputs=[obs_window], outputs=obs_outputs)
    obs_window.change(fn=fetch_observability, inputs=[obs_window], outputs=obs_outputs)
    obs_recent.select(fn=open_trace, inputs=[obs_ids], outputs=[obs_trace])

    demo.load(fn=fetch_status, outputs=[status_md])
    demo.load(fn=refresh_corpus, outputs=[corpus_table, delete_picker])

if __name__ == "__main__":
    host = os.environ.get("VRAG_UI_HOST", "0.0.0.0")
    print("\nOpen http://localhost:7860 — microphone access is blocked on other hostnames.\n")
    demo.launch(server_name=host, server_port=7860)
