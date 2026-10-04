import json
import os
import re
import time

import gradio as gr
import numpy as np
import requests
from urllib.parse import quote

from src.utils.audio_utils import resample_audio, int16_to_float32, float32_to_int16
from src.utils.text_sanitizer import SOURCE_HEADER, parse_tag, sanitize_for_speech
from src.voice.stt import WhisperSTT
from src.voice.tts import KokoroTTS

API_BASE = os.environ.get("VRAG_API_BASE", "http://localhost:8000/api")

custom_css = """
#trace_box textarea { background-color: #0d1117; color: #58a6ff; font-family: monospace; font-size: 12px; }
.metric-box { text-align: center; padding: 15px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; }
.metric-value { font-size: 1.5rem; font-weight: bold; color: #38bdf8; }
.metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }

"""

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Speech models live in the UI process and load on first use.
_stt = None
_tts = None

HISTORY_TURNS = 6


def get_stt():
    global _stt
    if _stt is None:
        _stt = WhisperSTT()
    return _stt


def get_tts():
    global _tts
    if _tts is None:
        _tts = KokoroTTS()
    return _tts


def voice_query(audio, history):
    """Transcribes a recording, answers it, and speaks the answer back."""
    history = history or []
    # Must match ask_outputs + voice_reply exactly.
    blank = (history, history, "", "", "_No audio captured._", *blank_metrics(), None)

    if audio is None:
        return blank

    started = time.perf_counter()
    sample_rate, array = audio
    samples = np.asarray(array)
    if samples.ndim > 1:                       # stereo -> mono
        samples = samples.mean(axis=1)
    if samples.dtype != np.int16:
        samples = float32_to_int16(samples.astype(np.float32))

    mono = resample_audio(int16_to_float32(samples), orig_sr=sample_rate, target_sr=16000)
    question = get_stt().transcribe(float32_to_int16(mono).tobytes())
    stt_ms = (time.perf_counter() - started) * 1000

    if not question.strip():
        return blank

    # Same endpoint as the text path, so voice turns get identical telemetry
    outputs = run_query(question, history)
    new_history = outputs[0]
    answer = new_history[-1]["content"] if new_history else ""

    spoken = sanitize_for_speech(answer)
    audio_out = None
    if spoken:
        out_rate, chunk = get_tts().synthesize_array(spoken)
        if len(chunk):
            audio_out = (out_rate, chunk)

    total_ms = (time.perf_counter() - started) * 1000
    print(f"[VOICE] q={question!r}")
    print(f"[VOICE] STT {stt_ms:.0f}ms | turn total {total_ms:.0f}ms")

    return (*outputs, audio_out)


def metric_card(value, label):
    return f"<div class='metric-box'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>"


def blank_metrics():
    return (
        metric_card("—", "Confidence"),
        metric_card("—", "Grounded"),
        metric_card("—", "Retrieval"),
        metric_card("—", "Total Latency"),
    )


def format_trace(trace):
    """Renders the PipelineTracer spans returned by the API."""
    lines = []
    for span in trace:
        lines.append(f"[{span['layer'].upper():<12}] {span['elapsed_ms']:>8.1f} ms  {span['details']}")
    return "\n".join(lines)


def format_sources(chunks):
    """Splits the injected citation header off each chunk for readable cards."""
    if not chunks:
        return "_No chunks passed the confidence gate._"

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        match = SOURCE_HEADER.match(chunk)
        if match:
            source, kind, number = parse_tag(match)
            locator = f"{kind.lower()} {number}"           # "page 12" for PDFs, "chunk 3" for text
            body = chunk[match.end():]
        else:
            source, locator, body = "unknown", "chunk -", chunk
        blocks.append(f"**[{i}] {source} · {locator}**\n\n{body.strip()}")
    return "\n\n---\n\n".join(blocks)


def fetch_status():
    try:
        data = requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception as e:
        return f"### API offline\n`{e}`"

    if data.get("ollama_status") == "online":
        return f"### Online · {data.get('generation_model', 'unknown model')}"
    return "### Ollama offline"


def fetch_corpus():
    try:
        rows = requests.get(f"{API_BASE}/corpus", timeout=30).json()
    except Exception as e:
        return [["API offline", 0, str(e)[:40]]]

    if not rows:
        return [["No documents in data/raw", 0, "Empty"]]
    return [[r["file"], r["chunks"], r["status"]] for r in rows]


def corpus_filenames():
    try:
        rows = requests.get(f"{API_BASE}/corpus", timeout=30).json()
    except Exception:
        return []
    return [r["file"] for r in rows if isinstance(r, dict) and r.get("chunks") is not None]


def refresh_corpus():
    """Table plus a refreshed delete picker."""
    return fetch_corpus(), gr.update(choices=corpus_filenames(), value=None)


def delete_document(filename):
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
        "Uploaded. Indexing runs in the background — hit Refresh in a moment.",
    )


def run_query(question, history):
    """Text turn: send prior turns so follow-up pronouns resolve, then update the transcript."""
    history = history or []

    if not question.strip():
        return history, history, "", "", "", *blank_metrics()

    try:
        data = requests.post(
            f"{API_BASE}/query",
            json={"query": question, "history": history[-HISTORY_TURNS:]},
            timeout=300,
        ).json()
    except Exception as e:
        return history, history, "", f"API error: {e}", "", *blank_metrics()

    if "detail" in data:
        return history, history, "", f"API error: {data['detail']}", "", *blank_metrics()

    answer = data.get("answer", "")
    trace = data.get("trace", [])
    retrieval_ms = sum(s["elapsed_ms"] for s in trace if s["layer"] in ("bm25", "vector", "fusion", "rerank"))

    grounded = data.get("grounded")
    grounded_label = {True: "PASS", False: "FAIL", None: "N/A"}[grounded]

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    rewritten = data.get("search_query", question)
    searched = "" if rewritten == question else f"_Searched as:_ **{rewritten}**"

    return (
        history,                              # state
        history,                              # transcript
        "",                                   # clear the input box
        format_trace(trace),
        format_sources(data.get("retrieved_context", [])),
        metric_card(f"{data.get('confidence_score', 0.0):.3f}", "Confidence"),
        metric_card(grounded_label, "Grounded"),
        metric_card(f"{retrieval_ms:.0f} ms", "Retrieval"),
        metric_card(f"{data.get('total_latency_ms', 0.0):.0f} ms", "Total"),
    )


def clear_conversation():
    return [], [], "", "", "_Ask a question to see the grounding chunks._", *blank_metrics()


with gr.Blocks(title="V-RAG Console", theme=gr.themes.Soft(), css=custom_css) as demo:

    with gr.Row(equal_height=True):
        gr.Markdown("## V-RAG Console\n*Local hybrid BM25 + vector retrieval with confidence gating*")
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
                        lines=2,
                    )
                    with gr.Row():
                        ask_btn = gr.Button("▶ Ask", variant="primary", scale=3)
                        clear_btn = gr.Button("Clear", scale=1)

                    with gr.Accordion("🎙️ Or speak your question", open=False):
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
                        voice_reply = gr.Audio(label="Spoken answer", autoplay=True, interactive=False)

                with gr.Column(scale=4):
                    with gr.Row():
                        m_conf = gr.HTML(metric_card("—", "Confidence"))
                        m_grounded = gr.HTML(metric_card("—", "Grounded"))
                    with gr.Row():
                        m_retrieval = gr.HTML(metric_card("—", "Retrieval"))
                        m_total = gr.HTML(metric_card("—", "Total"))

                    with gr.Accordion("📚 Retrieved sources", open=True):
                        sources_output = gr.Markdown("_Ask a question to see the grounding chunks._")

                    with gr.Accordion("🔬 Pipeline trace", open=False):
                        trace_output = gr.Textbox(
                            lines=9,
                            elem_id="trace_box",
                            show_label=False,
                            interactive=False,
                            placeholder="Timings appear here after a query.",
                        )

        # ------------------------------------------------------------- Corpus
        with gr.Tab("📂 Corpus"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="Add a PDF, TXT or MD file")
                    upload_status = gr.Markdown("")
                    refresh_btn = gr.Button("🔄 Refresh")

                    gr.Markdown("### Remove a document")
                    delete_picker = gr.Dropdown(
                        label="Document",
                        choices=[],
                        interactive=True,
                    )
                    delete_btn = gr.Button("🗑️ Delete from corpus", variant="stop")
                    gr.Markdown(
                        "_Deletes the file from `data/raw` and rebuilds both indexes "
                        "from what remains. This cannot be undone._"
                    )
                with gr.Column(scale=3):
                    corpus_table = gr.Dataframe(
                        headers=["File", "Chunks", "Status"],
                        datatype=["str", "number", "str"],
                        interactive=False,
                        label="Indexed documents",
                    )

    ask_outputs = [history_state, chatbot, query_input, trace_output, sources_output,
                   m_conf, m_grounded, m_retrieval, m_total]

    ask_btn.click(fn=run_query, inputs=[query_input, history_state], outputs=ask_outputs)
    query_input.submit(fn=run_query, inputs=[query_input, history_state], outputs=ask_outputs)
    clear_btn.click(fn=clear_conversation, outputs=ask_outputs)

    mic.stop_recording(fn=voice_query, inputs=[mic, history_state],
                       outputs=ask_outputs + [voice_reply])

    file_input.upload(fn=upload_document, inputs=[file_input],
                      outputs=[corpus_table, delete_picker, upload_status])
    refresh_btn.click(fn=refresh_corpus, outputs=[corpus_table, delete_picker])
    delete_btn.click(fn=delete_document, inputs=[delete_picker],
                     outputs=[corpus_table, delete_picker, upload_status])

    demo.load(fn=fetch_status, outputs=[status_md])
    demo.load(fn=refresh_corpus, outputs=[corpus_table, delete_picker])

if __name__ == "__main__":
    host = os.environ.get("VRAG_UI_HOST", "0.0.0.0")
    print("\nOpen http://localhost:7860 — microphone access is blocked on other hostnames.\n")
    demo.launch(server_name=host, server_port=7860)
