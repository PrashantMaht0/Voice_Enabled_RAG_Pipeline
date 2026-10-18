// The V-RAG live demo page: the dashboard's three tabs (Ask, Documents, Observability) in a browser.
// Before Load it fetches only itself, the sample index (chunks, manifest, suggested questions) and config.json;
// Load fetches the models, Pyodide and the sample vectors. Every piece of document text is set with textContent.
import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import { addTrace, getSetting, listTraces, putSetting, type Trace } from "./shared/db";
import { inspectDevice, type DeviceInfo } from "./models/device";
import { attachModels, detachModels, initDocuments, scopeDocs, type SampleDoc } from "./documents/documents";
import { Driver, type Event as CoreEvent } from "./pipeline/driver";
import { WorkerRpc, type Progress } from "./shared/rpc";
import type { UploadLimits } from "./documents/uploads";
import { MicBlocked, peakRms, QUIET_RMS, Recorder, Speaker } from "./voice/voice";

const BASE = import.meta.env.BASE_URL;
const HISTORY_TURNS = 6;
const TURN_CHARS = 2000;
const JUDGE_PREFIX_MARKER = "\n\nStatement:";     // where judge_messages starts the per sentence part

type Pinned = { id: string; revision: string; dtype?: string; max_tokens?: number };
type AnswerModel = { f16: string; f32: string; revision_f16: string; revision_f32: string; vram_mb_f16: number; vram_mb_f32: number;
                     download_mb: number };
type Config = {
  temperature: number; top_p: number; max_tokens: number; embedder_id: string; space: string; uploads: UploadLimits;
  judge_model: "3b" | "1b";
  models: { embedder: Pinned; reranker: Pinned; whisper: Pinned & { dtype_f16?: string };
            kokoro: { id: string; revision: string; dtype: string; voice: string }; answer_models: Record<"3b" | "1b", AnswerModel> };
  [key: string]: any;
};
type Doc = SampleDoc;
type Suggested = { question: string; kind: string; answer: string; citations: Citation[] };
type Citation = { source: string; page: number | null; chunk: number | null; kinds?: string[] };
type Chunk = { doc: string; kind: string };
type SpanRow = { layer: string; elapsed_ms: number; details: string };

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

const state = {
  config: null as Config | null,
  docs: [] as Doc[],
  chunks: [] as Chunk[],
  chunksText: "",
  device: null as DeviceInfo | null,
  driver: null as Driver | null,
  python: null as WorkerRpc | null,
  models: null as WorkerRpc | null,
  whisper: null as Promise<string> | null,
  speaker: null as Speaker | null,
  stopRequested: false,
  readyStatus: "",
  modelLabel: "",
  history: [] as { role: string; content: string }[],
  selectedTrace: null as number | null,
  busy: false,
};

async function json<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json() as Promise<T>;
}

function setStatus(text: string) { $("status").textContent = text; }

// --- tabs ---

function setupTabs() {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
  const select = (tab: HTMLButtonElement) => {
    for (const t of tabs) {
      const selected = t === tab;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
      $(t.getAttribute("aria-controls")!).hidden = !selected;
    }
    tab.focus();
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") select(tabs[(i + 1) % tabs.length]);
      if (e.key === "ArrowLeft") select(tabs[(i - 1 + tabs.length) % tabs.length]);
    });
  });
}

// --- static content ---

function citationLabel(c: Citation) {
  const where = c.page != null ? `p. ${c.page}` : `chunk ${c.chunk}`;
  const kinds = (c.kinds ?? []).filter(k => k !== "text");
  return `${c.source} · ${where}${kinds.length ? ` · ${kinds.join(", ")}` : ""}`;
}

function renderSuggestions(suggested: Suggested[]) {
  $("suggestions").replaceChildren(...suggested.map(s => {
    const chip = el("button", { type: "button", className: "chip" }, el("span", { className: "kind", textContent: s.kind }), s.question);
    chip.addEventListener("click", () => {
      const question = $<HTMLTextAreaElement>("question");
      question.value = s.question;
      if (!question.disabled) $<HTMLFormElement>("ask-form").requestSubmit();
    });
    return chip;
  }));
}

function renderPrecomputed(suggested: Suggested[]) {
  $("precomputed").replaceChildren(...suggested.map(s => el("article", {},
    el("span", { className: "badge", textContent: "precomputed with the local setup" }),
    el("h3", { textContent: s.question }),
    el("p", { textContent: s.answer.replace(/\s*\[Source:[^\]]*\]/g, "") }),
    el("div", { className: "meta" }, ...s.citations.map(c => el("span", { className: "cite", textContent: citationLabel(c) }))))));
}

// --- loading ---

function progressRow(label: string) {
  const bar = el("span");
  const pct = el("span", { textContent: "0%" });
  const detail = el("span", { className: "detail", textContent: "waiting" });
  $("progress").append(el("li", {}, el("span", { textContent: label }), el("span", { className: "bar" }, bar), pct, detail));
  return {
    update(value: number, text: string) {
      const percent = Math.max(0, Math.min(100, Math.round(value * 100)));
      bar.style.width = `${percent}%`;
      pct.textContent = `${percent}%`;
      detail.textContent = text;
    },
    fail(message: string) { detail.textContent = message; detail.className = "detail failed"; },
  };
}

function modelChoice() {
  const size = $<HTMLSelectElement>("model").value as "3b" | "1b";
  const record = state.config!.models.answer_models[size];
  const f16 = state.device!.f16;
  return { size, id: f16 ? record.f16 : record.f32, revision: f16 ? record.revision_f16 : record.revision_f32 };
}

function renderModelPicker() {
  const select = $<HTMLSelectElement>("model");
  const models = state.config!.models.answer_models;
  const device = state.device!;
  select.replaceChildren(...(["3b", "1b"] as const).map(size => {
    const gb = (models[size].download_mb / 1000).toFixed(1);
    const name = size === "3b" ? "Llama 3.2 3B, better answers" : "Llama 3.2 1B, faster and lighter";
    return el("option", { value: size, textContent: `${name} (${gb} GB download)${size === device.suggested ? " · suggested" : ""}` });
  }));
  select.value = device.suggested;
  const updateHint = () => {
    const record = models[select.value as "3b" | "1b"];
    const vram = device.f16 ? record.vram_mb_f16 : record.vram_mb_f32;
    $("model-hint").textContent = `${(record.download_mb / 1000).toFixed(1)} GB download, needs about ${(vram / 1000).toFixed(1)} GB of GPU `
      + `memory. ${device.reason[0].toUpperCase()}${device.reason.slice(1)}.`;
  };
  select.addEventListener("change", updateHint);
  updateHint();
}

async function loadModels() {
  const load = $<HTMLButtonElement>("load");
  const select = $<HTMLSelectElement>("model");
  load.disabled = select.disabled = true;
  load.textContent = "Loading…";
  $("progress").replaceChildren();
  const config = state.config!;
  const choice = modelChoice();
  const answerRow = progressRow("Answer model");
  const searchRow = progressRow("Search models");
  const groundingRow = progressRow("Grounding judge");
  const pythonRow = progressRow("Python core");
  setStatus(`Loading ${choice.id}…`);

  // Every worker this attempt starts, so a failed attempt frees its GPU memory before the visitor tries again
  const workers: Worker[] = [];
  // Each new Worker(new URL(...)) stays written out in full: that is the form Vite recognises and bundles
  const track = (worker: Worker) => { workers.push(worker); return worker; };
  let failed = false;
  const python = new WorkerRpc(track(new Worker(new URL("./pipeline/pyodide.worker.ts", import.meta.url), { type: "module" })));
  const models = new WorkerRpc(track(new Worker(new URL("./models/models.worker.ts", import.meta.url), { type: "module" })));

  // WebLLM is imported only now, so the page itself stays small before Load
  const { CreateWebWorkerMLCEngine, prebuiltAppConfig } = await import("@mlc-ai/web-llm");
  // WebLLM's own records for the builds (their GPU code and settings), downloading the weights from pinned commits
  const judgeRecord = config.models.answer_models[config.judge_model];
  const judge = { id: state.device!.f16 ? judgeRecord.f16 : judgeRecord.f32,
                  revision: state.device!.f16 ? judgeRecord.revision_f16 : judgeRecord.revision_f32 };
  const pinned = (id: string, revision: string) => ({ ...prebuiltAppConfig.model_list.find(m => m.model_id === id)!,
    model: `https://huggingface.co/mlc-ai/${id}/resolve/${revision}/` });
  const appConfig = { ...prebuiltAppConfig, model_list: [pinned(choice.id, choice.revision)]
    .concat(judge.id === choice.id ? [] : [pinned(judge.id, judge.revision)]) };
  const answer = CreateWebWorkerMLCEngine(
    track(new Worker(new URL("./models/llm.worker.ts", import.meta.url), { type: "module" })), choice.id,
    { appConfig, initProgressCallback: (p) => answerRow.update(p.progress, p.text) },
  ).then(engine => { answerRow.update(1, "ready"); return engine; },
         error => { answerRow.fail(String(error)); throw error; });

  const search = models.call("load", { embedder: config.models.embedder, reranker: config.models.reranker },
    (p: Progress) => searchRow.update(p.value, p.step))
    .then(() => searchRow.update(1, "ready"), error => { searchRow.fail(String(error)); throw error; });

  // The grounding judge (Llama 3.2 1B, the server's default checker) runs in its own engine, so it can check a
  // sentence while the answer model's stream waits; it loads after the answer model and may fail without failing
  // Load: answers are then shown unchecked, with the reason
  let judgeEngine: MLCEngineInterface | null = null;
  let checkProblem: string | null = "the grounding judge is still loading";
  const setJudge = (engine: MLCEngineInterface | null, problem: string | null) => {
    judgeEngine = engine; checkProblem = problem;
    if (state.driver) { state.driver.judge = engine; state.driver.checkProblem = problem; }
  };
  const judgeWorker = () => {
    const worker = track(new Worker(new URL("./models/llm.worker.ts", import.meta.url), { type: "module" }));
    worker.postMessage({ prefixCache: JUDGE_PREFIX_MARKER });
    return worker;
  };
  answer.then(() => {
    if (failed) throw new Error("Load failed");
    return CreateWebWorkerMLCEngine(judgeWorker(), judge.id,
      { appConfig, initProgressCallback: (p) => groundingRow.update(p.progress, p.text) });
  })
    .then(engine => { groundingRow.update(1, `ready, ${judge.id.replace(/-Instruct.*/, "")}`); setJudge(engine as MLCEngineInterface, null); },
          error => { groundingRow.fail(String(error)); setJudge(null, `the grounding judge did not load (${String(error).slice(0, 120)})`); });

  const core = fetch(`${BASE}index/embeddings.f32`).then(r => r.arrayBuffer())
    .then(vectors => python.call<number>("load",
      { pythonUrl: `${BASE}py.zip`, config: JSON.stringify(config), chunks: state.chunksText, vectors },
      (p: Progress) => pythonRow.update(p.value, p.step), [vectors]))
    .then(count => pythonRow.update(1, `ready, ${count} sample chunks`), error => { pythonRow.fail(String(error)); throw error; });

  // Uploads index as soon as the embedder and the Python index are up, without waiting for the answer model
  Promise.all([search, core]).then(() => { if (!failed) return attachModels(models, python); }).catch(() => undefined);

  try {
    const [engine] = await Promise.all([answer, search, core]);
    state.driver = new Driver(python, models, engine as MLCEngineInterface, config);
    state.driver.judge = judgeEngine;
    state.driver.checkProblem = checkProblem;
    state.python = python;
    state.models = models;
    state.modelLabel = choice.id;
    $("setup").hidden = true;
    for (const id of ["question", "ask", "mic"]) $<HTMLButtonElement>(id).disabled = false;
    $<HTMLTextAreaElement>("question").placeholder = "Ask about the documents…";
    state.readyStatus = `Ready · ${choice.size === "3b" ? "Llama 3.2 3B" : "Llama 3.2 1B"} on your GPU`;
    setStatus(state.readyStatus);
    $<HTMLTextAreaElement>("question").focus();
    void renderObservability();
  } catch (error) {
    failed = true;
    detachModels();
    for (const worker of workers) worker.terminate();
    // WebGPU reports running out of GPU memory as an allocation error or as a lost device
    const outOfMemory = /memory|OOM|allocation|device (was )?lost/i.test(String(error));
    load.disabled = select.disabled = false;
    load.textContent = "Try again";
    if (outOfMemory && choice.size === "3b") {
      select.value = "1b";
      select.dispatchEvent(new Event("change"));
      setStatus("The 3B model did not fit in this GPU's memory. The 1B model is selected; press Try again.");
    } else {
      setStatus("Loading failed. Check your connection and press Try again.");
    }
  }
}

// --- the Ask tab ---

const RETRIEVAL_LAYERS = ["embed", "bm25", "vector", "fusion", "rerank"];   // the dense search includes embedding

function formatTrace(spans: SpanRow[]) {
  return spans.map(s => `[${s.layer.toUpperCase().padEnd(12)}] ${s.elapsed_ms.toFixed(1).padStart(8)} ms  ${s.details}`).join("\n");
}

function setCards(values: { confidence?: string; grounded?: string; retrieval?: string; total?: string; audio?: string }) {
  if (values.audio !== undefined) $("m-audio").textContent = values.audio;
  if (values.confidence !== undefined) $("m-confidence").textContent = values.confidence;
  if (values.grounded !== undefined) $("m-grounded").textContent = values.grounded;
  if (values.retrieval !== undefined) $("m-retrieval").textContent = values.retrieval;
  if (values.total !== undefined) $("m-total").textContent = values.total;
}

function renderSources(event: CoreEvent) {
  const box = $("sources");
  const details: any[] = event.retrieved ?? [];
  const chunks: string[] = event.retrieved_context ?? [];
  if (!details.length) {
    box.replaceChildren(el("p", { className: "empty", textContent: "No passage scored above the confidence gate, so the answer model was not asked." }));
    return;
  }
  box.replaceChildren(...details.map((d, i) => el("article", { className: "source" },
    el("header", {}, el("span", { textContent: [d.source, d.page != null ? `p. ${d.page}` : null, d.kind].filter(Boolean).join(" · ") }),
      el("span", { textContent: `score ${Number(d.score).toFixed(2)}` })),
    el("p", { textContent: (chunks[i] ?? "").replace(/^\[Source:[^\]]*\]\s*/, "").slice(0, 320) }))));
}

type VoiceTurn = { started: number; sttMs: number };

/** The speaker for a voice turn, loading Kokoro on the first spoken answer (browser speech on phones or failure). */
const voiceLoading: Record<string, number> = {};
function showVoiceLoading(part: string, value: number) {
  voiceLoading[part] = value;
  const open = Object.entries(voiceLoading).filter(([, v]) => v < 1);
  setStatus(open.length ? `Loading ${open.map(([k, v]) => `${k} (${Math.round(v * 100)}%)`).join(" and ")}…` : state.readyStatus);
}

function speakerFor(started: number): Promise<Speaker> {
  return Speaker.create(state.device!, state.config!.models.kokoro, started,
    (value) => showVoiceLoading("the voice", value),
    (reason) => console.warn(`[demo] Kokoro did not load, using the browser's speech: ${reason}`))
    .then(speaker => { setStatus(state.readyStatus + (speaker.usesKokoro ? "" : " · browser speech")); return speaker; });
}

async function ask(query: string, voice: VoiceTurn | null = null) {
  if (!state.driver || state.busy || !query.trim()) return;
  const conversation = $("conversation");
  const docs = scopeDocs($<HTMLSelectElement>("scope").value);
  if (docs && !docs.length) {
    conversation.append(el("li", { className: "turn user", textContent: query }), el("li", { className: "turn assistant error",
      textContent: "None of your documents is ready yet. Add one in Documents, or choose All documents." }));
    return;
  }
  state.busy = true;
  const text = el("span", { className: "text", textContent: "" });
  const meta = el("div", { className: "meta" });
  const turn = el("li", { className: "turn assistant pending" }, text, meta);
  conversation.append(el("li", { className: "turn user", textContent: query }), turn);
  turn.scrollIntoView({ block: "end" });
  $<HTMLButtonElement>("ask").disabled = true;
  $("stop").hidden = false;
  setCards({ confidence: "…", grounded: "…", retrieval: "…", total: "…", audio: voice ? "…" : "—" });
  $("ask-trace").textContent = "Running…";

  const started = performance.now();
  // Voice: each approved sentence goes through the server's speech cleanup (in Pyodide), then to the speaker, in order
  const speaking = voice ? speakerFor(voice.started) : null;
  let speech: Promise<void> = Promise.resolve();
  state.stopRequested = false;
  if (speaking) void speaking.then(s => { state.speaker = s; if (state.stopRequested) s.stop(); });
  const spans: SpanRow[] = [];
  const sentences: string[] = [];
  let done: CoreEvent | null = null;
  let errorText: string | null = null;
  let record: Record<string, any> | null = null;
  try {
    record = await state.driver.ask({ query, history: state.history.slice(-HISTORY_TURNS), docs,
      grounding: $<HTMLInputElement>("grounding").checked, channel: voice ? "voice" : "typed" }, {
      onSpan: (span) => {
        spans.push({ layer: span.layer, elapsed_ms: span.ms, details: span.details });
        $("ask-trace").textContent = formatTrace(spans);
      },
      onEvent: (event) => {
        if (event.type === "meta") {
          renderSources(event);
          setCards({ confidence: Number(event.confidence_score ?? 0).toFixed(3) });
        } else if (event.type === "sentence") {
          sentences.push(event.text);
          text.textContent = sentences.join(" ");
          if (speaking) {
            const sentence = event.text as string;
            speech = speech.then(async () => {
              const pieces = JSON.parse(await state.python!.call<string>("speech", { sentence })) as string[];
              (await speaking).say(pieces);
            }).catch(() => undefined);
          }
          turn.scrollIntoView({ block: "end" });
        } else if (event.type === "done") {
          done = event;
          meta.replaceChildren(...(event.citations as Citation[]).map(c => el("span", { className: "cite", textContent: citationLabel(c) })),
            el("span", { textContent: `${(event.total_ms / 1000).toFixed(1)} s` }));
          const removed: string[] = event.dropped_sentences ?? [];
          if (removed.length) turn.append(el("details", { className: "removed" },
            el("summary", { textContent: `${removed.length} sentence${removed.length === 1 ? "" : "s"} removed: not supported by the documents` }),
            el("ul", {}, ...removed.map(r => el("li", { textContent: r })))));
        } else if (event.type === "error") {
          errorText = event.detail;
          turn.classList.add("error");
          text.textContent = event.detail;
        }
      },
    });
    if (!done && !errorText && !sentences.length) text.textContent = "Stopped.";
  } catch (error) {
    errorText = String(error);
    turn.classList.add("error");
    text.textContent = `Something went wrong: ${errorText}`;
  } finally {
    turn.classList.remove("pending");
    const totalMs = performance.now() - started;
    const finished = done as CoreEvent | null;
    const dropped: string[] = finished?.dropped_sentences ?? [];
    const grounded = !finished ? "N/A" : finished.grounded === true ? "PASS"
      : finished.grounded === false ? `FAIL (${dropped.length} removed)` : "Unchecked";
    const retrievalMs = spans.filter(s => RETRIEVAL_LAYERS.includes(s.layer)).reduce((sum, s) => sum + s.elapsed_ms, 0);
    setCards({ grounded, retrieval: `${retrievalMs.toFixed(0)} ms`, total: finished ? `${finished.total_ms.toFixed(0)} ms` : "—" });
    if (record?.confidence == null && !finished) setCards({ confidence: "—" });
    const usage = state.driver.usage;
    let firstAudioMs: number | null = null, spokenTotalMs: number | null = null;
    if (speaking) {                                  // Stop stays available until the answer has been spoken
      $<HTMLButtonElement>("ask").disabled = true;
      const speaker = await speaking;
      await speech;
      await speaker.finished();
      firstAudioMs = speaker.firstAudioMs;
      spokenTotalMs = speaker.lastAudioMs;
      setCards({ audio: firstAudioMs != null ? `${Math.round(firstAudioMs)} ms` : "—" });
      state.speaker = null;
    }
    if (spans.length) {
      $("ask-trace").textContent = `${formatTrace(spans)}\n\n[${"TOTAL".padEnd(12)}] ${totalMs.toFixed(1).padStart(8)} ms  `
        + `${usage.prompt_tokens ?? "?"} prompt and ${usage.output_tokens ?? "?"} generated tokens`;
    }
    await addTrace({
      asked_at: new Date().toISOString(), channel: voice ? "voice" : "typed", query, search_query: record?.search_query ?? query,
      path: record?.path ?? (errorText ? "error" : "aborted"), answer: record?.answer ?? null,
      confidence: record?.confidence ?? null, grounding_checker: record?.grounding_checker ?? null,
      sentences_dropped: record?.answer ? (record?.dropped ?? []).length : null, total_ms: totalMs,
      first_sentence_ms: record?.first_sentence_ms ?? null, stt_ms: voice?.sttMs ?? null, first_audio_ms: firstAudioMs, spoken_total_ms: spokenTotalMs,
      prompt_tokens: usage.prompt_tokens, output_tokens: usage.output_tokens, model: state.modelLabel,
      spans, retrieved: record?.retrieved ?? [], dropped_sentences: record?.dropped ?? [], error: errorText,
    }).catch(() => undefined);
    void renderObservability();
    if (finished) {
      state.history.push({ role: "user", content: query.slice(0, TURN_CHARS) },
                         { role: "assistant", content: String(finished.answer).slice(0, TURN_CHARS) });
      state.history = state.history.slice(-HISTORY_TURNS);
    }
    state.busy = false;
    $<HTMLButtonElement>("ask").disabled = false;
    $("stop").hidden = true;
  }
}

// --- the mic ---

const recorder = new Recorder();
const NOTHING_HEARD = "I did not hear a question. Press Speak, ask your question, then press it again.";

function voiceNotice(text: string) {
  $("conversation").append(el("li", { className: "turn assistant error", textContent: text }));
}

function resetMic() {
  const mic = $<HTMLButtonElement>("mic");
  mic.textContent = "Speak";
  mic.setAttribute("aria-pressed", "false");
  mic.classList.remove("recording");
}

let micStarting = false;

async function toggleMic() {
  if (micStarting) return;                       // a second press while the browser asks for the microphone
  if (recorder.recording) return finishRecording();
  if (state.busy || !state.models) return;
  const mic = $<HTMLButtonElement>("mic");
  micStarting = true;
  try {
    await recorder.start((seconds) => { mic.textContent = `Stop recording (${Math.floor(seconds)} s)`; }, () => void finishRecording());
  } catch (error) {
    voiceNotice(error instanceof MicBlocked
      ? "The microphone is blocked for this page. Allow it from the icon at the left of the address bar (site settings), then press Speak again."
      : `The microphone could not start: ${String((error as Error)?.message ?? error)}`);
    return;
  } finally {
    micStarting = false;
  }
  mic.textContent = "Stop recording (0 s)";
  mic.setAttribute("aria-pressed", "true");
  mic.classList.add("recording");
  // Whisper and the voice both start loading on the first mic press, while the visitor speaks, so the first spoken
  // answer does not wait for them one after the other
  state.whisper ??= state.models.call<string>("load_whisper", { pin: state.config!.models.whisper, f16: state.device!.f16 },
    (p: Progress) => showVoiceLoading("speech recognition", p.value))
    .then(dtype => { showVoiceLoading("speech recognition", 1); return dtype; },
          error => { state.whisper = null; showVoiceLoading("speech recognition", 1); throw error; });
  Speaker.warm(state.device!, state.config!.models.kokoro, (value) => showVoiceLoading("the voice", value));
}

async function finishRecording() {
  if (!recorder.recording) return;
  const started = performance.now();             // as the dashboard: before resampling and speech recognition
  resetMic();
  const mic = $<HTMLButtonElement>("mic");
  mic.disabled = true;
  try {
    const samples = await recorder.stop();
    if (peakRms(samples) < QUIET_RMS) { voiceNotice(NOTHING_HEARD); return; }
    mic.textContent = "Transcribing…";
    await state.whisper;
    const transcript = await state.models!.call<string>("transcribe", { audio: samples.buffer }, undefined, [samples.buffer]);
    const sttMs = performance.now() - started;
    if (!transcript || await state.python!.call<boolean>("heard_nothing", { transcript })) { voiceNotice(NOTHING_HEARD); return; }
    resetMic();
    mic.disabled = false;
    await ask(transcript, { started, sttMs });
  } catch (error) {
    voiceNotice(`Speech recognition failed: ${String((error as Error)?.message ?? error).slice(0, 160)}`);
  } finally {
    resetMic();
    mic.disabled = false;
  }
}

// --- the Observability tab (numbers from the dashboard's own summarize, run in Pyodide) ---

const WINDOW_MS: Record<string, number | null> = { "1h": 3600e3, "24h": 86400e3, "7d": 7 * 86400e3, all: null };
const STAGE_NAMES: Record<string, string> = {
  rewrite: "Rewrite", bm25: "Keyword search", vector: "Vector search", fusion: "Fusion", rerank: "Rerank",
  generation: "Generation", grounding: "Grounding check", first_sentence_ms: "First sentence", total_ms: "Total",
  stt_ms: "Speech recognition (voice)", first_audio_ms: "First audio (voice)", spoken_total_ms: "Spoken total (voice)",
};
const percent = (v: number | null) => v == null ? "—" : `${Math.round(v * 100)}%`;
const formatMs = (v: number | null) => {
  if (v == null) return "—";
  const ms = Math.round(v);
  return ms < 1000 ? `${ms.toLocaleString()} ms` : `${(ms / 1000).toFixed(1)} s`;
};
const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
const errorRate = (errors: number, asked: number) => {
  if (!asked) return "—";
  const value = errors / asked * 100;
  return errors && Math.round(value * 10) === 0 ? "<0.1%" : `${value.toFixed(1)}%`;
};

function card(value: string, label: string, note?: string | null) {
  return el("div", { className: "metric-box" }, el("div", { className: "metric-value", textContent: value }),
    el("div", { className: "metric-label", textContent: label }),
    ...(note ? [el("div", { className: "metric-note", textContent: note })] : []));
}

function traceDetail(t: Trace) {
  const lines = [t.query, `${t.asked_at.slice(0, 19).replace("T", " ")} · ${t.channel} · path ${t.path} · total `
    + `${Math.round(t.total_ms ?? 0)} ms${t.confidence != null ? ` · confidence ${t.confidence.toFixed(3)}` : ""}`];
  lines.push("", t.prompt_tokens == null ? "Tokens: not recorded"
    : `Tokens: ${t.prompt_tokens.toLocaleString()} prompt + ${(t.output_tokens ?? 0).toLocaleString()} generated`);
  lines.push("", "Stages", ...t.spans.map(s => `  ${s.layer.padEnd(11)} ${String(Math.round(s.elapsed_ms)).padStart(6)} ms  ${s.details}`));
  if (t.retrieved?.length) lines.push("", "Retrieved sources", ...t.retrieved.map((d: any) => `  ${d.source} · p.${d.page} · ${d.kind} · score ${d.score}`));
  if (t.dropped_sentences?.length) lines.push("", "Removed sentences (not supported by your documents)", ...t.dropped_sentences.map(d => `  ${d}`));
  lines.push("", "Answer", t.answer ?? "none");
  if (t.error) lines.push("", `Error: ${t.error}`);
  return lines.join("\n");
}

async function renderObservability() {
  const windowMs = WINDOW_MS[$<HTMLSelectElement>("obs-window").value];
  const all = await listTraces().catch(() => [] as Trace[]);
  const traces = windowMs == null ? all : all.filter(t => Date.now() - Date.parse(t.asked_at) <= windowMs);
  $("obs-empty").hidden = traces.length > 0;

  const body = $("obs-recent").querySelector("tbody")!;
  body.replaceChildren(...traces.map(t => {
    const row = el("tr", { tabIndex: 0 }, el("td", { textContent: t.asked_at.slice(0, 19).replace("T", " ") }),
      el("td", { textContent: t.channel }), el("td", { textContent: t.query }),
      el("td", { textContent: t.path === "answered" ? "yes" : "no" }));
    const open = () => {
      state.selectedTrace = t.id ?? null;
      $("obs-trace").textContent = traceDetail(t);
      for (const r of body.children) r.classList.toggle("selected", r === row);
    };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    if (t.id === state.selectedTrace) row.classList.add("selected");
    return row;
  }));

  const cards = $("obs-cards");
  const stages = $("obs-stages").querySelector("tbody")!;
  if (!state.python) {
    cards.replaceChildren(el("p", { className: "empty wide", textContent:
      "The cards are computed by the pipeline's own Python code, which loads with the models. Press Load models to see them." }));
    stages.replaceChildren();
    return;
  }
  const spans = traces.flatMap(t => t.spans.map(s => ({ trace_id: t.id, layer: s.layer, elapsed_ms: s.elapsed_ms })));
  const m = JSON.parse(await state.python.call<string>("metrics", { rows: JSON.stringify(traces), spans: JSON.stringify(spans) }));
  const c = m.counts, r = m.rates, tok = m.tokens, lat = m.latency;
  const refused = c.refused_gate + c.refused_grounding + c.refused_model + c.refused_empty;
  const note = (rate: number | null, text: string) => rate != null ? text : null;
  cards.replaceChildren(
    card(c.asked.toLocaleString(), "Questions"),
    card(c.answered.toLocaleString(), "Answered"),
    card(percent(r.retrieval_pass_rate), "Retrieval quality", note(r.retrieval_pass_rate, `${m.retrieval.passed} of ${m.retrieval.reached} passed the gate`)),
    card(percent(r.grounding_pass_rate), "Generation quality", note(r.grounding_pass_rate, `${m.grounding.passed} of ${m.grounding.checked} checked answers had nothing removed`)),
    card(percent(r.refusal_rate), "Refusal rate", note(r.refusal_rate, `${refused} refused`)),
    card(formatMs(lat.p50), "p50 latency", note(lat.p50, `over ${lat.n} answered`)),
    card(formatMs(lat.p99), "p99 latency", note(lat.p99, `over ${lat.n} answered`)),
    card(errorRate(c.errors, c.asked), "Error rate", c.asked ? plural(c.errors, "error") : null),
    card(tok.total.toLocaleString(), "Tokens used", tok.asked
      ? `${tok.per_question != null ? `${tok.per_question.toLocaleString()} per question · ` : ""}${tok.counted} of ${tok.asked} counted` : null),
    card(c.cache_hits.toLocaleString(), "Cache hits", "the demo keeps no answer cache"));
  const fmt = (v: number | null) => v == null ? "—" : String(Math.round(v));
  stages.replaceChildren(...Object.entries({ ...m.stages, ...m.voice } as Record<string, any>).filter(([, v]) => v.n)
    .map(([k, v]) => el("tr", {}, el("td", { textContent: STAGE_NAMES[k] ?? k }), el("td", { textContent: fmt(v.p50) }),
      el("td", { textContent: fmt(v.p99) }), el("td", { textContent: String(v.n) }))));
}

// --- boot ---

async function boot() {
  setupTabs();
  const [config, manifest, suggested, chunksText, device] = await Promise.all([
    json<Config>("config.json"), json<{ documents: Doc[] }>("index/manifest.json"), json<Suggested[]>("index/suggested.json"),
    fetch(`${BASE}index/chunks.json`).then(r => r.text()), inspectDevice()]);
  Object.assign(state, { config, docs: manifest.documents, device, chunksText, chunks: JSON.parse(chunksText) });
  await initDocuments({ config, sampleDocs: manifest.documents, sampleChunks: state.chunks,
                        onChange: () => void renderObservability() });
  renderSuggestions(suggested);
  const groundingSwitch = $<HTMLInputElement>("grounding");
  groundingSwitch.checked = await getSetting("grounding", true);
  groundingSwitch.addEventListener("change", () => void putSetting("grounding", groundingSwitch.checked));
  const scope = $<HTMLSelectElement>("scope");
  scope.value = await getSetting("scope", "all");
  scope.addEventListener("change", () => void putSetting("scope", scope.value));
  $("obs-window").addEventListener("change", () => void renderObservability());
  $("obs-refresh").addEventListener("click", () => void renderObservability());
  $("tab-obs-btn").addEventListener("click", () => void renderObservability());
  void renderObservability();

  // ?webgpu=off previews the page a browser without WebGPU sees
  if (!device.webgpu || new URLSearchParams(location.search).get("webgpu") === "off") {
    $("setup").hidden = true;
    $("no-webgpu").hidden = false;
    renderPrecomputed(suggested);
    $<HTMLTextAreaElement>("question").placeholder = "This browser cannot run the models. The suggested questions are answered above.";
    setStatus("WebGPU is not available in this browser");
    return;
  }
  renderModelPicker();
  $("load").addEventListener("click", () => void loadModels());

  const form = $<HTMLFormElement>("ask-form");
  const question = $<HTMLTextAreaElement>("question");
  form.addEventListener("submit", (e) => { e.preventDefault(); const q = question.value; question.value = ""; void ask(q); });
  question.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  $("stop").addEventListener("click", () => {        // one Stop ends generation and speech
    state.stopRequested = true;
    state.speaker?.stop();
    void state.driver?.stop();
  });
  $("mic").addEventListener("click", () => void toggleMic());
  $("clear").addEventListener("click", () => { state.history = []; $("conversation").replaceChildren(); });
}

boot().catch(error => setStatus(`The demo could not start: ${String(error)}`));
