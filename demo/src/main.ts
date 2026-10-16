// The V-RAG live demo page (spec 0009): the dashboard's three tabs (Ask, Documents, Observability) in a browser.
// Before Load it fetches only itself, the sample index (chunks, manifest, suggested questions) and config.json;
// Load fetches the models, Pyodide and the sample vectors. Every piece of document text is set with textContent.
import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import { addTrace, listTraces, type Trace } from "./db";
import { inspectDevice, type DeviceInfo } from "./device";
import { Driver, type Event as CoreEvent } from "./driver";
import { WorkerRpc, type Progress } from "./rpc";

const BASE = import.meta.env.BASE_URL;
const HISTORY_TURNS = 6;
const TURN_CHARS = 2000;

type Pinned = { id: string; revision: string; dtype?: string; max_tokens?: number };
type AnswerModel = { f16: string; f32: string; vram_mb_f16: number; vram_mb_f32: number; download_mb: number };
type Config = {
  temperature: number; top_p: number; max_tokens: number;
  models: { embedder: Pinned; reranker: Pinned; answer_models: Record<"3b" | "1b", AnswerModel> };
  [key: string]: any;
};
type Doc = { name: string; title: string; publisher: string; pages: number | null; license: string; source_url: string };
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

function renderDocuments() {
  $("documents").replaceChildren(...state.docs.map(d => el("article", { className: "doc" },
    el("h3", { textContent: d.title }),
    el("p", { className: "hint", textContent: `${d.publisher} · ${d.license}` }),
    el("a", { href: d.source_url, textContent: "Source", rel: "noopener" }))));
  const count = (doc: string, kind?: string) => state.chunks.filter(c => c.doc === doc && (!kind || c.kind === kind)).length;
  $("corpus").querySelector("tbody")!.replaceChildren(...state.docs.map(d => el("tr", {},
    el("td", { textContent: d.name }), el("td", { textContent: String(d.pages ?? "") }),
    el("td", { textContent: String(count(d.name)) }), el("td", { textContent: String(count(d.name, "table")) }),
    el("td", { textContent: String(count(d.name, "figure")) }), el("td", { textContent: "Indexed" }))));
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
  return { size, id: state.device!.f16 ? record.f16 : record.f32 };
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
    $("model-hint").textContent = `Needs about ${(vram / 1000).toFixed(1)} GB of GPU memory. ${device.reason[0].toUpperCase()}${device.reason.slice(1)}.`;
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
  const pythonRow = progressRow("Python core");
  setStatus(`Loading ${choice.id}…`);

  const python = new WorkerRpc(new Worker(new URL("./workers/pyodide.worker.ts", import.meta.url), { type: "module" }));
  const models = new WorkerRpc(new Worker(new URL("./workers/models.worker.ts", import.meta.url), { type: "module" }));

  // WebLLM is imported only now, so the page itself stays small before Load (AC-1)
  const { CreateWebWorkerMLCEngine } = await import("@mlc-ai/web-llm");
  const answer = CreateWebWorkerMLCEngine(
    new Worker(new URL("./workers/llm.worker.ts", import.meta.url), { type: "module" }), choice.id,
    { initProgressCallback: (p) => answerRow.update(p.progress, p.text) },
  ).then(engine => { answerRow.update(1, "ready"); return engine; },
         error => { answerRow.fail(String(error)); throw error; });

  const search = models.call("load", { embedder: config.models.embedder, reranker: config.models.reranker },
    (p: Progress) => searchRow.update(p.value, p.step))
    .then(() => searchRow.update(1, "ready"), error => { searchRow.fail(String(error)); throw error; });

  const core = fetch(`${BASE}index/embeddings.f32`).then(r => r.arrayBuffer())
    .then(vectors => python.call<number>("load",
      { pythonUrl: `${BASE}py.zip`, config: JSON.stringify(config), chunks: state.chunksText, vectors },
      (p: Progress) => pythonRow.update(p.value, p.step), [vectors]))
    .then(count => pythonRow.update(1, `ready, ${count} sample chunks`), error => { pythonRow.fail(String(error)); throw error; });

  try {
    const [engine] = await Promise.all([answer, search, core]);
    state.driver = new Driver(python, models, engine as MLCEngineInterface, config);
    state.python = python;
    state.modelLabel = choice.id;
    $("setup").hidden = true;
    for (const id of ["question", "ask"]) $<HTMLButtonElement>(id).disabled = false;
    $<HTMLTextAreaElement>("question").placeholder = "Ask about the sample documents…";
    setStatus(`Ready · ${choice.size === "3b" ? "Llama 3.2 3B" : "Llama 3.2 1B"} on your GPU`);
    $<HTMLTextAreaElement>("question").focus();
    void renderObservability();
  } catch (error) {
    const outOfMemory = /memory|OOM|allocation/i.test(String(error));
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

function setCards(values: { confidence?: string; grounded?: string; retrieval?: string; total?: string }) {
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
    el("header", {}, el("span", { textContent: `${d.source} · ${d.page != null ? `p. ${d.page}` : "text"} · ${d.kind}` }),
      el("span", { textContent: `score ${Number(d.score).toFixed(2)}` })),
    el("p", { textContent: (chunks[i] ?? "").replace(/^\[Source:[^\]]*\]\s*/, "").slice(0, 320) }))));
}

async function ask(query: string) {
  if (!state.driver || state.busy || !query.trim()) return;
  state.busy = true;
  const conversation = $("conversation");
  const text = el("span", { className: "text", textContent: "" });
  const meta = el("div", { className: "meta" });
  const turn = el("li", { className: "turn assistant pending" }, text, meta);
  conversation.append(el("li", { className: "turn user", textContent: query }), turn);
  turn.scrollIntoView({ block: "end" });
  $<HTMLButtonElement>("ask").disabled = true;
  $("stop").hidden = false;
  setCards({ confidence: "…", grounded: "…", retrieval: "…", total: "…" });
  $("ask-trace").textContent = "Running…";

  const started = performance.now();
  const spans: SpanRow[] = [];
  const sentences: string[] = [];
  let done: CoreEvent | null = null;
  let errorText: string | null = null;
  let record: Record<string, any> | null = null;
  try {
    record = await state.driver.ask({ query, history: state.history.slice(-HISTORY_TURNS), docs: null, grounding: false }, {
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
          turn.scrollIntoView({ block: "end" });
        } else if (event.type === "done") {
          done = event;
          meta.replaceChildren(...(event.citations as Citation[]).map(c => el("span", { className: "cite", textContent: citationLabel(c) })),
            el("span", { textContent: `${(event.total_ms / 1000).toFixed(1)} s` }));
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
    if (spans.length) {
      $("ask-trace").textContent = `${formatTrace(spans)}\n\n[${"TOTAL".padEnd(12)}] ${totalMs.toFixed(1).padStart(8)} ms  `
        + `${usage.prompt_tokens ?? "?"} prompt and ${usage.output_tokens ?? "?"} generated tokens`;
    }
    await addTrace({
      asked_at: new Date().toISOString(), channel: "typed", query, search_query: record?.search_query ?? query,
      path: record?.path ?? (errorText ? "error" : "aborted"), answer: record?.answer ?? null,
      confidence: record?.confidence ?? null, grounding_checker: record?.grounding_checker ?? null,
      sentences_dropped: record?.answer ? (record?.dropped ?? []).length : null, total_ms: totalMs,
      first_sentence_ms: record?.first_sentence_ms ?? null, stt_ms: null, first_audio_ms: null, spoken_total_ms: null,
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

// --- the Observability tab (spec 0007's tab; numbers from the dashboard's own summarize, run in Pyodide) ---

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
  renderDocuments();
  renderSuggestions(suggested);
  $("obs-window").addEventListener("change", () => void renderObservability());
  $("obs-refresh").addEventListener("click", () => void renderObservability());
  $("tab-obs-btn").addEventListener("click", () => void renderObservability());
  void renderObservability();

  if (!device.webgpu) {
    $("setup").hidden = true;
    $("no-webgpu").hidden = false;
    renderPrecomputed(suggested);
    setStatus("WebGPU is not available in this browser");
    return;
  }
  renderModelPicker();
  $("load").addEventListener("click", () => void loadModels());

  const form = $<HTMLFormElement>("ask-form");
  const question = $<HTMLTextAreaElement>("question");
  form.addEventListener("submit", (e) => { e.preventDefault(); const q = question.value; question.value = ""; void ask(q); });
  question.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  $("stop").addEventListener("click", () => void state.driver?.stop());
  $("clear").addEventListener("click", () => { state.history = []; $("conversation").replaceChildren(); });
}

boot().catch(error => setStatus(`The demo could not start: ${String(error)}`));
