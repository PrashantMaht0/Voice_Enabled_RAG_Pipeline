// The V-RAG live demo page (spec 0009). Before Load it fetches only itself, the sample manifest, the suggested
// questions and config.json; Load fetches the models, Pyodide and the sample index. Every piece of document text is
// set with textContent, never as HTML.
import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import { inspectDevice, type DeviceInfo } from "./device";
import { Driver, type Event as CoreEvent, type Span } from "./driver";
import { WorkerRpc, type Progress } from "./rpc";

const BASE = import.meta.env.BASE_URL;
const HISTORY_TURNS = 6;
const TURN_CHARS = 2000;
const KEEP_TRACES = 50;

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
type Trace = { asked_at: string; query: string; search_query: string; path: string; spans: Span[]; total_ms: number;
  model: string; prompt_tokens: number | null; output_tokens: number | null };

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

const state = {
  config: null as Config | null,
  docs: [] as Doc[],
  device: null as DeviceInfo | null,
  driver: null as Driver | null,
  engine: null as MLCEngineInterface | null,
  modelLabel: "",
  history: [] as { role: string; content: string }[],
  traces: [] as Trace[],
  busy: false,
};

async function json<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json() as Promise<T>;
}

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
  const grid = $("documents");
  grid.replaceChildren(...state.docs.map(d => el("article", { className: "panel" },
    el("h2", { textContent: d.title }),
    el("p", { className: "hint", textContent: d.publisher }),
    el("dl", {},
      el("dt", { textContent: "File" }), el("dd", { textContent: d.name }),
      el("dt", { textContent: "Pages" }), el("dd", { textContent: String(d.pages ?? "") }),
      el("dt", { textContent: "License" }), el("dd", { textContent: d.license }),
      el("dt", { textContent: "Source" }), el("dd", {}, el("a", { href: d.source_url, textContent: d.source_url, rel: "noopener" }))))));
}

function renderSuggestions(suggested: Suggested[]) {
  const box = $("suggestions");
  box.replaceChildren(...suggested.map(s => {
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

async function renderResults() {
  try {
    const results = await json<{ corpus_note: string; rows: any[] }>("results.json");
    $("results-note").textContent = results.corpus_note;
    const head = ["Setup", "Tables right", "Figures right", "Answered", "Trick questions refused", "Faithfulness", "Typed p50 (p95)", "Spoken p50"];
    const table = el("table", { className: "results-table" },
      el("thead", {}, el("tr", {}, ...head.map(h => el("th", { scope: "col", textContent: h })))),
      el("tbody", {}, ...results.rows.map(r => {
        const row = el("tr", {},
          el("td", { textContent: r.setup }),
          el("td", { textContent: `${r.tables_right} of ${r.tables_total}` }),
          el("td", { textContent: `${r.figures_right} of ${r.figures_total}` }),
          el("td", { textContent: `${r.answered} of ${r.answered_total}` }),
          el("td", { textContent: `${r.refused} of ${r.refused_total}` }),
          el("td", { textContent: r.faithfulness == null ? "not run" : String(r.faithfulness) }),
          el("td", { textContent: `${r.typed_p50_s} s (${r.typed_p95_s} s)` }),
          el("td", { textContent: `${r.spoken_p50_s} s` }));
        if (r.key === "A7") row.className = "chosen";
        return row;
      })));
    $("results").replaceChildren(table);
  } catch {
    $("results").replaceChildren(el("p", { className: "empty", textContent: "Results are not available in this build." }));
  }
}

// --- loading ---

function progressRow(label: string) {
  const bar = el("span");
  const pct = el("span", { textContent: "0%" });
  const detail = el("span", { className: "detail", textContent: "waiting" });
  const row = el("li", {}, el("span", { textContent: label }), el("span", { className: "bar" }, bar), pct, detail);
  $("progress").append(row);
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

  const core = Promise.all([
    fetch(`${BASE}index/chunks.json`).then(r => r.text()),
    fetch(`${BASE}index/embeddings.f32`).then(r => r.arrayBuffer()),
  ]).then(([chunks, vectors]) => python.call<number>("load",
    { pythonUrl: `${BASE}py.zip`, config: JSON.stringify(config), chunks, vectors },
    (p: Progress) => pythonRow.update(p.value, p.step), [vectors]))
    .then(count => pythonRow.update(1, `ready, ${count} sample chunks`), error => { pythonRow.fail(String(error)); throw error; });

  try {
    const [engine] = await Promise.all([answer, search, core]);
    state.engine = engine;
    state.driver = new Driver(python, models, engine, config);
    state.modelLabel = choice.id;
    $("setup").hidden = true;
    for (const id of ["question", "ask"]) $<HTMLButtonElement>(id).disabled = false;
    $<HTMLTextAreaElement>("question").placeholder = "Ask about the sample documents…";
    setStatus(`Ready · ${choice.size === "3b" ? "Llama 3.2 3B" : "Llama 3.2 1B"} on your GPU`);
    $<HTMLTextAreaElement>("question").focus();
  } catch (error) {
    const message = String(error);
    const outOfMemory = /memory|OOM|allocation/i.test(message);
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

function setStatus(text: string) { $("status").textContent = text; }

// --- asking ---

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
  const userTurn = el("li", { className: "turn user", textContent: query });
  const text = el("span", { className: "text", textContent: "" });
  const meta = el("div", { className: "meta" });
  const turn = el("li", { className: "turn assistant pending" }, text, meta);
  conversation.append(userTurn, turn);
  turn.scrollIntoView({ block: "end" });
  $<HTMLButtonElement>("ask").disabled = true;
  $("stop").hidden = false;

  const started = performance.now();
  const trace: Trace = { asked_at: new Date().toISOString(), query, search_query: query, path: "error", spans: [],
    total_ms: 0, model: state.modelLabel, prompt_tokens: null, output_tokens: null };
  let answer = "";
  const sentences: string[] = [];
  try {
    await state.driver.ask({ query, history: state.history.slice(-HISTORY_TURNS), docs: null, grounding: false }, {
      onSpan: (span) => trace.spans.push(span),
      onEvent: (event) => {
        if (event.type === "meta") {
          trace.search_query = event.search_query;
          renderSources(event);
          text.textContent = "";
        } else if (event.type === "sentence") {
          sentences.push(event.text);
          text.textContent = sentences.join(" ");
          turn.scrollIntoView({ block: "end" });
        } else if (event.type === "done") {
          answer = event.answer;
          trace.path = event.path;
          meta.replaceChildren(...(event.citations as Citation[]).map(c => el("span", { className: "cite", textContent: citationLabel(c) })),
            el("span", { textContent: `${(event.total_ms / 1000).toFixed(1)} s` }));
        } else if (event.type === "error") {
          turn.classList.add("error");
          text.textContent = event.detail;
        }
      },
    });
    if (!answer && !turn.classList.contains("error") && !sentences.length) text.textContent = "Stopped.";
  } catch (error) {
    turn.classList.add("error");
    text.textContent = `Something went wrong: ${String(error)}`;
  } finally {
    turn.classList.remove("pending");
    trace.total_ms = performance.now() - started;
    Object.assign(trace, state.driver.usage);
    state.traces.unshift(trace);
    state.traces.length = Math.min(state.traces.length, KEEP_TRACES);
    renderTraces(0);
    if (answer) {
      state.history.push({ role: "user", content: query.slice(0, TURN_CHARS) },
                         { role: "assistant", content: answer.slice(0, TURN_CHARS) });
      state.history = state.history.slice(-HISTORY_TURNS);
    }
    state.busy = false;
    $<HTMLButtonElement>("ask").disabled = false;
    $("stop").hidden = true;
  }
}

function renderTraces(selected: number) {
  const list = $("trace-list");
  if (!state.traces.length) return;
  list.replaceChildren(...state.traces.map((t, i) => {
    const button = el("button", { type: "button" }, el("span", { textContent: t.query }),
      el("small", { textContent: `${t.path} · ${(t.total_ms / 1000).toFixed(1)} s · ${new Date(t.asked_at).toLocaleTimeString()}` }));
    button.setAttribute("aria-pressed", String(i === selected));
    button.addEventListener("click", () => renderTraces(i));
    return el("li", {}, button);
  }));
  const t = state.traces[selected];
  $("trace-title").textContent = `Steps · ${t.query}`;
  const lines = t.spans.map(s => `${s.layer.padEnd(11)} ${String(Math.round(s.ms)).padStart(6)} ms  ${s.details}`);
  lines.push("", `search query  ${t.search_query}`, `path          ${t.path}`, `model         ${t.model}`,
    `tokens        ${t.prompt_tokens ?? "?"} prompt, ${t.output_tokens ?? "?"} generated`,
    `total         ${(t.total_ms / 1000).toFixed(2)} s`);
  $("trace-detail").textContent = lines.join("\n");
}

// --- boot ---

async function boot() {
  setupTabs();
  const [config, manifest, suggested, device] = await Promise.all([
    json<Config>("config.json"), json<{ documents: Doc[] }>("index/manifest.json"),
    json<Suggested[]>("index/suggested.json"), inspectDevice()]);
  Object.assign(state, { config, docs: manifest.documents, device });
  renderDocuments();
  renderSuggestions(suggested);
  void renderResults();

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
