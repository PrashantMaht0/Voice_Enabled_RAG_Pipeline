// The Documents tab (spec 0009, AC-7 to AC-10 and AC-13): the sample corpus, the visitor's uploads with their status
// (reading, indexing, ready, failed), upload, delete and "Clear my documents". Uploads are read by the Space, embedded
// here with the same embedder as the sample index, kept in IndexedDB and added to the Pyodide index when ready.
import {
  chunksOf, clearMyDocuments, deleteDocument, getQuota, listDocuments, putChunks, putDocument, putQuota,
  type ChunkRecord, type DocRecord,
} from "./db";
import type { WorkerRpc } from "./rpc";
import { hoursUntil, quotaLeft, readWithSpace, recordRead, refusal, SpaceError, type Phase, type UploadLimits } from "./uploads";

export type SampleDoc = { name: string; title: string; publisher: string; pages: number | null; license: string; source_url: string };
type Context = {
  config: { embedder_id: string; space: string; uploads: UploadLimits };
  sampleDocs: SampleDoc[];
  sampleChunks: { doc: string; kind: string }[];
  onChange: () => void;                  // after a clear (traces went too) or a document became ready or left
};

const EMBED_BATCH = 16;
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

let ctx: Context;
const uploads = new Map<string, DocRecord>();
const kinds = new Map<string, string[]>();            // an upload's chunk kinds, for the corpus counts
const live = new Map<string, string>();               // a status line while a document is being read or indexed
const busy = new Set<string>();                       // documents with work in progress (no delete until it ends)
let models: WorkerRpc | null = null;
let python: WorkerRpc | null = null;
let queue: Promise<void> = Promise.resolve();         // uploads are read one at a time
let askedPersist = false;

/** The documents a question searches for a scope (AC-10): null means every document in the index. */
export function scopeDocs(scope: string): string[] | null {
  if (scope === "sample") return ctx.sampleDocs.map(d => d.name);
  if (scope === "mine") return [...uploads.values()].filter(d => d.status === "ready").map(d => d.name);
  return null;
}

export async function initDocuments(context: Context) {
  ctx = context;
  for (const doc of await listDocuments().catch(() => [] as DocRecord[])) {
    if (doc.status === "reading") {                   // the page closed while the Space was reading it
      Object.assign(doc, { status: "failed", error: "The page was closed before the PDF reader finished. Delete it and upload it again." });
      await putDocument(doc).catch(() => undefined);
    }
    uploads.set(doc.name, doc);
    if (doc.status !== "failed") kinds.set(doc.name, (await chunksOf(doc.name).catch(() => [])).map(c => c.kind));
  }
  const input = $<HTMLInputElement>("upload-file");
  input.addEventListener("change", () => { addFiles([...(input.files ?? [])]); input.value = ""; });
  const zone = $("upload-zone");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); addFiles([...(e.dataTransfer?.files ?? [])]); });
  $("clear-docs").addEventListener("click", () => void clearAll());
  $("space-link").setAttribute("href", `https://huggingface.co/spaces/${ctx.config.space}`);
  await render();
}

/** After Load: index what was stored (ready uploads straight in; old embeddings or unfinished indexing again, AC-13). */
export async function attachModels(modelsRpc: WorkerRpc, pythonRpc: WorkerRpc) {
  models = modelsRpc;
  python = pythonRpc;
  for (const doc of uploads.values()) {
    if (doc.status === "failed" || busy.has(doc.name)) continue;
    const chunks = await chunksOf(doc.name);
    const stale = doc.status === "indexing" || chunks.some(c => c.embedder_id !== ctx.config.embedder_id || !c.embedding);
    if (stale) void track(doc.name, () => index(doc, chunks));
    else await addToIndex(doc.name, chunks);
  }
  await render();
}

function message(text: string, tone: "info" | "error" = "info") {
  const line = $("upload-message");
  line.textContent = text;
  line.classList.toggle("error", tone === "error");
}

function addFiles(files: File[]) {
  for (const file of files) queue = queue.then(() => upload(file)).catch(() => undefined);
}

async function track(name: string, work: () => Promise<void>) {
  busy.add(name);
  try { await work(); } finally { busy.delete(name); live.delete(name); await render(); }
}

async function upload(file: File) {
  const existing = [...ctx.sampleDocs.map(d => d.name), ...uploads.keys()];
  const reason = refusal(file, existing, await getQuota(), ctx.config.uploads);
  if (reason) { message(reason, "error"); return; }
  if (!askedPersist) {                                // keep uploads from being evicted under storage pressure
    askedPersist = true;
    void navigator.storage?.persist?.().catch(() => false);
  }
  const doc: DocRecord = { name: file.name, origin: "upload", pages: null, chunk_count: 0, added_at: new Date().toISOString(),
                           status: "reading", error: null };
  uploads.set(doc.name, doc);
  await putDocument(doc);
  message(`Sending ${file.name} to the PDF reader…`);
  await track(doc.name, async () => {
    const started = Date.now();
    let phase: Phase = { phase: "connecting" };
    const show = () => {
      const s = Math.round((Date.now() - started) / 1000);
      live.set(doc.name, phase.phase === "waking" ? `Waking up the PDF reader (${s} s)`
        : phase.phase === "waiting" ? `Waiting for the PDF reader (${phase.ahead} ahead, ${s} s)`
        : phase.phase === "reading" ? `Reading (${s} s)` : `Connecting (${s} s)`);
      void render();
    };
    const timer = setInterval(show, 1000);
    try {
      const reply = await readWithSpace(ctx.config.space, file, (p) => { phase = p; show(); });
      clearInterval(timer);
      await putQuota(recordRead(await getQuota()));
      const chunks: ChunkRecord[] = reply.chunks.slice(0, ctx.config.uploads.max_chunks).map(c => ({
        doc: doc.name, id: c.id, doc_hash: c.doc_hash, chunk_index: c.chunk_index, page: c.page, kind: c.kind || "text",
        content: c.content, embedding: null, embedder_id: null }));
      if (!chunks.length) throw new SpaceError(`No text could be read from ${file.name}.`, "empty");
      Object.assign(doc, { pages: reply.pages, chunk_count: chunks.length, status: "indexing" });
      await putChunks(chunks);
      await putDocument(doc);
      kinds.set(doc.name, chunks.map(c => c.kind));
      const tables = chunks.filter(c => c.kind === "table").length, figures = chunks.filter(c => c.kind === "figure").length;
      const n = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;
      message(`${file.name}: ${n(chunks.length, "chunk")} (${n(tables, "table")}, ${n(figures, "figure")}), `
        + `read in ${reply.seconds < 1 ? "under a second" : `${Math.round(reply.seconds)} s`}.`);
      await index(doc, chunks);
    } catch (error) {
      clearInterval(timer);
      const text = error instanceof SpaceError ? error.message : `Something went wrong: ${String(error).slice(0, 160)}`;
      if (uploads.has(doc.name)) {
        Object.assign(doc, { status: "failed", error: text });
        await putDocument(doc).catch(() => undefined);
      }
      message(text, "error");
    }
  });
}

/** Embeds what lacks a current embedding, stores it, and adds the document to the index. */
async function index(doc: DocRecord, chunks: ChunkRecord[]) {
  doc.status = "indexing";
  await putDocument(doc);
  if (!models || !python) return;                   // indexed after Load (attachModels)
  const todo = chunks.filter(c => c.embedder_id !== ctx.config.embedder_id || !c.embedding);
  for (let i = 0; i < todo.length; i += EMBED_BATCH) {
    if (!uploads.has(doc.name)) return;               // deleted or cleared meanwhile
    live.set(doc.name, `Indexing ${i} of ${todo.length} chunks`);
    void render();
    const batch = todo.slice(i, i + EMBED_BATCH);
    const vectors = await models.call<number[][]>("embed", { texts: batch.map(c => c.content) });
    batch.forEach((c, k) => { c.embedding = Float32Array.from(vectors[k]); c.embedder_id = ctx.config.embedder_id; });
    await putChunks(batch);
  }
  if (!uploads.has(doc.name)) return;
  await addToIndex(doc.name, chunks);
  doc.status = "ready";
  await putDocument(doc);
}

async function addToIndex(name: string, chunks: ChunkRecord[]) {
  if (!python) return;
  const dims = chunks[0]?.embedding?.length ?? 384;
  const vectors = new Float32Array(chunks.length * dims);
  chunks.forEach((c, i) => vectors.set(c.embedding!, i * dims));
  const rows = chunks.map(({ embedding, embedder_id, ...rest }) => rest);
  await python.call("add_doc", { name, chunks: JSON.stringify(rows), vectors: vectors.buffer }, undefined, [vectors.buffer]);
  ctx.onChange();
}

async function remove(name: string) {
  uploads.delete(name);
  kinds.delete(name);
  await deleteDocument(name);
  if (python) await python.call("remove_doc", { name });
  message(`Deleted ${name}.`);
  ctx.onChange();
  await render();
}

async function clearAll() {
  if (!confirm("Remove your uploaded documents, their chunks and your saved traces from this browser? "
             + "The upload counter and your settings stay.")) return;
  uploads.clear();
  kinds.clear();
  await clearMyDocuments();
  if (python) await python.call("clear_uploads");
  message("Your documents and traces are cleared from this browser.");
  ctx.onChange();
  await render();
}

// --- rendering ---

async function render() {
  const { left, resetsAt } = quotaLeft(await getQuota().catch(() => ({ key: "quota" as const, window_started_at: null, used: 0 })), ctx.config.uploads);
  const limits = ctx.config.uploads;
  $("upload-quota").textContent = left
    ? `${left} of ${limits.per_day} uploads left today.`
    : `No uploads left today; the next is available in ${hoursUntil(resetsAt!)}.`;
  $<HTMLButtonElement>("clear-docs").disabled = uploads.size === 0;

  $("documents").replaceChildren(...ctx.sampleDocs.map(d => el("article", { className: "doc" },
    el("h3", { textContent: d.title }),
    el("p", { className: "hint", textContent: `${d.publisher} · ${d.license}` }),
    el("a", { href: d.source_url, textContent: "Source", rel: "noopener" }))));

  const count = (list: string[], kind?: string) => String(list.filter(k => !kind || k === kind).length);
  const sampleKinds = (name: string) => ctx.sampleChunks.filter(c => c.doc === name).map(c => c.kind);
  const rows = ctx.sampleDocs.map(d => {
    const k = sampleKinds(d.name);
    return el("tr", {}, el("td", { textContent: d.name }), el("td", { textContent: String(d.pages ?? "") }),
      el("td", { textContent: count(k) }), el("td", { textContent: count(k, "table") }), el("td", { textContent: count(k, "figure") }),
      el("td", { textContent: "Sample, indexed" }), el("td"));
  });
  const mine = [...uploads.values()].sort((a, b) => a.added_at.localeCompare(b.added_at));
  for (const d of mine) {
    const k = kinds.get(d.name) ?? [];
    const status = live.get(d.name) ?? (d.status === "failed" ? `Failed: ${d.error ?? "unknown reason"}`
      : d.status === "ready" ? "Ready" : d.status === "indexing" ? (models ? "Indexing" : "Indexes after Load models") : "Reading");
    const cell = el("td", { className: `status ${d.status}`, textContent: status });
    const remover = el("button", { type: "button", className: "small", textContent: "Delete", disabled: busy.has(d.name) });
    remover.setAttribute("aria-label", `Delete ${d.name}`);
    remover.addEventListener("click", () => void remove(d.name));
    rows.push(el("tr", {}, el("td", { textContent: d.name }), el("td", { textContent: d.pages == null ? "" : String(d.pages) }),
      el("td", { textContent: d.chunk_count ? String(d.chunk_count) : "" }), el("td", { textContent: k.length ? count(k, "table") : "" }),
      el("td", { textContent: k.length ? count(k, "figure") : "" }), cell, el("td", {}, remover)));
  }
  $("corpus").querySelector("tbody")!.replaceChildren(...rows);
  $("my-empty").hidden = mine.length > 0;
}
