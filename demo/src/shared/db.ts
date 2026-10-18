// Browser storage (IndexedDB "vrag-demo" version 1): documents, chunks, upload_quota, traces (the last
// 50) and settings. When IndexedDB is not available (some private windows), every store lives in memory for the visit.
type SavedSpan = { layer: string; elapsed_ms: number; details: string; prompt_tokens?: number | null; output_tokens?: number | null };
export type Trace = {
  id?: number; asked_at: string; channel: string; query: string; search_query: string; path: string;
  answer: string | null; confidence: number | null; grounding_checker: string | null; sentences_dropped: number | null;
  total_ms: number | null; first_sentence_ms: number | null; stt_ms: number | null; first_audio_ms: number | null;
  spoken_total_ms: number | null; prompt_tokens: number | null; output_tokens: number | null; model: string;
  spans: SavedSpan[]; retrieved: any[]; dropped_sentences: string[]; error: string | null;
};
type DocStatus = "reading" | "indexing" | "ready" | "failed";
export type DocRecord = {
  name: string; origin: "upload"; pages: number | null; chunk_count: number; added_at: string;
  status: DocStatus; error: string | null;
};
export type ChunkRecord = {
  doc: string; id: string; doc_hash: string | null; chunk_index: number | null; page: number | null; kind: string;
  content: string; embedding: Float32Array | null; embedder_id: string | null;
};
export type Quota = { key: "quota"; window_started_at: string | null; used: number };

type StoreName = "documents" | "chunks" | "upload_quota" | "traces" | "settings";
const NAME = "vrag-demo", VERSION = 1, KEEP_TRACES = 50;
const KEYS: Record<StoreName, (value: any) => string> = {
  documents: v => v.name, chunks: v => JSON.stringify([v.doc, v.id]), upload_quota: v => v.key,
  traces: v => String(v.id), settings: v => v.key,
};
let database: Promise<IDBDatabase | null> | null = null;
const memory: Record<StoreName, Map<string, any>> = {
  documents: new Map(), chunks: new Map(), upload_quota: new Map(), traces: new Map(), settings: new Map(),
};

function open(): Promise<IDBDatabase | null> {
  database ??= new Promise(resolve => {
    try {
      const request = indexedDB.open(NAME, VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        db.createObjectStore("documents", { keyPath: "name" });
        db.createObjectStore("chunks", { keyPath: ["doc", "id"] }).createIndex("doc", "doc");
        db.createObjectStore("upload_quota", { keyPath: "key" });
        db.createObjectStore("traces", { keyPath: "id", autoIncrement: true });
        db.createObjectStore("settings", { keyPath: "key" });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return database;
}

const done = (tx: IDBTransaction) => new Promise<void>((resolve, reject) => {
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error);
  tx.onabort = () => reject(tx.error);
});

async function getAll<T>(store: StoreName): Promise<T[]> {
  const db = await open();
  if (!db) return [...memory[store].values()];
  const tx = db.transaction(store, "readonly");
  const request = tx.objectStore(store).getAll();
  await done(tx);
  return request.result as T[];
}

async function get<T>(store: StoreName, key: IDBValidKey): Promise<T | undefined> {
  const db = await open();
  if (!db) return memory[store].get(typeof key === "string" ? key : JSON.stringify(key));
  const tx = db.transaction(store, "readonly");
  const request = tx.objectStore(store).get(key);
  await done(tx);
  return request.result as T | undefined;
}

async function putMany(store: StoreName, values: any[]): Promise<void> {
  const db = await open();
  if (!db) { for (const v of values) memory[store].set(KEYS[store](v), v); return; }
  const tx = db.transaction(store, "readwrite");
  for (const v of values) tx.objectStore(store).put(v);
  await done(tx);
}

// --- traces ---

export async function addTrace(trace: Trace): Promise<Trace> {
  const db = await open();
  if (!db) {
    const ids = [...memory.traces.keys()].map(Number);
    trace.id = (ids.length ? Math.max(...ids) : 0) + 1;
    memory.traces.set(String(trace.id), trace);
    for (const id of ids.sort((a, b) => a - b).slice(0, Math.max(0, ids.length + 1 - KEEP_TRACES))) memory.traces.delete(String(id));
    return trace;
  }
  const tx = db.transaction("traces", "readwrite");
  const store = tx.objectStore("traces");
  const added = store.add(trace);
  added.onsuccess = () => {
    trace.id = added.result as number;
    const keys = store.getAllKeys();
    keys.onsuccess = () => {
      const all = keys.result as number[];
      for (const key of all.slice(0, Math.max(0, all.length - KEEP_TRACES))) store.delete(key);
    };
  };
  await done(tx);
  return trace;
}

export async function listTraces(): Promise<Trace[]> {
  return (await getAll<Trace>("traces")).sort((a, b) => (b.id ?? 0) - (a.id ?? 0));
}

// --- uploaded documents and their chunks ---

export const listDocuments = () => getAll<DocRecord>("documents");
export const putDocument = (doc: DocRecord) => putMany("documents", [doc]);
export const putChunks = (chunks: ChunkRecord[]) => putMany("chunks", chunks);

export async function chunksOf(doc: string): Promise<ChunkRecord[]> {
  const db = await open();
  const rows: ChunkRecord[] = db
    ? await (async () => {
        const tx = db.transaction("chunks", "readonly");
        const request = tx.objectStore("chunks").index("doc").getAll(doc);
        await done(tx);
        return request.result as ChunkRecord[];
      })()
    : [...memory.chunks.values()].filter(c => c.doc === doc);
  return rows.sort((a, b) => (a.chunk_index ?? 0) - (b.chunk_index ?? 0));
}

/** Deletes one document and all its chunks, in one transaction. */
export async function deleteDocument(name: string): Promise<void> {
  const db = await open();
  if (!db) {
    memory.documents.delete(name);
    for (const [key, c] of memory.chunks) if (c.doc === name) memory.chunks.delete(key);
    return;
  }
  const tx = db.transaction(["documents", "chunks"], "readwrite");
  tx.objectStore("documents").delete(name);
  tx.objectStore("chunks").delete(IDBKeyRange.bound([name], [name, []]));
  await done(tx);
}

/** "Clear my documents": uploads, chunks and traces go; the upload counter and settings stay. */
export async function clearMyDocuments(): Promise<void> {
  const db = await open();
  const stores: StoreName[] = ["documents", "chunks", "traces"];
  if (!db) { for (const s of stores) memory[s].clear(); return; }
  const tx = db.transaction(stores, "readwrite");
  for (const s of stores) tx.objectStore(s).clear();
  await done(tx);
}

// --- the upload counter and settings ---

export async function getQuota(): Promise<Quota> {
  return (await get<Quota>("upload_quota", "quota")) ?? { key: "quota", window_started_at: null, used: 0 };
}
export const putQuota = (quota: Quota) => putMany("upload_quota", [quota]);

export async function getSetting<T>(key: string, fallback: T): Promise<T> {
  try {
    const row = await get<{ key: string; value: T }>("settings", key);
    return row ? row.value : fallback;
  } catch {
    return fallback;
  }
}
export const putSetting = (key: string, value: unknown) => putMany("settings", [{ key, value }]).catch(() => undefined);
