// Browser storage (spec 0009, IndexedDB "vrag-demo" version 1): all five stores are created now, so uploads
// (documents, chunks, upload_quota) need no migration later. Traces keep the last 50. When IndexedDB is not
// available (some private windows), traces live in memory for the visit.
export type SavedSpan = { layer: string; elapsed_ms: number; details: string; prompt_tokens?: number | null; output_tokens?: number | null };
export type Trace = {
  id?: number; asked_at: string; channel: string; query: string; search_query: string; path: string;
  answer: string | null; confidence: number | null; grounding_checker: string | null; sentences_dropped: number | null;
  total_ms: number | null; first_sentence_ms: number | null; stt_ms: number | null; first_audio_ms: number | null;
  spoken_total_ms: number | null; prompt_tokens: number | null; output_tokens: number | null; model: string;
  spans: SavedSpan[]; retrieved: any[]; dropped_sentences: string[]; error: string | null;
};

const NAME = "vrag-demo", VERSION = 1, KEEP_TRACES = 50;
let database: Promise<IDBDatabase | null> | null = null;
const memory: Trace[] = [];

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
});

export async function addTrace(trace: Trace): Promise<Trace> {
  const db = await open();
  if (!db) {
    trace.id = (memory[0]?.id ?? 0) + 1;
    memory.unshift(trace);
    memory.length = Math.min(memory.length, KEEP_TRACES);
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
  const db = await open();
  if (!db) return [...memory];
  const tx = db.transaction("traces", "readonly");
  const request = tx.objectStore("traces").getAll();
  await done(tx);
  return (request.result as Trace[]).reverse();
}
