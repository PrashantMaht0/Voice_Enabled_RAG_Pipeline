// Uploads: the page's own checks before anything is sent, the 24 hour quota per browser,
// and the call to the PDF reading Space with its waking, waiting and reading phases. The Space is the authority on
// page count, figures and chunk count; the page checks only what it can know without sending the file.
import type { Quota } from "../shared/db";

export type UploadLimits = { per_day: number; max_mb: number; max_pages: number; max_chunks: number; types: string[] };
type SpaceChunk = { id: string; doc_hash: string | null; chunk_index: number | null; page: number | null; kind: string; content: string };
type SpaceReply = { document: string; pages: number | null; chunks: SpaceChunk[]; seconds: number; readers: Record<string, string> };
export type Phase = { phase: "connecting" | "waking" | "waiting" | "reading"; ahead?: number };

const DAY_MS = 24 * 3600 * 1000;
const MAX_CHUNK_CHARS = 20_000;       // far above a real chunk; anything longer is not the reader's output

const isCount = (v: unknown) => v === null || (Number.isInteger(v) && (v as number) >= 0);

/** The chunks of a Space reply that have the shape the index and the Python core expect; the rest are dropped. */
export function validChunks(chunks: unknown[]): SpaceChunk[] {
  return chunks.filter((c): c is SpaceChunk => {
    const chunk = c as Record<string, unknown> | null;
    return !!chunk && typeof chunk.id === "string" && typeof chunk.content === "string" && chunk.content.length > 0
      && chunk.content.length <= MAX_CHUNK_CHARS && typeof chunk.kind === "string"
      && (chunk.doc_hash === null || typeof chunk.doc_hash === "string")
      && isCount(chunk.page) && isCount(chunk.chunk_index);
  });
}

const extension = (name: string) => (name.match(/\.[^.]+$/)?.[0] ?? "").toLowerCase();

/** The quota after the window rolls over: a window older than 24 hours starts again at 0. */
function currentQuota(quota: Quota, now = Date.now()): Quota {
  const started = quota.window_started_at ? Date.parse(quota.window_started_at) : NaN;
  return Number.isFinite(started) && now - started < DAY_MS ? quota : { key: "quota", window_started_at: null, used: 0 };
}

/** Uploads left today, and when the window resets (null when none was used yet). */
export function quotaLeft(quota: Quota, limits: UploadLimits, now = Date.now()) {
  const q = currentQuota(quota, now);
  return { left: Math.max(0, limits.per_day - q.used), resetsAt: q.window_started_at ? Date.parse(q.window_started_at) + DAY_MS : null };
}

/** Counts one successful read. */
export function recordRead(quota: Quota, now = Date.now()): Quota {
  const q = currentQuota(quota, now);
  return { key: "quota", window_started_at: q.window_started_at ?? new Date(now).toISOString(), used: q.used + 1 };
}

export function hoursUntil(at: number, now = Date.now()) {
  const hours = Math.max(1, Math.ceil((at - now) / 3600e3));
  return `${hours} hour${hours === 1 ? "" : "s"}`;
}

/** Why the page refuses this file before sending it, or null (type, size, a name clash, the daily quota). */
export function refusal(file: { name: string; size: number }, existing: string[], quota: Quota, limits: UploadLimits,
                        now = Date.now()): string | null {
  if (!limits.types.includes(extension(file.name)))
    return `${file.name} is not a PDF, Markdown or text file.`;
  if (file.size > limits.max_mb * 1024 * 1024)
    return `${file.name} is larger than ${limits.max_mb} MB.`;
  if (file.size === 0)
    return `${file.name} is empty.`;
  const lower = file.name.toLowerCase();
  if (existing.some(n => n.toLowerCase() === lower))
    return `A document named ${file.name} is already here. Delete it first to replace it.`;
  const { left, resetsAt } = quotaLeft(quota, limits, now);
  if (!left)
    return `You have used today's ${limits.per_day} uploads. The next one is available in ${hoursUntil(resetsAt!, now)}.`;
  return null;
}

/** A refusal or failure from the Space, with the plain reason the page shows. */
export class SpaceError extends Error {
  constructor(message: string, readonly code: string) { super(message); }
}

/** The Space's errors arrive as "code: message" (space/app.py); ZeroGPU's own quota error passes through as is. */
function explain(raw: string): SpaceError {
  const text = raw.trim();
  if (/quota/i.test(text) && /gpu/i.test(text)) {
    const when = text.match(/try again in ([0-9:]+)/i)?.[1];
    return new SpaceError(`Your free GPU time on Hugging Face is used up for now${when ? ` (try again in ${when})` : ", it resets within 24 hours"}. `
      + "Text and Markdown files still work, and so do questions about the sample documents.", "gpu_quota");
  }
  const coded = text.match(/^([a-z_]+):\s*(.+)$/s);
  if (coded) return new SpaceError(coded[2], coded[1]);
  return new SpaceError(text ? `The PDF reader failed: ${text.slice(0, 200)}` : "The PDF reader failed.", "space_error");
}

type GradioClient = Awaited<ReturnType<typeof import("@gradio/client").Client.connect>>;
let client: Promise<GradioClient> | null = null;
let spacePhase: ((phase: Phase) => void) | null = null;

/** Sends one file to the Space and returns its chunks; `onPhase` hears waking, waiting (with the queue) and reading. */
export async function readWithSpace(space: string, file: File, onPhase: (phase: Phase) => void): Promise<SpaceReply> {
  onPhase({ phase: "connecting" });
  spacePhase = onPhase;
  const { Client, handle_file } = await import("@gradio/client");     // loaded on the first upload only
  client ??= Client.connect(space, {
    record_history: false,           // never copy a visitor's chunks into localStorage
    events: ["data", "status"],
    status_callback: (s) => {
      if (["sleeping", "building", "starting"].includes(s.status)) spacePhase?.({ phase: "waking" });
    },
  });
  let app: GradioClient;
  try {
    app = await client;
  } catch (error) {
    client = null;
    throw new SpaceError(`The PDF reader is not available right now (${String((error as Error)?.message ?? error).slice(0, 120)}). `
      + "Questions about the documents already here still work.", "space_down");
  }
  try {
    for await (const message of app.submit("/parse_document", [handle_file(file)])) {
      if (message.type === "status") {
        if (message.stage === "pending" && (message.position ?? 0) > 0) onPhase({ phase: "waiting", ahead: message.position });
        else if (message.stage === "pending" || message.stage === "generating") onPhase({ phase: "reading" });
        else if (message.stage === "error") {
          const detail = typeof message.message === "string" ? message.message : message.title ?? "";
          throw explain(detail || String(message.title ?? ""));
        }
      } else if (message.type === "data") {
        const reply = (message.data as unknown[])[0] as SpaceReply;
        if (!reply || !Array.isArray(reply.chunks)) throw new SpaceError("The PDF reader sent back an unexpected reply.", "space_error");
        const chunks = validChunks(reply.chunks);
        if (reply.chunks.length && !chunks.length) throw new SpaceError("The PDF reader sent back an unexpected reply.", "space_error");
        return { ...reply, chunks };
      }
    }
  } catch (error) {
    if (error instanceof SpaceError) throw error;
    const detail = (error as any)?.message ?? String(error);
    throw explain(String(detail));
  } finally {
    spacePhase = null;
  }
  throw new SpaceError("The PDF reader closed the connection before replying.", "space_error");
}
