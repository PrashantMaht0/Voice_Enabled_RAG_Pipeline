// The browser driver: runs one question through the shared Python core in the Pyodide worker and
// answers each effect it yields with the browser's models. Search effects never reach here (the bridge answers
// them next to the index); Retrieve, the cache and the trace store have browser answers of their own.
import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import type { WorkerRpc } from "../shared/rpc";

type Settings = { temperature: number; top_p: number; max_tokens: number };
type Span = { layer: string; ms: number; details: string };
export type Event = { type: string; [key: string]: any };
type Usage = { prompt_tokens: number | null; output_tokens: number | null };

export type Question = { query: string; history: { role: string; content: string }[]; docs: string[] | null; grounding: boolean;
                         channel?: "typed" | "voice" };
type Handlers = { onEvent: (event: Event) => void; onSpan: (span: Span) => void };

// The judge answers in plain words ("Yes." or "No."), read by its first word. Forcing a JSON reply made this 4 bit
// WebLLM build say yes to nearly everything: on 147 test sentences it caught 0 of 83 unsupported ones, against 9 in
// plain words, with 1 and 2 of 54 supported ones dropped (2026-10-18).
const JUDGE_MAX_TOKENS = 4;

export class Driver {
  private stream: AsyncIterator<any> | null = null;
  private stopped = false;
  usage: Usage = { prompt_tokens: null, output_tokens: null };
  /** The grounding judge's own engine, and why grounding cannot run (still loading, failed), or null once ready. */
  judge: MLCEngineInterface | null = null;
  checkProblem: string | null = "the grounding judge is still loading";

  constructor(private python: WorkerRpc, private models: WorkerRpc, private engine: MLCEngineInterface,
              private settings: Settings) {}

  /** Runs one question; resolves with the core's record (path, answer, confidence, …), or null after an error. */
  async ask(question: Question, handlers: Handlers): Promise<Record<string, any> | null> {
    this.stopped = false;
    this.usage = { prompt_tokens: null, output_tokens: null };
    const request = JSON.stringify({ query: question.query, history: question.history, channel: question.channel ?? "typed" });
    let item = JSON.parse(await this.python.call<string>("start",
      { request, docs: JSON.stringify(question.docs), grounding: question.grounding }));
    while (!item.end) {
      if (this.stopped) return null;
      let result: unknown = null;
      if (item.effect) result = await this.perform(item, handlers);
      else handlers.onEvent(item);
      if (this.stopped) return null;
      item = JSON.parse(await this.python.call<string>("step", { result: JSON.stringify(result ?? null) }));
    }
    if (item.type === "error") handlers.onEvent(item);
    return item.record ?? null;
  }

  async stop(): Promise<void> {
    this.stopped = true;
    try { this.engine.interruptGenerate(); } catch { /* nothing is generating */ }
    this.stream = null;
    await this.python.call("stop");
  }

  private async perform(effect: any, handlers: Handlers): Promise<unknown> {
    switch (effect.effect) {
      case "Span":
        handlers.onSpan({ layer: effect.layer, ms: effect.ms ?? 0, details: effect.details });
        return null;
      case "Rewrite":
        try {
          const reply = await this.engine.chat.completions.create({
            messages: effect.messages, temperature: this.settings.temperature, max_tokens: 64, stream: false,
          });
          return reply.choices[0]?.message?.content ?? null;
        } catch {
          return null;                    // the core then searches the question as asked
        }
      case "Retrieve":                    // the core does the retrieval steps itself
      case "CacheLookup":                 // the browser keeps no answer cache
      case "CacheStore":
        return null;
      case "Embed":
        return this.models.call("embed", { texts: effect.texts });
      case "Rerank":
        return this.models.call("rerank", { query: effect.query, texts: effect.texts });
      case "Generate": {
        const chunks = await this.engine.chat.completions.create({
          messages: effect.messages, temperature: this.settings.temperature, top_p: this.settings.top_p,
          max_tokens: this.settings.max_tokens, stream: true, stream_options: { include_usage: true },
        });
        this.stream = (chunks as AsyncIterable<any>)[Symbol.asyncIterator]();
        return null;
      }
      case "NextToken": {
        while (this.stream) {
          const next = await this.stream.next();
          if (next.done) break;
          const chunk = next.value;
          if (chunk.usage) this.usage = { prompt_tokens: chunk.usage.prompt_tokens, output_tokens: chunk.usage.completion_tokens };
          const token = chunk.choices?.[0]?.delta?.content;
          if (token) return { token };
        }
        this.stream = null;
        return { done: true, ...this.usage };
      }
      case "CheckSentence":               // the judge's raw reply goes back; the core reads it, and fails open on an error
        if (this.checkProblem || !this.judge) return { error: this.checkProblem ?? "the grounding judge is not loaded" };
        try {
          const reply = await this.judge.chat.completions.create({
            messages: effect.judge.messages, temperature: 0, max_tokens: JUDGE_MAX_TOKENS, stream: false,
          });
          return { reply: reply.choices[0]?.message?.content ?? "" };
        } catch (error) {
          return { error: `the grounding judge failed (${String((error as Error)?.message ?? error).slice(0, 120)})` };
        }
      default:
        throw new Error(`unknown effect ${effect.effect}`);
    }
  }
}
