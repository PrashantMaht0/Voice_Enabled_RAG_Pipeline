// WebLLM in its own worker, one per engine: the answer model and the grounding judge. The page drives each through
// CreateWebWorkerMLCEngine. The judge's page first posts { prefixCache: marker }, so that worker reads the judge's
// shared prompt prefix (the passages, before the marker) once per answer instead of once per sentence
// (prefixCache.ts). The answer worker gets no such message and runs WebLLM unchanged.
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";
import { installPrefixCache } from "./prefixCache";

const handler = new WebWorkerMLCEngineHandler();
const patched = new WeakSet<object>();
let marker: string | null = null;

self.onmessage = (message: MessageEvent) => {
  if (typeof message.data?.prefixCache === "string") {
    marker = message.data.prefixCache;
    return;
  }
  // The pipeline exists once the model has loaded; install the cache on it before its first request
  for (const pipe of (marker ? (handler as any).engine?.loadedModelIdToPipeline?.values?.() ?? [] : []) as Iterable<object>) {
    if (!patched.has(pipe)) {
      patched.add(pipe);
      try { installPrefixCache(pipe as Record<string, any>, marker!); } catch { /* WebLLM's own prefill stays */ }
    }
  }
  handler.onmessage(message);
};
