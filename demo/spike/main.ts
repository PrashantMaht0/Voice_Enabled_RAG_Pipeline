import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";

const log = (...a: unknown[]) => {
  const line = a.map(x => typeof x === "string" ? x : JSON.stringify(x)).join(" ");
  (document.getElementById("log") as HTMLElement).textContent += line + "\n";
  console.log("[spike]", line);
};
(window as any).spikeLog = log;

document.getElementById("gpu")!.onclick = async () => {
  log("crossOriginIsolated:", String(self.crossOriginIsolated));
  const gpu = (navigator as any).gpu;
  if (!gpu) return log("navigator.gpu: missing");
  const adapter = await gpu.requestAdapter();
  if (!adapter) return log("adapter: none");
  log("features:", [...adapter.features].sort());
  log("limits:", { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize });
  log("mobile:", String((navigator as any).userAgentData?.mobile));
};

document.getElementById("llm")!.onclick = async () => {
  const t0 = performance.now();
  const engine = await CreateWebWorkerMLCEngine(
    new Worker(new URL("./llm.worker.ts", import.meta.url), { type: "module" }),
    "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    { initProgressCallback: (p: any) => log("load", Math.round(p.progress * 100) + "%", p.text.slice(0, 60)) },
  );
  log("loaded in", Math.round(performance.now() - t0), "ms");
  const t1 = performance.now();
  const chunks = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Say hello in five words." }], temperature: 0, stream: true,
    stream_options: { include_usage: true },
  });
  let text = "", usage: any = null, first = 0;
  for await (const c of chunks) {
    if (!first && c.choices[0]?.delta?.content) first = performance.now() - t1;
    text += c.choices[0]?.delta?.content ?? "";
    if (c.usage) usage = c.usage;
  }
  log("answer:", text, "| first token ms:", Math.round(first), "| usage:", usage);
};

document.getElementById("tjs")!.onclick = () => {
  const w = new Worker(new URL("./tjs.worker.ts", import.meta.url), { type: "module" });
  w.onmessage = e => log("tjs:", e.data);
  w.postMessage("go");
};

document.getElementById("tts")!.onclick = () => {
  const w = new Worker(new URL("./tts.worker.ts", import.meta.url), { type: "module" });
  w.onmessage = e => log("tts:", e.data);
  w.postMessage("go");
};

document.getElementById("rr")!.onclick = () => {
  const w = new Worker(new URL("./rerank.worker.ts", import.meta.url), { type: "module" });
  w.onmessage = e => log("rerank:", e.data);
  w.postMessage("go");
};
