import * as ort from "onnxruntime-web";
import { AutoTokenizer, env } from "@huggingface/transformers";
const post = (m: unknown) => (self as any).postMessage(m);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/";
const MAX = 512, GATE = 0.65;

// [CLS] query [SEP] passage [SEP], trimming the longer side one token at a time (the Rust "longest_first" rule).
function pair(tok: any, query: string, passage: string) {
  const q: number[] = tok.encode(query, { add_special_tokens: false });
  const p: number[] = tok.encode(passage, { add_special_tokens: false });
  let a = q.length, b = p.length;
  while (a + b > MAX - 3) { if (a > b) a--; else b--; }
  const ids = [101, ...q.slice(0, a), 102, ...p.slice(0, b), 102];
  const types = [...new Array(a + 2).fill(0), ...new Array(b + 1).fill(1)];
  return { ids, types };
}

self.onmessage = async () => { try {
  env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = "/models/";
  const tok = await AutoTokenizer.from_pretrained("flashrank");
  const ref = await (await fetch("/rerank_reference.json")).json();
  const session = await ort.InferenceSession.create("/models/flashrank/flashrank-MiniLM-L-12-v2_Q.onnx", { executionProviders: ["wasm"] });
  const t = performance.now();
  let worst = 0, agree = 0, near = 0, done = 0;
  for (const r of ref) {
    const pairs = r.passages.map((p: string) => pair(tok, r.query, p));
    const width = Math.max(...pairs.map((x: any) => x.ids.length));
    const n = pairs.length;
    const ids = new BigInt64Array(n * width), mask = new BigInt64Array(n * width), types = new BigInt64Array(n * width);
    pairs.forEach((x: any, row: number) => x.ids.forEach((id: number, k: number) => {
      ids[row * width + k] = BigInt(id); mask[row * width + k] = 1n; types[row * width + k] = BigInt(x.types[k]);
    }));
    const dims = [n, width];
    const feeds: Record<string, ort.Tensor> = {};
    for (const name of session.inputNames)
      feeds[name] = new ort.Tensor("int64", name === "input_ids" ? ids : name === "attention_mask" ? mask : types, dims);
    const out = await session.run(feeds);
    const scores = Array.from(out[session.outputNames[0]].data as Float32Array).map(x => 1 / (1 + Math.exp(-x)));
    scores.forEach((s, i) => { worst = Math.max(worst, Math.abs(s - r.flashrank_scores[i])); });
    const mine = Math.max(...scores), theirs = Math.max(...r.flashrank_scores);
    agree += Number((mine >= GATE) === (theirs >= GATE));
    near += Number(Math.abs(theirs - GATE) < 0.01);
    done++; if (done % 10 === 0) post({ progress: done, ms: Math.round(performance.now() - t), worst });
  }
  post({ questions: ref.length, gate_agreement: agree, max_abs_diff: worst, top_scores_within_001_of_gate: near, ms: Math.round(performance.now() - t) });
  } catch (e) { post({ error: String(e), stack: String((e as any)?.stack).slice(0, 300) }); }
};
