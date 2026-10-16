import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, env } from "@huggingface/transformers";
env.allowLocalModels = false;
const post = (m: unknown) => (self as any).postMessage(m);

self.onmessage = async () => {
  try {
    let t = performance.now();
    const embed = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { device: "webgpu", dtype: "fp32", revision: "ea104dacec62" } as any);
    const v = await embed(["What port must be open for SSL?"], { pooling: "cls", normalize: true });
    post({ embedder: "ok", dims: v.dims, ms: Math.round(performance.now() - t), first: Array.from(v.data.slice(0, 3)) });

    t = performance.now();
    const rtok = await AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-12-v2");
    const rmod = await AutoModelForSequenceClassification.from_pretrained("Xenova/ms-marco-MiniLM-L-12-v2", { device: "webgpu", dtype: "fp32" } as any);
    const q = "What port must be open for SSL?";
    const docs = ["The router TX-9000 requires port 443 open for SSL connections.", "Bananas are yellow."];
    const enc = rtok(new Array(docs.length).fill(q), { text_pair: docs, padding: true, truncation: true });
    const out = await rmod(enc);
    post({ reranker: "ok", logits: Array.from(out.logits.data), ms: Math.round(performance.now() - t) });

    t = performance.now();
    const ntok = await AutoTokenizer.from_pretrained("Xenova/nli-deberta-v3-small");
    const nmod = await AutoModelForSequenceClassification.from_pretrained("Xenova/nli-deberta-v3-small", { device: "webgpu", dtype: "fp32" } as any);
    const nenc = ntok(["The router needs port 443 open."], { text_pair: ["Port 443 must be open for SSL."], padding: true, truncation: true });
    const nout = await nmod(nenc);
    post({ nli: "ok", labels: (nmod as any).config.id2label, logits: Array.from(nout.logits.data), ms: Math.round(performance.now() - t) });

    t = performance.now();
    const asr = await pipeline("automatic-speech-recognition", "onnx-community/whisper-base.en", { device: "webgpu" } as any);
    const silence = new Float32Array(16000);
    const r = await asr(silence);
    post({ whisper: "ok", text: (r as any).text, ms: Math.round(performance.now() - t) });
  } catch (e) {
    post({ error: String(e) });
  }
};
