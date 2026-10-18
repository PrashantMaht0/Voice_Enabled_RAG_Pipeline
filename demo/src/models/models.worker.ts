// The search models: bge-small embeddings and the ms-marco cross encoder, on WebGPU with transformers.js, each loaded
// at its pinned revision. Whisper base.en (speech to text) loads here on the first mic press.
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Tensor,
  env,
  pipeline,
  type AutomaticSpeechRecognitionPipeline,
  type FeatureExtractionPipeline,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

env.allowLocalModels = false;

type ModelPin = { id: string; revision: string; dtype?: string; dtype_f16?: string; max_tokens?: number };
type Pair = { ids: number[]; types: number[] };
type PairModel = { tokenizer: PreTrainedTokenizer; model: PreTrainedModel; max: number; cls: number; sep: number; types: boolean };
type Call = { id: number; op: string; args: any };

let embedder: FeatureExtractionPipeline | null = null;
let whisper: AutomaticSpeechRecognitionPipeline | null = null;
let reranker: PairModel | null = null;

const reply = (id: number, body: object) => (self as unknown as Worker).postMessage({ id, ...body });

function progressFor(id: number) {
  const files = new Map<string, { loaded: number; total: number }>();
  return (info: any) => {
    if (info.status === "progress" && info.file) {
      files.set(`${info.name}/${info.file}`, { loaded: info.loaded ?? 0, total: info.total ?? 0 });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      reply(id, { progress: { step: info.file, value: total ? loaded / total : 0 } });
    }
  };
}

/** A sequence pair classifier at its pinned revision, with its special tokens and whether it takes token types. */
async function pairModel(pin: ModelPin, dtype: string, progress: (info: any) => void): Promise<PairModel> {
  const tokenizer = await AutoTokenizer.from_pretrained(pin.id, { revision: pin.revision, progress_callback: progress });
  const model = await AutoModelForSequenceClassification.from_pretrained(pin.id,
    { device: "webgpu", dtype: dtype as any, revision: pin.revision, progress_callback: progress });
  const ids = tokenizer.encode("") as number[];          // [CLS] [SEP]
  const inputs: string[] = (model as any).sessions?.model?.inputNames ?? ["token_type_ids"];
  return { tokenizer, model, max: pin.max_tokens ?? 512, cls: ids[0], sep: ids[ids.length - 1],
           types: inputs.includes("token_type_ids") };
}

async function load(id: number, pins: { embedder: ModelPin; reranker: ModelPin }) {
  const progress = progressFor(id);
  const { embedder: e, reranker: r } = pins;
  embedder = await pipeline("feature-extraction", e.id,
    { device: "webgpu", dtype: e.dtype as any, revision: e.revision, progress_callback: progress }) as FeatureExtractionPipeline;
  reranker = await pairModel(r, r.dtype ?? "fp32", progress);
}

async function embed(texts: string[]): Promise<number[][]> {
  const output = await embedder!(texts, { pooling: "cls", normalize: true });
  return output.tolist() as number[][];
}

// [CLS] query [SEP] passage [SEP], trimming the longer side one token at a time to the limit: the rule of the
// server's tokenizer ("longest first"), which transformers.js does not apply to pairs.
function pair(m: PairModel, first: string, second: string): Pair {
  const q = m.tokenizer.encode(first, { add_special_tokens: false }) as number[];
  const p = m.tokenizer.encode(second, { add_special_tokens: false }) as number[];
  let a = q.length, b = p.length;
  while (a + b > m.max - 3) { if (a > b) a--; else b--; }
  return {
    ids: [m.cls, ...q.slice(0, a), m.sep, ...p.slice(0, b), m.sep],
    types: [...new Array(a + 2).fill(0), ...new Array(b + 1).fill(1)],
  };
}

/** Logits for a batch of model inputs, as rows. */
async function classify(m: PairModel, pairs: Pair[]): Promise<number[][]> {
  const width = Math.max(...pairs.map(p => p.ids.length));
  const n = pairs.length;
  const ids = new BigInt64Array(n * width), mask = new BigInt64Array(n * width), types = new BigInt64Array(n * width);
  pairs.forEach((p, row) => p.ids.forEach((token, k) => {
    ids[row * width + k] = BigInt(token);
    mask[row * width + k] = 1n;
    types[row * width + k] = BigInt(p.types[k]);
  }));
  const dims = [n, width];
  const feeds: Record<string, Tensor> = {
    input_ids: new Tensor("int64", ids, dims),
    attention_mask: new Tensor("int64", mask, dims),
  };
  if (m.types) feeds.token_type_ids = new Tensor("int64", types, dims);
  const output = await m.model(feeds);
  const data = Array.from(output.logits.data as Float32Array);
  const labels = data.length / n;
  return Array.from({ length: n }, (_, row) => data.slice(row * labels, (row + 1) * labels));
}

async function rerank(query: string, texts: string[]): Promise<number[]> {
  if (!texts.length) return [];
  return (await classify(reranker!, texts.map(t => pair(reranker!, query, t)))).map(([x]) => 1 / (1 + Math.exp(-x)));
}

/** Whisper: the encoder in fp16 when the GPU has shader-f16 (else fp32); the merged decoder always fp32, since the
 *  fp16 merged decoder at this revision is rejected by onnxruntime ("subgraph output is an outer scope value"). */
async function loadWhisper(id: number, pin: ModelPin & { decoder_dtype?: string }, f16: boolean) {
  const dtype = { encoder_model: f16 && pin.dtype_f16 ? pin.dtype_f16 : pin.dtype ?? "fp32",
                  decoder_model_merged: pin.decoder_dtype ?? pin.dtype ?? "fp32" };
  whisper = await pipeline("automatic-speech-recognition", pin.id, {
    device: "webgpu", revision: pin.revision, progress_callback: progressFor(id), dtype: dtype as any,
  }) as AutomaticSpeechRecognitionPipeline;
  return `encoder ${dtype.encoder_model}, decoder ${dtype.decoder_model_merged}`;
}

async function transcribe(audio: Float32Array): Promise<string> {       // at most 30 seconds: one Whisper window
  const output = await whisper!(audio) as { text: string };
  return output.text.trim();
}

self.onmessage = async (event: MessageEvent<Call>) => {
  const { id, op, args } = event.data;
  try {
    if (op === "load") reply(id, { result: await load(id, args) ?? null });
    else if (op === "embed") reply(id, { result: await embed(args.texts) });
    else if (op === "rerank") reply(id, { result: await rerank(args.query, args.texts) });
    else if (op === "load_whisper") reply(id, { result: await loadWhisper(id, args.pin, args.f16) });
    else if (op === "transcribe") reply(id, { result: await transcribe(new Float32Array(args.audio)) });
    else throw new Error(`unknown op ${op}`);
  } catch (error) {
    reply(id, { error: String(error) });
  }
};
