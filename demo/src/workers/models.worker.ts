// The search and grounding models (spec 0009): bge-small embeddings, the ms-marco cross encoder and the
// nli-deberta-v3-small entailment model, on WebGPU with transformers.js, each loaded at its pinned revision.
// Whisper base.en (speech to text) loads here on the first mic press; it runs before an answer starts, so it never
// waits behind the grounding checks.
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

type ModelPin = { id: string; revision: string; dtype?: string; dtype_f16?: string; max_tokens?: number;
                  premise_tokens?: number; overlap_tokens?: number; max_sentence_tokens?: number };
type Pair = { ids: number[]; types: number[] };
type PairModel = { tokenizer: PreTrainedTokenizer; model: PreTrainedModel; max: number; cls: number; sep: number; types: boolean };
type Call = { id: number; op: string; args: any };

let embedder: FeatureExtractionPipeline | null = null;
let whisper: AutomaticSpeechRecognitionPipeline | null = null;
let reranker: PairModel | null = null;
let nli: (PairModel & { entailment: number; premise: number; overlap: number; maxSentence: number }) | null = null;

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

/** The grounding model: fp16 when the GPU has shader-f16, else fp32; labels come from its own config. */
async function loadNli(id: number, pin: ModelPin, f16: boolean) {
  const dtype = f16 && pin.dtype_f16 ? pin.dtype_f16 : pin.dtype ?? "fp32";
  const loaded = await pairModel(pin, dtype, progressFor(id));
  const labels = Object.entries((loaded.model as any).config.id2label as Record<string, string>);
  const entailment = labels.find(([, name]) => name.toLowerCase() === "entailment");
  if (!entailment) throw new Error("the NLI model has no entailment label");
  nli = { ...loaded, entailment: Number(entailment[0]), premise: pin.premise_tokens ?? 64,
          overlap: pin.overlap_tokens ?? 24, maxSentence: pin.max_sentence_tokens ?? 256 };
  return dtype;
}

async function embed(texts: string[]): Promise<number[][]> {
  const output = await embedder!(texts, { pooling: "cls", normalize: true });
  return output.tolist() as number[][];
}

// [CLS] query [SEP] passage [SEP], trimming the longer side one token at a time to the limit: the rule of the
// server's tokenizer ("longest first"), which transformers.js does not apply to pairs (build task 1).
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

/** Model inputs [CLS] window part [SEP] sentence [SEP]: the whole window when it is short enough, else overlapping
 *  short parts covering all of it, each input within the model's limit. The same rule as token_windows in
 *  src/verification/grounding.py, with its numbers from config.json. */
function tokenWindows(m: NonNullable<typeof nli>, window: string, sentence: number[]): Pair[] {
  const premise = m.tokenizer.encode(window, { add_special_tokens: false }) as number[];
  const hypothesis = sentence.slice(0, m.maxSentence);
  const budget = Math.min(m.premise, m.max - 3 - hypothesis.length);
  const starts: number[] = [];
  if (premise.length <= budget) starts.push(0);
  else for (let start = 0; start < premise.length - m.overlap; start += budget - m.overlap) starts.push(start);
  return starts.map(start => {
    const part = premise.slice(start, start + budget);
    return { ids: [m.cls, ...part, m.sep, ...hypothesis, m.sep],
             types: [...new Array(part.length + 2).fill(0), ...new Array(hypothesis.length + 1).fill(1)] };
  });
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

const NLI_BATCH = 32;                                  // short premises: about 90 tokens an input

/** The entailment probability of the sentence against each token window of each window (premise window, hypothesis
 *  sentence, as the server pairs them), by a softmax over the three labels. */
async function entailment(sentence: string, windows: string[]): Promise<number[]> {
  const hypothesis = nli!.tokenizer.encode(sentence, { add_special_tokens: false }) as number[];
  const inputs = windows.flatMap(w => tokenWindows(nli!, w, hypothesis));
  const scores: number[] = [];
  for (let i = 0; i < inputs.length; i += NLI_BATCH) {
    const rows = await classify(nli!, inputs.slice(i, i + NLI_BATCH));
    for (const logits of rows) {
      const top = Math.max(...logits);
      const exps = logits.map(x => Math.exp(x - top));
      scores.push(exps[nli!.entailment] / exps.reduce((a, b) => a + b, 0));
    }
  }
  return scores;
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
    else if (op === "load_nli") reply(id, { result: await loadNli(id, args.pin, args.f16) });
    else if (op === "nli") reply(id, { result: await entailment(args.sentence, args.windows) });
    else if (op === "load_whisper") reply(id, { result: await loadWhisper(id, args.pin, args.f16) });
    else if (op === "transcribe") reply(id, { result: await transcribe(new Float32Array(args.audio)) });
    else throw new Error(`unknown op ${op}`);
  } catch (error) {
    reply(id, { error: String(error) });
  }
};
