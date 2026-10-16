// The search models (spec 0009): bge-small embeddings and the ms-marco cross encoder, on WebGPU with transformers.js,
// each loaded at its pinned revision. Grounding (NLI) and speech (Whisper) join in later build tasks.
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Tensor,
  env,
  pipeline,
  type FeatureExtractionPipeline,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

env.allowLocalModels = false;

type ModelPin = { id: string; revision: string; dtype?: string; max_tokens?: number };
type Call = { id: number; op: string; args: any };

let embedder: FeatureExtractionPipeline | null = null;
let rerankTokenizer: PreTrainedTokenizer | null = null;
let rerankModel: PreTrainedModel | null = null;
let rerankMax = 512;
let special = { cls: 101, sep: 102 };

const reply = (id: number, body: object) => (self as unknown as Worker).postMessage({ id, ...body });

async function load(id: number, pins: { embedder: ModelPin; reranker: ModelPin }) {
  const files = new Map<string, { loaded: number; total: number }>();
  const progress = (info: any) => {
    if (info.status === "progress" && info.file) {
      files.set(`${info.name}/${info.file}`, { loaded: info.loaded ?? 0, total: info.total ?? 0 });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      reply(id, { progress: { step: info.file, value: total ? loaded / total : 0 } });
    }
  };
  const { embedder: e, reranker: r } = pins;
  embedder = await pipeline("feature-extraction", e.id,
    { device: "webgpu", dtype: e.dtype as any, revision: e.revision, progress_callback: progress }) as FeatureExtractionPipeline;
  rerankTokenizer = await AutoTokenizer.from_pretrained(r.id, { revision: r.revision, progress_callback: progress });
  rerankModel = await AutoModelForSequenceClassification.from_pretrained(r.id,
    { device: "webgpu", dtype: r.dtype as any, revision: r.revision, progress_callback: progress });
  rerankMax = r.max_tokens ?? 512;
  const ids = rerankTokenizer.encode("") as number[];     // [CLS] [SEP]
  special = { cls: ids[0], sep: ids[ids.length - 1] };
}

async function embed(texts: string[]): Promise<number[][]> {
  const output = await embedder!(texts, { pooling: "cls", normalize: true });
  return output.tolist() as number[][];
}

// [CLS] query [SEP] passage [SEP], trimming the longer side one token at a time to the limit: the rule of the
// server's tokenizer ("longest first"), which transformers.js does not apply to pairs (build task 1).
function pair(query: string, passage: string) {
  const q = rerankTokenizer!.encode(query, { add_special_tokens: false }) as number[];
  const p = rerankTokenizer!.encode(passage, { add_special_tokens: false }) as number[];
  let a = q.length, b = p.length;
  while (a + b > rerankMax - 3) { if (a > b) a--; else b--; }
  return {
    ids: [special.cls, ...q.slice(0, a), special.sep, ...p.slice(0, b), special.sep],
    types: [...new Array(a + 2).fill(0), ...new Array(b + 1).fill(1)],
  };
}

async function rerank(query: string, texts: string[]): Promise<number[]> {
  if (!texts.length) return [];
  const pairs = texts.map(t => pair(query, t));
  const width = Math.max(...pairs.map(p => p.ids.length));
  const n = pairs.length;
  const ids = new BigInt64Array(n * width), mask = new BigInt64Array(n * width), types = new BigInt64Array(n * width);
  pairs.forEach((p, row) => p.ids.forEach((token, k) => {
    ids[row * width + k] = BigInt(token);
    mask[row * width + k] = 1n;
    types[row * width + k] = BigInt(p.types[k]);
  }));
  const dims = [n, width];
  const output = await rerankModel!({
    input_ids: new Tensor("int64", ids, dims),
    attention_mask: new Tensor("int64", mask, dims),
    token_type_ids: new Tensor("int64", types, dims),
  });
  return Array.from(output.logits.data as Float32Array).map(x => 1 / (1 + Math.exp(-x)));
}

self.onmessage = async (event: MessageEvent<Call>) => {
  const { id, op, args } = event.data;
  try {
    if (op === "load") reply(id, { result: await load(id, args) ?? null });
    else if (op === "embed") reply(id, { result: await embed(args.texts) });
    else if (op === "rerank") reply(id, { result: await rerank(args.query, args.texts) });
    else throw new Error(`unknown op ${op}`);
  } catch (error) {
    reply(id, { error: String(error) });
  }
};
