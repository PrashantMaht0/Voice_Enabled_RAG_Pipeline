// Spoken answers: Kokoro 82M with the af_bella voice, fp32 on WebGPU (fp16 speaks garbled audio),
// loaded on the first spoken answer. kokoro-js 1.2.1 runs on its own copy of transformers.js and takes no revision, so the pinned revision is
// set on that copy's download path, and the voice file (which kokoro-js always fetches from main) is put into
// kokoro-js's own voice cache from the pinned revision first.
import { KokoroTTS } from "kokoro-js";
import { env } from "kokoro-transformers";

type Pin = { id: string; revision: string; dtype: string; voice: string };
type Call = { id: number; op: string; args: any };

let tts: KokoroTTS | null = null;
let voice = "af_bella";
const reply = (id: number, body: object, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage({ id, ...body }, transfer);

async function pinVoice(pin: Pin) {
  const main = `https://huggingface.co/${pin.id}/resolve/main/voices/${pin.voice}.bin`;
  const pinned = `https://huggingface.co/${pin.id}/resolve/${pin.revision}/voices/${pin.voice}.bin`;
  try {
    const cache = await caches.open("kokoro-voices");
    if (await cache.match(main)) return;
    const response = await fetch(pinned);
    if (!response.ok) throw new Error(`voice ${response.status}`);
    await cache.put(main, response);
  } catch {
    // no Cache API (some private windows): kokoro-js then fetches the voice itself
  }
}

async function load(id: number, pin: Pin) {
  env.allowLocalModels = false;
  env.remotePathTemplate = `{model}/resolve/${pin.revision}/`;
  voice = pin.voice;
  await pinVoice(pin);
  const files = new Map<string, { loaded: number; total: number }>();
  tts = await KokoroTTS.from_pretrained(pin.id, {
    dtype: pin.dtype as any, device: "webgpu",
    progress_callback: (info: any) => {
      if (info.status !== "progress" || !info.file) return;
      files.set(info.file, { loaded: info.loaded ?? 0, total: info.total ?? 0 });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      reply(id, { progress: { step: info.file, value: total ? loaded / total : 0 } });
    },
  });
}

self.onmessage = async (event: MessageEvent<Call>) => {
  const { id, op, args } = event.data;
  try {
    if (op === "load") { await load(id, args.pin); reply(id, { result: null }); }
    else if (op === "speak") {
      const audio = await tts!.generate(args.text, { voice: voice as any });
      const samples = audio.audio as Float32Array;
      reply(id, { result: { samples, rate: audio.sampling_rate } }, [samples.buffer]);
    } else throw new Error(`unknown op ${op}`);
  } catch (error) {
    reply(id, { error: String(error) });
  }
};
