import { KokoroTTS } from "kokoro-js";
const post = (m: unknown) => (self as any).postMessage(m);
const SENTENCE = "Azure App Service plans start at about fifty dollars a month, as the pricing table on page four shows.";
self.onmessage = async () => {
  for (const [device, dtype] of [["webgpu", "fp32"], ["webgpu", "fp16"], ["wasm", "q8"]] as const) {
    try {
      const t = performance.now();
      const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", { dtype, device } as any);
      const load = performance.now() - t;
      const times: number[] = [];
      let seconds = 0;
      for (let i = 0; i < 3; i++) {
        const t2 = performance.now();
        const audio = await tts.generate(SENTENCE, { voice: "af_bella" });
        times.push(Math.round(performance.now() - t2));
        seconds = (audio as any).audio.length / (audio as any).sampling_rate;
      }
      post({ device, dtype, load_ms: Math.round(load), gen_ms: times, audio_s: Math.round(seconds * 10) / 10 });
    } catch (e) {
      post({ device, dtype, error: String(e) });
    }
  }
};
