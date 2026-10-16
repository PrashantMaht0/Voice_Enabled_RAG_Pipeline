// Voice (spec 0009, AC-6): the recorder (at most 30 seconds, resampled to 16 kHz mono for Whisper, with a loudness
// check, since Whisper writes " you" for silence) and the speaker (Kokoro sentences played back to back, or the
// browser's own speech on phones and when Kokoro cannot load). One stop() ends speech at once.
import { WorkerRpc } from "./rpc";

export const MAX_RECORDING_S = 30;
const RATE = 16000;
const FRAME = RATE / 20;               // 50 ms frames for the loudness check
// A recording whose loudest 50 ms never reaches this RMS is treated as silence (about -40 dBFS). Checked 2026-10-16:
// synthesized questions through a recorded stream peak at 0.20 to 0.24, digital silence at 0; a real microphone in a
// quiet room is not yet measured (build task 9), and transcripts that are only filler are caught after Whisper anyway.
export const QUIET_RMS = 0.01;

export class MicBlocked extends Error {}

/** The loudest 50 ms of a recording, as RMS (0 to 1). */
export function peakRms(samples: Float32Array): number {
  let peak = 0;
  for (let start = 0; start < samples.length; start += FRAME) {
    let sum = 0;
    const end = Math.min(samples.length, start + FRAME);
    for (let i = start; i < end; i++) sum += samples[i] * samples[i];
    peak = Math.max(peak, Math.sqrt(sum / Math.max(1, end - start)));
  }
  return peak;
}

/** Any decoded audio as 16 kHz mono, through an OfflineAudioContext. */
export async function to16kMono(buffer: AudioBuffer): Promise<Float32Array> {
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(buffer.duration * RATE)), RATE);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  return (await offline.startRendering()).getChannelData(0);
}

export class Recorder {
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private timer: number | null = null;
  private finished: Promise<Blob> | null = null;
  startedAt = 0;

  /** Starts recording; stops by itself after 30 seconds. Throws MicBlocked when the microphone is not allowed. */
  async start(onTick: (seconds: number) => void, onAutoStop: () => void) {
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    } catch (error) {
      const name = (error as DOMException)?.name;
      if (name === "NotAllowedError" || name === "SecurityError") throw new MicBlocked(String(error));
      throw error;
    }
    this.chunks = [];
    this.recorder = new MediaRecorder(stream);
    this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this.finished = new Promise(resolve => {
      this.recorder!.onstop = () => {
        for (const track of stream.getTracks()) track.stop();
        resolve(new Blob(this.chunks, { type: this.recorder!.mimeType }));
      };
    });
    this.recorder.start();
    this.startedAt = performance.now();
    this.timer = window.setInterval(() => {
      const seconds = (performance.now() - this.startedAt) / 1000;
      onTick(seconds);
      if (seconds >= MAX_RECORDING_S) onAutoStop();
    }, 250);
  }

  get recording() { return this.recorder?.state === "recording"; }

  /** Stops and returns the recording as 16 kHz mono samples. */
  async stop(): Promise<Float32Array> {
    if (this.timer != null) window.clearInterval(this.timer);
    this.timer = null;
    if (this.recorder?.state === "recording") this.recorder.stop();
    const blob = await this.finished!;
    const context = new AudioContext();
    try {
      return await to16kMono(await context.decodeAudioData(await blob.arrayBuffer()));
    } finally {
      void context.close();
    }
  }
}

type Pin = { id: string; revision: string; dtype: string; voice: string };

/** Speaks approved sentence pieces in order; times are milliseconds since `started`. */
export class Speaker {
  firstAudioMs: number | null = null;
  lastAudioMs: number | null = null;
  private context: AudioContext | null = null;
  private playhead = 0;
  private sources: AudioBufferSourceNode[] = [];
  private chain: Promise<void> = Promise.resolve();
  private stopped = false;
  private ending: Promise<void> | null = null;

  constructor(private engine: { tts: WorkerRpc } | null, private started: number) {
    if (engine) this.context = new AudioContext();
  }

  /** Kokoro when it loads (and not on a phone), else the browser's speech. Loads once, on the first spoken answer. */
  static async create(device: { mobile: boolean }, pin: Pin, started: number, onProgress: (value: number) => void,
                      onFallback: (reason: string) => void): Promise<Speaker> {
    if (device.mobile) return new Speaker(null, started);
    try {
      ttsWorker ??= loadKokoro(pin, onProgress);
      return new Speaker({ tts: await ttsWorker }, started);
    } catch (error) {
      ttsWorker = null;
      onFallback(String((error as Error)?.message ?? error));
      return new Speaker(null, started);
    }
  }

  /** Starts loading Kokoro early (on the mic press), so it loads while the visitor speaks; never throws. */
  static warm(device: { mobile: boolean }, pin: Pin, onProgress: (value: number) => void) {
    if (device.mobile || ttsWorker) return;
    ttsWorker = loadKokoro(pin, onProgress);
    ttsWorker.catch(() => { ttsWorker = null; });
  }

  get usesKokoro() { return this.engine !== null; }

  /** Queues pieces; each is synthesized after the one before, and played when the one before ends. */
  say(pieces: string[]) {
    for (const text of pieces) {
      this.chain = this.chain.then(() => this.stopped ? undefined : this.engine ? this.kokoro(text) : this.browser(text));
    }
  }

  /** Resolves when everything queued has been spoken (or stopped). */
  finished(): Promise<void> {
    this.ending ??= this.chain.then(() => this.engine ? this.drained() : this.browserDrained());
    return this.ending;
  }

  stop() {
    this.stopped = true;
    for (const s of this.sources) { try { s.stop(); } catch { /* already ended */ } }
    this.sources = [];
    if (!this.engine) speechSynthesis.cancel();
    void this.context?.close().catch(() => undefined);
  }

  private mark() {
    const ms = performance.now() - this.started;
    this.firstAudioMs ??= ms;
    this.lastAudioMs = ms;
  }

  private async kokoro(text: string) {
    const { samples, rate } = await this.engine!.tts.call<{ samples: Float32Array; rate: number }>("speak", { text });
    if (this.stopped || !this.context) return;
    this.mark();                                     // ready, as the dashboard measures it
    const buffer = this.context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(new Float32Array(samples), 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.playhead = Math.max(this.playhead, this.context.currentTime);
    source.start(this.playhead);
    this.playhead += buffer.duration;
    this.sources.push(source);
    source.onended = () => { this.sources = this.sources.filter(s => s !== source); };
  }

  private drained(): Promise<void> {
    if (this.stopped || !this.context) return Promise.resolve();
    const wait = Math.max(0, this.playhead - this.context.currentTime) * 1000;
    return new Promise(resolve => setTimeout(resolve, wait));
  }

  private browser(text: string): Promise<void> {
    return new Promise(resolve => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.onstart = () => this.mark();
      utterance.onend = utterance.onerror = () => resolve();
      speechSynthesis.speak(utterance);
      resolve();                                     // queued; the browser plays its queue in order
    });
  }

  private browserDrained(): Promise<void> {
    return new Promise(resolve => {
      const check = () => (this.stopped || (!speechSynthesis.speaking && !speechSynthesis.pending)) ? resolve() : setTimeout(check, 200);
      check();
    });
  }
}

let ttsWorker: Promise<WorkerRpc> | null = null;

async function loadKokoro(pin: Pin, onProgress: (value: number) => void): Promise<WorkerRpc> {
  const rpc = new WorkerRpc(new Worker(new URL("./workers/tts.worker.ts", import.meta.url), { type: "module" }));
  await rpc.call("load", { pin }, (p) => onProgress(p.value));
  return rpc;
}
