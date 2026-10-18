// The judge's prefix cache on a fake WebLLM pipeline: reuse, prompts it must leave alone, and the fallback on error.
import assert from "node:assert/strict";
import { test } from "node:test";
import { installPrefixCache } from "../src/models/prefixCache.ts";

const MARKER = "\n\nStatement:";
const WORD = [900, 901];                         // "Statement:" as the fake tokenizer encodes it

/** A pipeline that "tokenizes" a prompt as given ids and records how many tokens each prefill forwards. */
function fakePipeline(prompts: Record<string, number[]>, options: { failImport?: boolean } = {}) {
  const forwarded: number[] = [];
  let ownPrefills = 0;
  const pipe: Record<string, any> = {
    artifact: undefined, filledKVCacheLength: 0, prefillChunkSize: 0,
    conversation: { isTextCompletion: false, messages: [] as unknown[][],
                    appendMessage(role: string, text: string) { this.messages.push([role, text]); },
                    appendReplyHeader(role: string) { this.messages.push([role, undefined]); } },
    tokenizer: { encode: (text: string) => (text === "Statement:" ? WORD : []) },
    tvm: { beginScope() {}, endScope() {}, detachFromCurrentScope: (x: unknown) => x },
    async getInputData() { const text = this.conversation.messages.at(-2)[1]; return [[prompts[text]], prompts[text].length]; },
    async embedAndForward(chunks: number[][], length: number) { this.filledKVCacheLength += length; forwarded.push(length); return "logits"; },
    async exportPromptCheckpoint() { return { length: this.filledKVCacheLength }; },
    async importPromptCheckpoint(checkpoint: { length: number }) {
      if (options.failImport) throw new Error("import failed");
      this.filledKVCacheLength = checkpoint.length;
    },
    resetKVCache() { this.filledKVCacheLength = 0; },
    async forwardPrefill() { ownPrefills++; this.conversation.messages.push(["own"]); return { logits: "own" }; },
  };
  installPrefixCache(pipe, MARKER);
  const ask = async (text: string) => {
    const result = await pipe.forwardPrefill(text, "user", undefined, {});
    pipe.filledKVCacheLength = 0;               // WebLLM resets the cache between independent requests
    pipe.conversation.messages = [];
    return result;
  };
  return { ask, forwarded, ownPrefills: () => ownPrefills };
}

const context = [1, 2, 3, 4, 5, 6];
const prompts = {
  [`one${MARKER} a`]: [...context, ...WORD, 10, 11],
  [`two${MARKER} b`]: [...context, ...WORD, 20, 21, 22],
  "an answer prompt": [7, 8, 9],
};

test("the second sentence of an answer forwards only its own tokens", async () => {
  const pipe = fakePipeline(prompts);
  await pipe.ask(`one${MARKER} a`);
  await pipe.ask(`two${MARKER} b`);
  assert.deepEqual(pipe.forwarded, [6, 4, 5]);
});

test("a prompt without the marker is read by WebLLM's own prefill", async () => {
  const pipe = fakePipeline(prompts);
  assert.deepEqual(await pipe.ask("an answer prompt"), { logits: "own" });
  assert.equal(pipe.ownPrefills(), 1);
});

test("an error undoes the change, hands the request to WebLLM and turns the cache off", async () => {
  const pipe = fakePipeline(prompts, { failImport: true });
  const warn = console.warn;
  console.warn = () => undefined;
  try {
    await pipe.ask(`one${MARKER} a`);
    assert.deepEqual(await pipe.ask(`two${MARKER} b`), { logits: "own" });
    assert.deepEqual(await pipe.ask(`one${MARKER} a`), { logits: "own" });
  } finally {
    console.warn = warn;
  }
  assert.equal(pipe.ownPrefills(), 2);
});
