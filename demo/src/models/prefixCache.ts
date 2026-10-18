// Prompt prefix caching for the grounding judge. The judge's prompt is the server's own (`judge_messages`):
// "Context:\n{passages}\n\nStatement: {sentence}\n\nIs the statement fully supported ...". Every sentence of one
// answer shares everything before its "Statement:", so the judge engine reads that prefix once, keeps a checkpoint
// of its KV cache, and for the next sentence imports the checkpoint and reads only the rest. The prompt, the
// sampling and the grammar stay WebLLM's own; only the repeated prefill is skipped.
//
// WebLLM 0.2.86 has no prefix cache, so this wraps a private pipeline method (`forwardPrefill`) and uses its KV
// checkpoint functions (`exportPromptCheckpoint`, `importPromptCheckpoint`). A prompt without the marker, or any
// shape it does not expect, goes to WebLLM's own prefill; any error undoes its changes, turns the cache off for
// the rest of the session and hands the request to WebLLM's own prefill too.

type Pipeline = Record<string, any>;

/** The index where the last occurrence of `marker` in `ids` starts, or -1. */
function lastStart(ids: number[], marker: number[]): number {
  outer: for (let start = ids.length - marker.length; start >= 0; start--) {
    for (let k = 0; k < marker.length; k++) if (ids[start + k] !== marker[k]) continue outer;
    return start;
  }
  return -1;
}

/** Wraps the pipeline's prefill. `markerText` starts the part that changes per sentence ("\n\nStatement:"). */
export function installPrefixCache(pipe: Pipeline, markerText: string): void {
  const original = pipe.forwardPrefill.bind(pipe);
  // Split before the word itself: the newlines in front can merge with the passage's last characters into one
  // token, but a word after a newline always starts a token of its own
  const word: number[] = Array.from(pipe.tokenizer.encode(markerText.trimStart()));
  let saved: { ids: number[]; checkpoint: unknown } | null = null;
  let usable = word.length > 0;

  pipe.forwardPrefill = async function (this: Pipeline, inp: unknown, msgRole: string, inpRoleStr: unknown, genConfig: any) {
    if (!usable || typeof inp !== "string" || !inp.includes(markerText) || this.artifact !== undefined
        || this.filledKVCacheLength !== 0 || this.conversation.isTextCompletion || genConfig?.enable_thinking === false) {
      return original(inp, msgRole, inpRoleStr, genConfig);
    }
    const messagesBefore = this.conversation.messages.length;
    try {
      return await cachedPrefill.call(this, inp, msgRole, inpRoleStr);
    } catch (error) {
      console.warn("[judge] prefix cache off, using WebLLM's own prefill:", error);
      usable = false;
      saved = null;
      this.conversation.messages.length = messagesBefore;
      this.tvm.beginScope();
      try { this.resetKVCache(); this.filledKVCacheLength = 0; } finally { this.tvm.endScope(); }
      return original(inp, msgRole, inpRoleStr, genConfig);
    }
  };

  async function cachedPrefill(this: Pipeline, inp: string, msgRole: string, inpRoleStr: unknown) {
    this.conversation.appendMessage(msgRole, inp, inpRoleStr);
    this.conversation.appendReplyHeader("assistant");
    const [inputData, promptLen] = await this.getInputData();
    const ids: number[] = (inputData as unknown[][]).flat() as number[];
    const split = lastStart(ids, word);

    const forward = async (tokens: number[]) => {
      let logits: any;
      const size = this.prefillChunkSize > 0 ? this.prefillChunkSize : tokens.length;
      for (let i = 0; i < tokens.length; i += size) {
        const chunk = tokens.slice(i, i + size);
        const before = this.filledKVCacheLength;
        logits = await this.embedAndForward([chunk], chunk.length);
        if (this.filledKVCacheLength !== before + chunk.length) throw new Error("KV length mismatch");
      }
      return logits;
    };

    let logits: any;
    this.tvm.beginScope();
    try {
      const reusable = split > 0 && saved !== null && saved.ids.length === split
        && saved.ids.every((id, k) => id === ids[k]);
      if (reusable) {
        await this.importPromptCheckpoint(saved!.checkpoint);
        if (this.filledKVCacheLength !== split) throw new Error("imported KV length mismatch");
        logits = await forward(ids.slice(split));
      } else if (split > 0) {
        await forward(ids.slice(0, split));
        const checkpoint = await this.exportPromptCheckpoint(undefined, false);
        if (checkpoint === undefined) throw new Error("this model library exports no checkpoint");
        saved = { ids: ids.slice(0, split), checkpoint };
        logits = await forward(ids.slice(split));
      } else {
        logits = await forward(ids);
      }
      logits = this.tvm.detachFromCurrentScope(logits);
    } finally {
      this.tvm.endScope();
    }
    return { logits, promptLen, promptTokenIds: ids, assistantPrefixTokenIds: [] };
  }
}
