// The browser driver with fake workers and engines: the judge's grammar, and Stop while an effect is running.
import assert from "node:assert/strict";
import { test } from "node:test";
import { Driver, compactJsonGrammar } from "../src/pipeline/driver.ts";

const settings = { temperature: 0, top_p: 0.1, max_tokens: 256 };

test("the judge grammar allows only the compact JSON the schema describes", () => {
  const schema = { required: ["supported"], properties: { supported: { enum: ["yes", "no"] } } };
  assert.equal(compactJsonGrammar(schema), 'root ::= "{" "\\"supported\\": " ("\\"yes\\"" | "\\"no\\"") "}"');
});

test("Stop during an effect ends the question without asking Python for another step", async () => {
  const calls: string[] = [];
  let finishRerank!: (scores: number[]) => void;
  const python = {
    call: async (op: string) => {
      calls.push(op);
      if (op === "start") return JSON.stringify({ effect: "Rerank", query: "q", texts: ["a"] });
      if (op === "step") return JSON.stringify({ end: true, record: { path: "answered" } });
      return null;
    },
  };
  const models = { call: () => new Promise<number[]>(resolve => { finishRerank = resolve; }) };
  const engine = { interruptGenerate: () => undefined };
  const driver = new Driver(python as any, models as any, engine as any, settings);

  const asked = driver.ask({ query: "q", history: [], docs: null, grounding: true }, { onEvent: () => undefined, onSpan: () => undefined });
  await new Promise(resolve => setTimeout(resolve, 0));
  await driver.stop();
  finishRerank([0.9]);

  assert.equal(await asked, null);
  assert.deepEqual(calls, ["start", "stop"]);
});

test("a checker problem goes back to the core as an error, so the answer fails open with the reason", async () => {
  const python = {
    call: async (op: string, args: any) => op === "start"
      ? JSON.stringify({ effect: "CheckSentence", judge: { messages: [], schema: {} } })
      : JSON.stringify({ end: true, record: { reply: JSON.parse(args.result) } }),
  };
  const driver = new Driver(python as any, {} as any, {} as any, settings);
  driver.checkProblem = "the grounding judge is still loading";

  const record = await driver.ask({ query: "q", history: [], docs: null, grounding: true }, { onEvent: () => undefined, onSpan: () => undefined });

  assert.deepEqual(record, { reply: { error: "the grounding judge is still loading" } });
});
