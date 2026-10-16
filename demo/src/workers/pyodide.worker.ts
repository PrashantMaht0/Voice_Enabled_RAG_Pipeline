// The Python side of the demo (spec 0009): Pyodide running src/pipeline/core.py and src/demo_index.py through
// demo/py/bridge.py. Every call passes JSON strings, so no live Python object crosses into the page.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

let pyodide: any = null;
let bridge: any = null;

type Call = { id: number; op: string; args: any };
const reply = (id: number, body: object) => (self as unknown as Worker).postMessage({ id, ...body });

self.onmessage = async (event: MessageEvent<Call>) => {
  const { id, op, args } = event.data;
  try {
    if (op === "load") {
      const progress = (step: string, value: number) => reply(id, { progress: { step, value } });
      progress("Python runtime", 0.1);
      const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`);
      pyodide = await loadPyodide({ indexURL: PYODIDE });
      progress("numpy and pyyaml", 0.5);
      await pyodide.loadPackage(["numpy", "pyyaml"]);
      progress("answer core", 0.8);
      const zip = await (await fetch(args.pythonUrl)).arrayBuffer();
      pyodide.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
      pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
      bridge = pyodide.pyimport("bridge");
      bridge.configure(args.config);
      progress("sample index", 0.9);
      const count = bridge.load_sample(args.chunks, new Uint8Array(args.vectors));
      reply(id, { result: count });
    } else if (op === "start") {
      reply(id, { result: bridge.start(args.request, args.docs, args.grounding) });
    } else if (op === "step") {
      reply(id, { result: bridge.step(args.result) });
    } else if (op === "add_doc") {
      bridge.add_doc(args.name, args.chunks, new Uint8Array(args.vectors));
      reply(id, { result: null });
    } else if (op === "remove_doc") {
      bridge.remove_doc(args.name);
      reply(id, { result: null });
    } else if (op === "clear_uploads") {
      bridge.clear_uploads();
      reply(id, { result: null });
    } else if (op === "speech") {
      reply(id, { result: bridge.speech(args.sentence) });
    } else if (op === "heard_nothing") {
      reply(id, { result: bridge.heard_nothing(args.transcript) });
    } else if (op === "metrics") {
      reply(id, { result: bridge.metrics(args.rows, args.spans) });
    } else if (op === "stop") {
      bridge?.stop();
      reply(id, { result: null });
    } else {
      throw new Error(`unknown op ${op}`);
    }
  } catch (error) {
    reply(id, { error: String(error) });
  }
};

export {};
