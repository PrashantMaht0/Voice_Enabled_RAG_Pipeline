// Runs the tests/pipeline tests inside Pyodide, with only the files the browser loads (src/pipeline/browser_files.txt).
// Usage (from the repo root): npm --prefix demo/pyodide-runner test
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const files = readFileSync(join(root, "src/pipeline/browser_files.txt"), "utf8")
  .split("\n").map(l => l.trim()).filter(l => l && !l.startsWith("#"));
const tests = ["tests/pipeline/core_test.py", "tests/pipeline/demo_index_test.py"];
files.push("demo/py/bridge.py", ...tests);

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "pyyaml", "pytest"]);
for (const file of files) {
  const target = `/work/${file}`;
  pyodide.FS.mkdirTree(dirname(target));
  pyodide.FS.writeFile(target, readFileSync(join(root, file)));
}
const code = await pyodide.runPythonAsync(`
import os, sys
os.chdir("/work")
sys.path.insert(0, "/work")
import pytest
pytest.main(["-q", "-p", "no:cacheprovider", "--rootdir", "/work", ${tests.map(t => JSON.stringify(t)).join(", ")}])
`);
process.exit(Number(code));
