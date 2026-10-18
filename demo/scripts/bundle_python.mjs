// Bundles the Python the browser runs (src/pipeline/browser_files.txt plus demo/python/bridge.py) into
// demo/public/py.zip, which the Pyodide worker unpacks. Runs before dev and build.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { zipSync } from "fflate";

const demo = join(dirname(fileURLToPath(import.meta.url)), "..");
const root = join(demo, "..");
const files = readFileSync(join(root, "src/pipeline/browser_files.txt"), "utf8")
  .split("\n").map(l => l.trim()).filter(l => l && !l.startsWith("#"));
const entries = Object.fromEntries(files.map(f => [f, readFileSync(join(root, f))]));
entries["bridge.py"] = readFileSync(join(demo, "python/bridge.py"));
const zip = zipSync(entries, { level: 9, mtime: new Date("2026-01-01") });
writeFileSync(join(demo, "public/py.zip"), zip);
console.log(`py.zip: ${Object.keys(entries).length} files, ${(zip.length / 1024).toFixed(0)} KB`);
