import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// GitHub Pages serves the site under the repository name; VITE_BASE overrides it (spec 0009).
export default defineConfig({
  base: process.env.VITE_BASE ?? "/Voice_Enabled_RAG_Pipeline/",
  // kokoro-js brings its own transformers.js; the speech worker sets that copy's download path (revision pin)
  resolve: { alias: { "kokoro-transformers": fileURLToPath(new URL("./node_modules/kokoro-js/node_modules/@huggingface/transformers", import.meta.url)) } },
  optimizeDeps: { include: ["@mlc-ai/web-llm", "@huggingface/transformers", "@gradio/client", "kokoro-js", "kokoro-transformers"] },
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 8000 },
});
