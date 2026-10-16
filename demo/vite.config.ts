import { defineConfig } from "vite";

// GitHub Pages serves the site under the repository name; VITE_BASE overrides it (spec 0009).
export default defineConfig({
  base: process.env.VITE_BASE ?? "/Voice_Enabled_RAG_Pipeline/",
  optimizeDeps: { include: ["@mlc-ai/web-llm", "@huggingface/transformers"] },
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 8000 },
});
