import { defineConfig } from "vite";

// Pre-bundle the model libraries, so the first worker import does not trigger a dependency reload.
export default defineConfig({
  optimizeDeps: { include: ["@mlc-ai/web-llm", "@huggingface/transformers", "kokoro-js", "onnxruntime-web"] },
  worker: { format: "es" },
});
