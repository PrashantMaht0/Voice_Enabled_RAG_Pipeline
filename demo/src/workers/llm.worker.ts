// The answer model (spec 0009): WebLLM in its own worker; the page drives it through CreateWebWorkerMLCEngine.
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (message: MessageEvent) => handler.onmessage(message);
