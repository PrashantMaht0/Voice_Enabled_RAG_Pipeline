"""The local Ollama client: streamed answers from the retrieved chunks, and follow up rewriting."""
from typing import Any, Dict, Generator, List, Optional

import ollama

from src.prompts.prompt_template import (
    ANSWER_STYLES,
    system_prompt,
    QUERY_REWRITE_PROMPT,
    FALLBACK_RESPONSE,
    build_rag_context_prompt,
    format_chat_history
)
from src.utils.config import pick

class LocalLLMClient:
    """Answers with `llm.generation_model` and rewrites follow ups with `llm.rewriter_model`, both at
    temperature 0 so the same question gets the same answer."""

    def __init__(
        self,
        generation_model: str = None,
        rewriter_model: str = None,
        host: Optional[str] = None
    ):
        self.generation_model = pick(generation_model, "llm.generation_model", "llama3.2:3b")
        self.rewriter_model = pick(rewriter_model, "llm.rewriter_model", "llama3.2:1b")
        self.temperature = pick(None, "llm.temperature", 0.0)
        self.max_tokens = pick(None, "llm.max_tokens", 256)
        self.answer_style = pick(None, "llm.answer_style", "full")
        if self.answer_style not in ANSWER_STYLES:
            print(f"[LLM] WARNING: unknown llm.answer_style {self.answer_style!r}, using 'full'")
            self.answer_style = "full"
        self.client = ollama.Client(host=host) if host else ollama
        
        # Stores the final token count after generation finishes
        self.last_eval_count = 0 

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Rewrites conversational queries into standalone queries."""
        if not chat_history:
            return query.strip()

        formatted_history = format_chat_history(chat_history)
        prompt = QUERY_REWRITE_PROMPT.format(history=formatted_history, query=query)

        try:
            response = self.client.chat(
                model=self.rewriter_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_predict": 64}
            )
            rewritten = response["message"]["content"].strip().strip('"')
            return rewritten if rewritten else query
        except Exception:
            return query

    def stream_generate(
        self,
        query: str,
        context_chunks: List[str]
    ) -> Generator[str, None, None]:
        """The answer's tokens as Ollama streams them; the fallback refusal when no chunk passed the gate.

        last_eval_count holds the number of generated tokens once the stream ends."""
        self.last_eval_count = 0
        
        if not context_chunks:
            yield FALLBACK_RESPONSE
            return

        system_instruction = system_prompt(self.answer_style).format(fallback=FALLBACK_RESPONSE)
        user_content = build_rag_context_prompt(query, context_chunks)

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        try:
            stream = self.client.chat(
                model=self.generation_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "top_p": 0.1,
                    "num_predict": self.max_tokens
                }
            )

            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                    
                if chunk.get("done", False):          # the last chunk carries the token count
                    self.last_eval_count = chunk.get("eval_count", 0)

        except Exception as e:
            yield f" Error processing response: {str(e)}"

    def generate(self, query: str, context_chunks: List[str], tracer: Optional[Any] = None) -> str:
        """The whole answer at once (the stream joined), timed as the "generation" span when a tracer is
        given. Used where a client has no stream_generate."""
        if tracer:
            tracer.start_span("generation")
        full_text = "".join(self.stream_generate(query, context_chunks)).strip()
        if tracer:
            tracer.end_span("generation", f"LLM call — {self.last_eval_count} tokens generated")
        return full_text