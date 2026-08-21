from typing import Any, Generator, List, Dict, Optional
import ollama
from src.prompts.prompt_template import (
    SYSTEM_RAG_PROMPT,
    QUERY_REWRITE_PROMPT,
    NORMALIZE_QUERY_PROMPT,
    FALLBACK_RESPONSE,
    build_rag_context_prompt,
    format_chat_history
)
from src.utils.config import pick

class LocalLLMClient:
    """Local Ollama client for streaming, temperature-0 generation."""

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
        except Exception as e:
            return query

    def normalize_query(self, query: str) -> str:
        """Turns a Hinglish question into an English search query."""
        try:
            response = self.client.chat(
                model=self.generation_model,
                messages=[{"role": "user", "content": NORMALIZE_QUERY_PROMPT.format(query=query)}],
                options={"temperature": self.temperature, "num_predict": 64}
            )
            return self._first_usable_line(response["message"]["content"], query)
        except Exception:
            return query

    @staticmethod
    def _first_usable_line(reply: str, fallback: str) -> str:
        """Extracts a single query line, falling back when the model rambles."""
        text = (reply or "").strip()
        if "English:" in text:                       # model repeated the template
            text = text.split("English:")[-1]

        for line in text.splitlines():
            line = line.strip().strip('"').strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith(("here is", "here are", "sure", "the rewritten")):
                continue
            if line[0].isdigit() and line[1:3] in (". ", ") "):   # numbered list item
                line = line[3:].strip()
            if 3 < len(line) <= 200:
                return line
        return fallback

    def stream_generate(
        self,
        query: str,
        context_chunks: List[str]
    ) -> Generator[str, None, None]:
        """Yields tokens individually and captures metadata when finished."""
        self.last_eval_count = 0
        
        if not context_chunks:
            yield FALLBACK_RESPONSE
            return

        system_instruction = SYSTEM_RAG_PROMPT.format(fallback=FALLBACK_RESPONSE)
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
                    
                # Ollama returns metadata on the final streaming chunk
                if chunk.get("done", False):
                    # eval_count represents the number of output tokens generated
                    self.last_eval_count = chunk.get("eval_count", 0)

        except Exception as e:
            yield f" Error processing response: {str(e)}"

    def generate(self, query: str, context_chunks: List[str], tracer: Optional[Any] = None) -> str:
        """
        Synchronous generation helper that optionally integrates with the PipelineTracer.
        """
        if tracer: tracer.start_span("generation")
        
        tokens = list(self.stream_generate(query, context_chunks))
        full_text = "".join(tokens).strip()
        
        if tracer: 
            tracer.end_span("generation", f"LLM call — {self.last_eval_count} tokens generated")
            
        return full_text