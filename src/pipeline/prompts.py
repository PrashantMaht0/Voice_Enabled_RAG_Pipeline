"""The messages the answer and rewrite models receive, and how a rewrite reply is cleaned (spec 0009)."""
from typing import Dict, List

from src.prompts.prompt_template import (FALLBACK_RESPONSE, QUERY_REWRITE_PROMPT, build_rag_context_prompt,
                                         format_chat_history, system_prompt)


def answer_messages(query: str, chunks: List[str], answer_style: str) -> List[Dict[str, str]]:
    """The system prompt for the answer style, then the retrieved chunks and the question."""
    return [{"role": "system", "content": system_prompt(answer_style).format(fallback=FALLBACK_RESPONSE)},
            {"role": "user", "content": build_rag_context_prompt(query, chunks)}]


def rewrite_messages(query: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """One user message asking for a standalone version of a follow up question."""
    prompt = QUERY_REWRITE_PROMPT.format(history=format_chat_history(history), query=query)
    return [{"role": "user", "content": prompt}]


def clean_rewrite(reply, query: str) -> str:
    """The rewrite model's reply without surrounding quotes; the question as asked when the reply is empty."""
    rewritten = (reply or "").strip().strip('"')
    return rewritten if rewritten else query
