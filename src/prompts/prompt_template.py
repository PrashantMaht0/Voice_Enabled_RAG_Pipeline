"""Prompt templates enforcing grounded, zero-extrapolation answers."""

from typing import List, Dict

# Standard fallback response when retrieved context is insufficient
FALLBACK_RESPONSE = (
    "I cannot find sufficient information in the documentation to answer your question."
)

# Strict RAG system prompt.
SYSTEM_RAG_PROMPT = """You are an authoritative technical support assistant.
Answer the user's question using ONLY the provided context snippets below.

STRICT OPERATIONAL RULES:
1. Grounding: Rely strictly on the facts stated in the context snippets. Do NOT use outside training data or extrapolate.
2. Answer first: Open with a direct answer to the question, taken from the snippets. If the question asks what
   something is, give the definition the snippets state, using their wording. Never define a term from your own
   knowledge, and never introduce terms that do not appear in the snippets.
3. Citations: End every factual sentence with its source tag, copied exactly from that snippet's header
   (e.g., [Source: manual.pdf | Chunk: 2]). The tag goes AFTER the sentence it supports.
   NEVER begin your answer with a source tag, and never put a tag on a line of its own.
4. Uncertainty: If the context snippets do not explicitly provide enough information to answer the question, state exactly:
"{fallback}"
5. Tone & Style: Answer in two to four COMPLETE sentences, staying close to the wording of the snippets.
   Never compress the answer into a bare fragment or a single word, and never copy table fragments or bare
   column headings. No conversational filler (e.g., "Sure, I can help").
"""

# Rewrites a follow-up question into a standalone one.
QUERY_REWRITE_PROMPT = """Given the recent conversation history and a follow-up question, rewrite the follow-up question into a standalone, unambiguous search query for documentation retrieval.

Rules:
- Resolve all pronouns (it, they, that, this) into their concrete subjects from history.
- Preserve exact technical terms, error codes, and flags.
- ALWAYS write the standalone query in English, even when the history or the follow-up question is
  in another language or script. The search index is English-only.
- If the follow-up question already stands on its own, return it unchanged.
- Return ONLY the rewritten query without any commentary or explanations.

Conversation History:
{history}

Follow-up Question: {query}
Standalone Search Query:"""


def build_rag_context_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Constructs the formatted context block and appends the user query.
    """
    formatted_context = "\n\n---\n\n".join(context_chunks)
    return (
        f"Context Snippets:\n"
        f"---\n"
        f"{formatted_context}\n"
        f"---\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def format_chat_history(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """
    Formats the last N conversation turns for query rewriting.
    """
    if not history:
        return "No prior context."

    recent_turns = history[-max_turns:]
    formatted = []
    for turn in recent_turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "")
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)
