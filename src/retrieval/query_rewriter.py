"""Follow up rewriting: a question that leans on the conversation becomes one that stands alone."""
from typing import Dict, List

from src.llm.llm_client import LocalLLMClient


class QueryRewriter:
    """Resolves pronouns and references against the chat history with the small rewriter model
    (`llm.rewriter_model`), so retrieval searches for what the user actually means."""

    def __init__(self):
        self.llm_client = LocalLLMClient()

    def resolve_query(self, current_query: str, chat_history: List[Dict[str, str]]) -> str:
        """The standalone question; the first turn (no history) is returned unchanged."""
        if not chat_history:
            return current_query
        return self.llm_client.rewrite_query(current_query, chat_history)
