from typing import List, Dict
from src.llm.llm_client import LocalLLMClient

class QueryRewriter:
    """
    Transforms multi-turn conversational queries into standalone context-free queries
    to prevent vector search degradation on pronouns (e.g., 'how do I reset it?').
    """
    def __init__(self):
        # We rely on the lightweight 1B model defined in the LLMClient for speed
        self.llm_client = LocalLLMClient()
    
    def resolve_query(self, current_query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Takes the current query and history, and yields a dense standalone string.
        Bypasses LLM rewriting if this is the first turn.
        """
        if not chat_history:
            return current_query
            
        return self.llm_client.rewrite_query(current_query, chat_history)