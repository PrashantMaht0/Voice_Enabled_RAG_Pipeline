from typing import List, Dict
from src.llm.llm_client import LocalLLMClient

class QueryRewriter:
    """Turns conversational follow-ups into standalone queries."""
    def __init__(self):
        # We rely on the lightweight 1B model defined in the LLMClient for speed
        self.llm_client = LocalLLMClient()
    
    def resolve_query(self, current_query: str, chat_history: List[Dict[str, str]]) -> str:
        """Resolves pronouns against chat history, skipping the first turn."""
        if not chat_history:
            return current_query
            
        return self.llm_client.rewrite_query(current_query, chat_history)