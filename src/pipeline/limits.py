"""Question size limits: one request cannot send a huge prompt to the models."""
from typing import Dict, List


class QuestionTooLarge(ValueError):
    """A question or history over the limits: shown to the asker with its reason (FastAPI answers 422)."""


def check_query(query: str, max_chars: int) -> str:
    if len(query) > max_chars:
        raise QuestionTooLarge(f"the question is longer than {max_chars} characters")
    return query


def check_history(history: List[Dict[str, str]], max_turns: int, max_turn_chars: int) -> List[Dict[str, str]]:
    if len(history) > max_turns:
        raise QuestionTooLarge(f"history has more than {max_turns} turns")
    for turn in history:
        if turn.get("role") not in ("user", "assistant"):
            raise QuestionTooLarge("each history turn needs role 'user' or 'assistant'")
        if len(turn.get("content", "")) > max_turn_chars:
            raise QuestionTooLarge(f"a history turn is longer than {max_turn_chars} characters")
    return history
