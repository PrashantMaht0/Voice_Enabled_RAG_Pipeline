"""Every setting the answer rules read, in one object: the server builds it from config.yaml, the
browser from config.json. The defaults are the ones the server used before the core existed."""
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class Settings:
    hybrid_alpha: float = 0.5
    top_k_retrieval: int = 30
    top_k_rerank_pool: int = 12
    top_k_rerank: int = 3
    confidence_threshold: float = 0.65
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens: int = 256
    answer_style: str = "full"
    context_window: int = 4096
    checker: Optional[str] = None            # the grounding checker's name ("nli:…", "judge:…"); None = unchecked
    grounding_threshold: float = 0.5
    max_query_chars: int = 1000
    max_history_turns: int = 6
    max_turn_chars: int = 2000

    @classmethod
    def from_config(cls, get) -> "Settings":
        """From config.yaml through `src.utils.config.get`."""
        keys = {
            "hybrid_alpha": "retrieval.hybrid_alpha", "top_k_retrieval": "retrieval.top_k_retrieval",
            "top_k_rerank_pool": "retrieval.top_k_rerank_pool", "top_k_rerank": "retrieval.top_k_rerank",
            "confidence_threshold": "retrieval.confidence_threshold", "temperature": "llm.temperature",
            "top_p": "llm.top_p", "max_tokens": "llm.max_tokens", "answer_style": "llm.answer_style",
            "context_window": "llm.context_window", "grounding_threshold": "grounding.threshold",
            "max_query_chars": "api.max_query_chars", "max_history_turns": "api.max_history_turns",
            "max_turn_chars": "api.max_turn_chars",
        }
        defaults = cls()
        return cls(**{name: get(key, getattr(defaults, name)) for name, key in keys.items()})

    @classmethod
    def from_dict(cls, values: dict) -> "Settings":
        """From the browser's config.json (unknown keys are ignored)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    def as_dict(self) -> dict:
        return asdict(self)
