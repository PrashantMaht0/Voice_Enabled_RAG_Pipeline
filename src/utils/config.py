"""Reads config/config.yaml; explicit arguments always win."""
import os
from functools import lru_cache

import yaml

CONFIG_PATH = os.environ.get("VRAG_CONFIG", "config/config.yaml")


@lru_cache(maxsize=1)
def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        print(f"[Config] {path} not found — falling back to built-in defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(dotted_key: str, default=None):
    """get('retrieval.top_k_rerank', 3) -> value from config.yaml, or the default."""
    # Read the module attribute at call time so VRAG_CONFIG (and tests) can redirect it.
    node = load_config(CONFIG_PATH)
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def pick(explicit, dotted_key: str, default):
    """Explicit argument wins; otherwise config; otherwise the built-in default."""
    return explicit if explicit is not None else get(dotted_key, default)
