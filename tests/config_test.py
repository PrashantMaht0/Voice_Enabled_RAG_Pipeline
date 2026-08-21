import textwrap

import pytest

from src.utils import config as cfg
from src.ingestion.chunking import DocumentChunker


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    """Points the loader at a throwaway YAML file."""
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent("""
        ingestion:
          chunk_size: 128
          chunk_overlap: 8
        retrieval:
          confidence_threshold: 0.42
    """))
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(path))
    cfg.load_config.cache_clear()
    yield path
    cfg.load_config.cache_clear()


def test_config_values_reach_the_modules(custom_config):
    """config.yaml must actually drive behaviour, not just document it."""
    chunker = DocumentChunker()

    assert chunker.chunk_size == 128
    assert chunker.overlap == 8


def test_explicit_arguments_still_win(custom_config):
    assert DocumentChunker(chunk_size=999).chunk_size == 999


def test_missing_keys_fall_back_to_defaults(custom_config):
    assert cfg.get("retrieval.confidence_threshold") == 0.42
    assert cfg.get("retrieval.nonexistent_key", "fallback") == "fallback"
    assert cfg.get("entirely.missing.path", 7) == 7
