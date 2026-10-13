"""config.yaml drives the modules: values reach them, explicit arguments win, missing keys fall back."""
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


def test_the_image_installs_torch_and_torchvision_from_the_same_cpu_index():
    """A PyPI torchvision is built for CUDA torch; beside CPU torch it broke `import timm`, so the
    table detector failed to load in Docker and table reading silently fell back to none (spec 0004)."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pins = dict(re.findall(r"^(torch|torchvision)==(\S+)", (root / "requirements.txt").read_text(), re.M))
    assert set(pins) == {"torch", "torchvision"}, "torch and torchvision must both be pinned"

    dockerfile = (root / "Dockerfile").read_text()
    cpu_install = dockerfile[dockerfile.index("download.pytorch.org/whl/cpu"):dockerfile.index("pip install --no-cache-dir -r")]
    assert "^torch==" in cpu_install and "^torchvision==" in cpu_install
