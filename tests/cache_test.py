"""Two-level query cache: exact, semantic, eviction and corpus invalidation."""
import numpy as np
import pytest

from src.utils.query_cache import QueryCache, corpus_fingerprint


class _StubEmbedder:
    """Deterministic embeddings: same first word -> near-identical vector."""
    def embed_text(self, text):
        head = text.lower().split()[0] if text.split() else ""
        rng = np.random.default_rng(abs(hash(head)) % (2**32))
        base = rng.normal(size=16)
        # tiny perturbation keyed on full text, so paraphrases stay close but distinct
        jitter = np.random.default_rng(abs(hash(text)) % (2**32)).normal(size=16) * 0.01
        return (base + jitter).tolist()


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "manual.pdf").write_text("original")
    return tmp_path


def make_cache(corpus, **kwargs):
    kwargs.setdefault("embedder", _StubEmbedder())
    return QueryCache(raw_dir=str(corpus), **kwargs)


PAYLOAD = {"answer": "A resource group is a container.", "confidence_score": 0.99}


def test_exact_hit(corpus):
    cache = make_cache(corpus)
    cache.put("What is a resource group?", PAYLOAD)

    payload, level = cache.get("What is a resource group?")

    assert level == "exact"
    assert payload["answer"] == PAYLOAD["answer"]


def test_exact_match_ignores_case_and_spacing(corpus):
    cache = make_cache(corpus)
    cache.put("What is a Resource Group?", PAYLOAD)

    _, level = cache.get("  what is a   RESOURCE group?  ")

    assert level == "exact"


def test_semantic_hit_for_a_paraphrase(corpus):
    cache = make_cache(corpus, similarity_threshold=0.9)
    cache.put("resource group explained", PAYLOAD)

    payload, level = cache.get("resource group meaning")

    assert level == "semantic"
    assert payload["answer"] == PAYLOAD["answer"]


def test_unrelated_query_misses(corpus):
    cache = make_cache(corpus, similarity_threshold=0.9)
    cache.put("resource group explained", PAYLOAD)

    payload, level = cache.get("kubernetes ingress controllers")

    assert payload is None and level is None


def test_changed_source_document_invalidates_immediately(corpus):
    """A stale answer must not outlive the document it came from."""
    cache = make_cache(corpus)
    cache.put("What is a resource group?", PAYLOAD)
    assert cache.get("What is a resource group?")[1] == "exact"

    (corpus / "manual.pdf").write_text("rewritten content, different size")

    assert cache.get("What is a resource group?") == (None, None)
    assert cache.stats()["invalidations"] == 1


def test_new_and_removed_documents_invalidate(corpus):
    cache = make_cache(corpus)
    cache.put("q", PAYLOAD)
    (corpus / "second.pdf").write_text("added")
    assert cache.get("q") == (None, None)

    cache.put("q", PAYLOAD)
    (corpus / "second.pdf").unlink()
    assert cache.get("q") == (None, None)


def test_explicit_invalidation(corpus):
    cache = make_cache(corpus)
    cache.put("q", PAYLOAD)

    cache.invalidate("test")

    assert cache.get("q") == (None, None)
    assert cache.stats()["entries"] == 0


def test_lru_eviction_bounds_memory(corpus):
    """Distinct topics, so the semantic layer cannot mask eviction."""
    topics = ["alpha", "bravo", "charlie", "delta", "echo"]
    cache = make_cache(corpus, max_entries=3, similarity_threshold=0.999)
    for topic in topics:
        cache.put(f"{topic} configuration guide", {"answer": topic})

    assert cache.stats()["entries"] == 3
    assert cache.get("alpha configuration guide") == (None, None)     # evicted
    assert cache.get("echo configuration guide")[1] == "exact"        # retained


def test_semantic_layer_can_serve_a_neighbour_after_exact_eviction(corpus):
    """After eviction, a semantically equivalent entry still answers."""
    cache = make_cache(corpus, max_entries=2, similarity_threshold=0.9)
    cache.put("resource group explained", {"answer": "container"})
    cache.put("resource group meaning", {"answer": "container"})
    cache.put("resource group definition", {"answer": "container"})

    payload, level = cache.get("resource group explained")

    assert level == "semantic"
    assert payload["answer"] == "container"


def test_disabled_cache_never_serves(corpus):
    cache = make_cache(corpus, enabled=False)
    cache.put("q", PAYLOAD)

    assert cache.get("q") == (None, None)
    assert cache.stats()["entries"] == 0


def test_stats_track_hit_rate(corpus):
    cache = make_cache(corpus, similarity_threshold=0.99)
    cache.put("alpha query", PAYLOAD)
    cache.get("alpha query")                 # exact
    cache.get("zeta unrelated question")     # miss

    stats = cache.stats()
    assert stats["hits_exact"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_fingerprint_is_stable_and_content_sensitive(corpus):
    first = corpus_fingerprint(str(corpus))
    assert first == corpus_fingerprint(str(corpus))

    (corpus / "manual.pdf").write_text("changed content entirely")
    assert corpus_fingerprint(str(corpus)) != first


def test_works_without_an_embedder(corpus):
    """Exact caching must still function if embeddings are unavailable."""
    cache = QueryCache(embedder=None, raw_dir=str(corpus))
    cache.put("q", PAYLOAD)

    assert cache.get("q")[1] == "exact"
    assert cache.get("something else") == (None, None)
