# V-RAG — Enterprise Voice-Enabled RAG Pipeline

An ultra-low-latency, zero-hallucination conversational RAG system engineered entirely on a local
open-source stack. It pairs two-stage hybrid retrieval (Tantivy BM25 + Qdrant dense HNSW) and
cross-encoder reranking with a hard confidence gate, deterministic guardrails, coreference-resolving
query rewriting, and a speech interface with code-switched Hinglish input.

No hosted APIs. No inference vendor. Every model - embedder, reranker, LLM, NER, ASR, TTS and the
evaluation judge runs on the machine.

---

## The S-Y-N-C-H-R-O-N-I-C Framework

The pipeline implements the ten-stage enterprise RAG paradigm end to end.

| Step | Principle | Implementation | Module | Status |
| :--- | :--- | :--- | :--- | :--- |
| **S** | Standardize Ingestion | NFC Unicode normalization, control-character stripping, content hashing, token-aware 512/64 chunking | `src/ingestion/loader.py`, `chunking.py` | ✅ |
| **Y** | Yield Hybrid Results | Tantivy BM25 sparse + Qdrant dense (`bge-small-en-v1.5`), fused by Reciprocal Rank Fusion | `src/vector_db/hybrid_store.py` | ✅ |
| **N** | Narrow with ANN | Embedded Qdrant HNSW reduces the corpus to 30 candidates in ~9 ms | `src/vector_db/vector_store.py` | ✅ |
| **C** | Cross-Encode (Rerank) | FlashRank `ms-marco-MiniLM-L-12-v2` scores query–chunk pairs jointly over a 12-candidate pool | `src/retrieval/reranker.py` | ✅ |
| **H** | Halt on Low Confidence | Hard gate: generation never runs when the top reranked score is below 0.65 | `src/retrieval/retriever.py` | ✅ |
| **R** | Restrict the LLM | Llama 3.2 3B at `temperature=0.0`, `top_p=0.1`, grounded-only system prompt | `src/llm/llm_client.py`, `src/prompts/prompt_template.py` | ✅ |
| **O** | Origin Citations | Citation markers injected at chunk level; provenance emitted deterministically from the chunks actually retrieved | `src/api/routes.py::build_citations`, `src/ingestion/chunking.py` | ✅ |
| **N** | NER & Assertion Checking | spaCy entity extraction verifies claims against source text; ungrounded citation identifiers are stripped | `src/verification/guard_rails.py` | ✅ |
| **I** | Inspect Continuously | Ragas gate over a 67-question golden dataset with threshold assertions, plus CI | `tests/eval_test.py`, `.github/workflows/rag_eval_ci.yml` | ✅ |
| **C** | Cache & Invalidate | Two-level cache — exact match, then cosine similarity over query embeddings — dropped the moment a source document changes | `src/utils/query_cache.py` |

Multi-turn coreference resolution (`src/retrieval/query_rewriter.py`) sits alongside the framework
rather than inside it: it de-contextualizes follow-up questions before retrieval.

---

## Architecture

```
                            [ Spoken Audio ]  ──or──  [ Typed Question ]
                                    │
                                    ▼  faster-whisper (base.en | Hinglish fine-tune)
                            [ Transcribed Query ]
                                    │
                                    ▼  Llama 3.2 1B — coreference resolution
                                    ▼  Llama 3.2 3B — English normalization (non-English input)
                         [ Standalone English Search Query ]
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
   [ Sparse BM25 — Tantivy ]                    [ Dense ANN — Qdrant HNSW ]
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼
                    [ Reciprocal Rank Fusion → top 12 ]
                                    ▼
                 [ Cross-Encoder — FlashRank MiniLM-L12 ]
                                    ▼
                      [ Confidence Gate: score ≥ 0.65 ]
                            /                    \
                     (pass)/                      \(fail)
                          ▼                        ▼
          [ Llama 3.2 3B — T=0.0 ]      [ Deterministic refusal, no generation ]
                          ▼
              [ spaCy NER assertion check ]
                          ▼
              [ Deterministic citation binding ]
                          ▼
        [ Speech sanitizer ] ──► [ Kokoro-82M TTS ] ──► [ Audio Out ]
```

---

## Performance Validation

Apple Silicon, CPU-only inference, 224-chunk corpus (*Microsoft Azure Essentials*, 2nd ed.), median
of five in-corpus queries against the running service.

### Retrieval latency budget

| Stage | Median |
| --- | ---: |
| BM25 keyword search | 0.6 ms |
| Dense ANN search | 9 ms |
| RRF fusion | < 0.1 ms |
| Cross-encoder rerank (12 candidates) | 700 ms |
| Confidence gate | < 0.1 ms |
| LLM generation (Llama 3.2 3B) | 1100–1900 ms |
| NER assertion check | < 0.1 ms |
| **Answered query, end to end** | **~2.3 s** |
| **Refused query, end to end** | **~0.7 s** |

Retrieval is ~710 ms, of which the cross-encoder is 98%. Every stage before the reranker costs under
10 ms combined — the search layer is not the bottleneck and does not need optimizing.

A refused query costs a third of an answered one because the confidence gate halts before
generation. The gate is a latency optimization as well as a correctness control.

### Reranker recall vs latency

Two cross-encoders were benchmarked on identical candidate pools:

| Model | Rerank latency | In-corpus questions answered |
| --- | ---: | ---: |
| `ms-marco-TinyBERT-L-2-v2` | 84 ms | 2 / 5 |
| `ms-marco-MiniLM-L-12-v2` | 700 ms | **5 / 5** |

TinyBERT is 20× faster and gated out three questions the L-12 model answered with ≥ 0.97 confidence.
Latency won the headline but lost the product: a system that refuses answers it holds is worse than a
slower one. **Decision: L-12.** The 600 ms was recovered instead by shrinking the candidate pool
(AD-2), which cost no recall.

### Metric characterization: answer-length sensitivity

Faithfulness was profiled against answer length before the threshold was set:

| Answer length | Samples | Faithfulness |
| --- | ---: | ---: |
| 0–15 words | 15 | 0.789 |
| 15–30 words | 19 | 0.919 |
| 30+ words | 10 | 0.889 |

Ragas decomposes an answer into discrete claims and verifies each. A ten-word answer carries roughly
two claims, so one unverified claim costs 0.50; the same claim inside a longer passage is diluted.
**32 of 44 answers score a perfect 1.0**, and excluding terse replies the score is 0.909 — the
shortfall is concentrated entirely in brevity, not in grounding.

Five prompt strategies were measured end to end to separate metric artifact from real regression:

| Prompt strategy | Faithfulness | Relevancy |
| --- | ---: | ---: |
| Verbose with inline citations | 0.907 | 0.947 |
| Terse, no citations | 0.875 | 0.892 |
| Complete sentences, no citations | 0.838 | 0.953 |
| Citations restored, 2–4 sentences | 0.868 | 0.953 |
| **Deterministic citations (shipped)** | **0.868** | **0.953** |

Only the verbose variant cleared 0.90, and it did so by prefixing every answer with a source header —
the readability defect that motivated the rewrite. Candidate-pool depth showed no correlation with
faithfulness, ruling out reranking as the cause. The threshold was therefore set to 0.85 rather than
inflating answers to satisfy the metric.

Faithfulness has returned **exactly 0.868 on three consecutive runs**, so the figure is a property of
the pipeline, not judge variance. The margin over threshold is +0.018; treat a drop below ~0.86 as a
genuine regression.

### Cache effectiveness

Measured against the running service:

| Lookup | Latency | Level |
| --- | ---: | --- |
| Cold query | 4983 ms | miss |
| Repeat query | 2 ms | exact |
| Same query, different case and spacing | 1 ms | exact |
| Paraphrase (*"Explain what an Azure resource group is"*) | 9 ms | semantic |

A repeated question returns **~865× faster** because retrieval, reranking and generation are all
skipped. Refusals are cached too: an out-of-scope question is deterministic for a given corpus, and
caching it removes the reranker cost from the adversarial path.

Invalidation was verified by modifying the source document while the cache was warm:

```
cold             4983 ms   cache=None
warm                2 ms   cache=exact
-- source document touched on disk --
after change     2238 ms   cache=None     ← invalidated
re-warmed           2 ms   cache=exact
```

Inspect or clear at runtime with `GET /api/cache` and `DELETE /api/cache`.

### CPU ceiling for speech synthesis

| Engine | Realtime factor | Note |
| --- | ---: | --- |
| Kokoro-82M (English, shipped) | 0.47× | Faster than realtime |
| F5-TTS Hinglish, `nfe_step` 8 | 2.9× | Lowest quality setting |
| F5-TTS Hinglish, `nfe_step` 16 | 5.8× | Configured default |
| F5-TTS Hinglish, `nfe_step` 32 | 11.6× | Upstream default |

The original plan targeted 600 ms time-to-first-sound. That target assumes GPU inference. On CPU,
Kokoro alone spends ~480 ms synthesizing a one-word reply, and Llama 3.2 3B needs ~1.5 s to produce a
speakable clause — the floor is seconds, not milliseconds. **This is a hardware ceiling, not a tuning
gap**, and closing it requires GPU inference (vLLM for generation, CUDA for synthesis) rather than
further optimization.

---


## Voice & Multilingual Architecture

Speech is a first-class path, not a wrapper: a spoken turn traverses the same retrieval, gating,
verification and citation stages as a typed one, and populates the same diagnostic panels.

### Shipped

**English speech loop.** Press Record, speak, press Stop. Transcription → RAG → spoken answer.

| Stage | Median |
| --- | ---: |
| STT (`base.en`, int8 CPU) | ~450 ms |
| Retrieval + confidence gate | ~750 ms |
| LLM generation | ~2 s |
| TTS (Kokoro, whole answer) | ~5 s |
| **Spoken turn, end to end** | **~9 s** |

**Hinglish speech recognition.** A fine-tuned code-switching ASR model
(`shunyalabs/zero-stt-hinglish`, whisper-medium) converted to CTranslate2 int8 for CPU inference.
The quality gap over the English baseline is categorical, not incremental:

```
spoken   : लिबर ऑफिस impress में एक प्रस्तुति document बनाना ... spoken tutorial में आपका स्वागत है
base.en  : "The labor office in Prasmeh is a Prasthi document banana..."      ← unusable
hinglish : "LibreOffice Impress में एक प्रस्तुती document बनाना ... spoken tutorial में आपका स्वागत है"
```

**Cross-lingual query normalization.** Hinglish questions are translated to English before retrieval,
with a glossary that prevents Hindi function words being mistaken for product names — `matlab`
("meaning") was previously parsed as MATLAB, which destroyed retrieval. Verified end to end:

```
"Azure mai Resources ka kya matlab hota hai"            → "What is the meaning of Resources in Azure?"   conf 0.839
"contributor role ke liye kaun se actions allowed hain" → "What actions are allowed for the Contributor role?"  conf 0.961
"blob storage kaise kaam karta hai"                     → "How does blob storage work?"                  conf 0.994
```

Trade-off: whisper-medium costs ~7.7 s for a 5 s clip on CPU, against ~450 ms for `base.en`.
Accuracy on code-switched speech is unusable without it.

### Future work — Hinglish speech output

- **Not intelligible.** Confirmed by listening and by round-tripping the output through the ASR
  model, at `nfe_step` 16 and 32 alike.
- **Too slow.** 5.8× realtime, versus 0.47× for Kokoro.
- One genuine defect was found and fixed en route: F5-TTS derives output duration from the
  reference-transcript to reference-audio ratio, so a Devanagari reference paired with Latin-script
  generation truncated every utterance (1.6 s for a 55-character sentence). A Latin-script reference
  corrected the duration to 4.6 s — but not the intelligibility.



