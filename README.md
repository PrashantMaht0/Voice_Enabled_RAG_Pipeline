# V-RAG: Enterprise Voice-Enabled RAG Pipeline

A low-latency, zero-hallucination question-answering system for your own documents. Ask by typing or
by speaking, in English or Hinglish(Hindi + English), and get an answer drawn only from the documents you supplied —
with the sources it used.

Everything runs on your machine. No cloud APIs, no vendor keys, no data leaving the laptop.

---

## The S-Y-N-C-H-R-O-N-I-C Framework

| Step | Principle | Definition |
| :--- | :--- | :--- |
| **S** | Standardize Ingestion | Clean, normalize Unicode, strip artifacts, and hash all incoming documents to ensure identical processing. |
| **Y** | Yield Hybrid Results | Combine dense vector embeddings with sparse keyword matching (BM25) to capture both semantic meaning and exact phrases or clause numbers. |
| **N** | Narrow with ANN | Use Approximate Nearest Neighbor algorithms (HNSW, ScaNN) to rapidly down-select millions of vectors into a manageable candidate pool in milliseconds. |
| **C** | Cross-Encode (Rerank) | Pass the candidate pool through a heavy cross-encoder that scores queries and chunks jointly, sorting the best context to the top. |
| **H** | Halt on Low Confidence | Calculate a source confidence score from retrieval math, freshness and authority. Gate the pipeline so it halts and refuses to answer when the context isn't trustworthy. |
| **R** | Restrict the LLM | Use hard, explicit system prompt constraints and temperature 0.0 to strip out creativity and enforce strict compliance. |
| **O** | Origin Citations | Mandate that every claim in the generated output maps back to an explicit source document, page number and paragraph offset. |
| **N** | NER & Assertion Checking | Run automated post-processing using Named Entity Recognition to pull factual claims from the response and verify they are grounded in the source text. |
| **I** | Inspect Continuously | Monitor quality metrics continuously — context relevance, faithfulness and answer relevance — to catch performance drift immediately. |
| **C** | Cache & Invalidate | Implement an exact and semantic two-level caching layer to save latency and cost, invalidating entries the second a source document changes. |

## Credits

This project is built on the S-Y-N-C-H-R-O-N-I-C framework described by **Vishal Mysore** in
[How to Design a RAG Pipeline for 10 Million Documents with Zero Hallucination](https://medium.com/@visrow/how-to-design-a-rag-pipeline-for-10-million-documents-with-zero-hallucination-live-demo-057e37bcdbf6).
The framework supplied the ten design principles above; this repository is an independent
implementation of them on a fully local stack.

---
## Project Screen Shots

- Main Console 
  ![Main Console](https://github.com/PrashantMaht0/Voice_Enabled_RAG_Pipeline/blob/main/assets/Main_dashboard.png)

- Corpus Tab
  ![Corpus Tab](https://github.com/PrashantMaht0/Voice_Enabled_RAG_Pipeline/blob/main/assets/Corpus_Tab.png)

- Engish Audio Query Result 
  ![English Query Result](https://github.com/PrashantMaht0/Voice_Enabled_RAG_Pipeline/blob/main/assets/English_query_res.png)

- Hinglish Audio Query Result
  ![Hinglish Query Result](https://github.com/PrashantMaht0/Voice_Enabled_RAG_Pipeline/blob/main/assets/Hinglish_query_res.png)

---
## Repository Layout

```
Voice_Enabled_RAG_Pipeline/
├── .github/
│   └── workflows/
│       └── rag_eval_ci.yml          # CI: fast tests on every PR, Ragas gate on demand
├── assest/
│   └── RAG_Architecture.png         # Architecture diagram
├── config/
│   └── config.yaml                  # Every tunable setting: models, thresholds, paths, cache
├── data/
│   ├── raw/                         # Your source documents (PDF, TXT, MD)
│   └── eval/
│       └── golden_dataset.json      # 67 evaluation questions (47 in-scope + 20 adversarial)
├── docs/
│   └── ragas_gate_run.txt           # Recorded output of a passing quality-gate run
├── src/
│   ├── api/
│   │   └── routes.py                # All HTTP endpoints: query, upload, corpus, cache, health
│   ├── ingestion/
│   │   ├── loader.py                # Unicode normalization (NFC) and document parsing
│   │   └── chunking.py              # Token-aware chunker with citation markers
│   ├── embeddings/
│   │   └── embedder.py              # BGE embedding client
│   ├── vector_db/
│   │   ├── hybrid_store.py          # Tantivy (BM25) + Qdrant (dense) search and fusion
│   │   └── vector_store.py          # Index lifecycle: create, ingest, reset
│   ├── retrieval/
│   │   ├── query_rewriter.py        # Resolves pronouns in multi-turn conversation
│   │   ├── reranker.py              # FlashRank cross-encoder scoring
│   │   └── retriever.py             # Reciprocal rank fusion and the 0.65 confidence gate
│   ├── prompts/
│   │   └── prompt_template.py       # Hard-constrained system prompts
│   ├── llm/
│   │   └── llm_client.py            # Streaming Ollama client, temperature 0.0
│   ├── voice/
│   │   ├── stt.py                   # Faster-Whisper transcription (English and Hinglish)
│   │   ├── tts.py                   # Kokoro-82M speech synthesis
│   │   └── tts_f5.py                # Experimental Hinglish speech synthesis
│   ├── verification/
│   │   └── guard_rails.py           # spaCy NER and citation grounding checks
│   └── utils/
│       ├── config.py                # Reads config.yaml
│       ├── query_cache.py           # Two-level cache with corpus invalidation
│       ├── text_sanitizer.py        # Strips citation markers before speech
│       ├── audio_utils.py           # Audio resampling and format conversion
│       └── metrics.py               # Per-stage latency tracing
├── tests/                           # 59 fast tests, 2 integration tests, 1 quality gate
├── app.py                           # Gradio dashboard entry point
├── main.py                          # FastAPI backend entry point
├── docker-compose.yml               # Ollama + app containers
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Architecture

![V-RAG architecture](https://github.com/PrashantMaht0/Voice_Enabled_RAG_Pipeline/blob/main/assets/RAG_Architecture.png)

A question travels through the pipeline in stages. Each stage can stop it: if the documents don't
contain a trustworthy answer, the system refuses rather than inventing one.

1. **Understand the question.** Speech is transcribed. Follow-up questions like *"how do I scale it?"*
   are rewritten into standalone ones, and non-English questions are translated for searching.
2. **Search twice.** A keyword search and a meaning-based search run in parallel, then their results
   are merged: one catches exact terms and error codes, the other catches paraphrases.
3. **Re-read the shortlist.** A slower, more careful model re-reads the top candidates against the
   question and reorders them.
4. **Decide whether to answer.** If the best passage scores below 0.65, the pipeline stops here and
   says it doesn't know. The language model is never called.
5. **Answer under constraint.** The model writes an answer using only the supplied passages, with
   creativity switched off.
6. **Check the answer.** Facts and names in the answer are verified against the source text, and the
   sources actually used are attached.

---

## How well does it work?

Four checks run against a set of 67 questions:  47 answerable from the documents, and 20 designed to
tempt the system into making something up.

| What is measured | Target | Result |
| --- | --- | --- |
| Answers supported by the documents | ≥ 85% | **87%** |
| Answers that address the question asked | ≥ 85% | **95%** |
| Trick questions correctly refused | 100% | **20 of 20** |
| Answerable questions actually answered | ≥ 90% | **94%** |

## How fast is it?

On a laptop CPU, with no GPU:

| | Time |
| --- | ---: |
| Typed question, first time | ~2.3 seconds |
| Typed question, asked again | **~2 milliseconds** |
| Question it refuses to answer | ~0.7 seconds |
| Spoken question, start to spoken answer | ~9 seconds |

---

## Voice and language support

**English** — press Record, speak, press Stop, and the answer is read back aloud.

**Hinglish** — Hindi and English mixed together, the way people actually speak. A dedicated speech
model handles it:

```
you say    : "Azure mai Resources ka kya matlab hota hai"
Hinglish model hears: "Azure mai Resources ka kya matlab hota hai"  
searched as: "What is the meaning of Resources in Azure?"
```

Hinglish questions are translated to English before searching, because the documents are in English.
The answer comes back in English and is read aloud in the English voice.

---

## Run the pipeline locally

**1. Install**

```bash
python -m venv dev-env
./dev-env/bin/pip install -r requirements.txt
./dev-env/bin/python -m spacy download en_core_web_sm
```

**2. Download the models**

The language model comes from [Ollama](https://ollama.com):

```bash
ollama pull llama3.2:3b        # writes the answers
ollama pull llama3.2:1b        # rewrites follow-up questions
```

Search models (embedding and re-ranking) download automatically the first time you ask a question.
For spoken answers, fetch the voice model:

```bash
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

For Hinglish speech input, see [Hinglish setup](#hinglish-setup).

**3. Add your documents**

Put PDF, TXT or MD files into `data/raw/`, or upload them later from the dashboard.

**4. Start it**

Two terminals, both from the project folder:

```bash
./dev-env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000   # the engine
./dev-env/bin/python app.py                                           # the dashboard
```

Open **http://localhost:7860**. Use `localhost`, not your machine's network address — browsers only
allow microphone access on `localhost`.

The dashboard shows the answer, the confidence score, the time each stage took, and the exact
passages the answer came from. The **Corpus** tab lists your indexed documents and lets you add or
remove them.

**5. Run the quality tests**

Everyday tests, a few seconds:

```bash
./dev-env/bin/python -m pytest -m "not integration and not ragas_ci" -q
```

Pass or fail is printed in the terminal.

The full quality gate scores answer accuracy and takes about an hour, because a second local AI model
grades every answer. It needs its own environment, since it depends on older libraries than the app:

```bash
python -m venv eval-env
./eval-env/bin/pip install ragas==0.2.15 langchain==0.3.30 langchain-core==0.3.86 \
    langchain-community==0.3.27 langchain-openai==0.2.14 langchain-ollama==0.2.3 pytest requests
ollama pull llama3.1:8b

./eval-env/bin/python -m pytest tests/eval_test.py -m ragas_ci -q -s
```

Leave the engine running while it does. Scores print in the terminal, and a per-question breakdown
is written to `data/eval/last_run_scores.csv`.

### Changing settings

Everything tunable lives in `config/config.yaml` — no code changes needed:

| Setting | What it does |
| --- | --- |
| `retrieval.confidence_threshold` | How sure the system must be before answering. Raise it to refuse more often. |
| `retrieval.top_k_rerank` | How many passages the answer may draw from. |
| `llm.generation_model` | Which Ollama model writes the answers. |
| `cache.enabled`, `cache.max_entries` | Turn caching off, or change how much is remembered. |
| `voice.languages` | Which speech models are used for English and Hinglish. |
| `ingestion.chunk_size` | How large each document piece is. Changing this needs a re-index. |

Save the file and restart the engine for changes to take effect.

### Hinglish setup

```bash
./dev-env/bin/python -c "from huggingface_hub import snapshot_download as d; \
  d(repo_id='shunyalabs/zero-stt-hinglish', local_dir='models/zero-stt-hinglish', \
    allow_patterns=['*.json','*.txt','*.safetensors'])"

./dev-env/bin/ct2-transformers-converter --model models/zero-stt-hinglish \
  --output_dir models/zero-stt-hinglish-ct2 --quantization int8 \
  --copy_files tokenizer.json preprocessor_config.json tokenizer_config.json \
               special_tokens_map.json added_tokens.json normalizer.json vocab.json merges.txt
```

Then choose **Hinglish** in the dashboard's language selector.

### Docker

```bash
docker compose up
```

---

## Future work

- **Hinglish spoken answers.** Hinglish questions work today, but answers are spoken in the English
  voice. A Hinglish voice is wired up and can be switched on in `config.yaml`, though the available
  models are not yet good enough to ship.
- **More languages.** The same pattern — a speech model per language plus translation before search —
  extends to Tamil, Bengali, Spanish and others without touching the search engine.
- **Documents in other languages.** Today the documents themselves must be English. A multilingual
  embedding model would lift that limit.
- **Running on a GPU.** Would take spoken replies from seconds to well under one, and is the only way
  to reach the original sub-second target.
